"""Tests for R4 rigor fields: assumptions ledger, detectability, best-alternative/inferential validation."""
import inspect
import pytest

from valagents.agents.completer import complete_idea
from valagents.agents.predictor import predict
from valagents.agents.validation_designer import design_validation
from valagents.artifact import (
    Assumption,
    AtomicClaim,
    FormalClaim,
    IdeaArtifact,
    IdeaCompletion,
    Novelty,
    Prediction,
    ValidationPlan,
)
from tests.fake_llm import FakeLLM


ART = IdeaArtifact(
    raw_idea="a seed idea",
    formal_claim=FormalClaim(statement="x changes y", falsifiable=True),
    claim_graph=[AtomicClaim(id="c1", statement="x changes y", type="mechanistic")],
)

FC = FormalClaim(statement="x changes y faster", falsifiable=True)


# ---- R4.1: Assumptions ledger ----

@pytest.mark.asyncio
async def test_completer_parses_assumptions_including_novel_load_bearing(cfg):
    body = (
        "COMPLETION_STATUS: completed_candidate | COMPLETED_IDEA: x changes y via m | "
        "MECHANISM: m bridges x and y | WEAKEST_LINK: c1\n"
        "ASSUMPTION: standard background fact | STATUS: standard\n"
        "ASSUMPTION: contested prior assumption | STATUS: contested\n"
        "ASSUMPTION: new required assertion | STATUS: novel_load_bearing"
    )
    out = await complete_idea(ART, FakeLLM(lambda a, m: body), cfg)
    assert out is not None
    assert len(out.assumptions) == 3
    assert out.assumptions[0].text == "standard background fact"
    assert out.assumptions[0].status == "standard"
    assert out.assumptions[1].status == "contested"
    assert out.assumptions[2].text == "new required assertion"
    assert out.assumptions[2].status == "novel_load_bearing"


@pytest.mark.asyncio
async def test_completer_no_assumption_lines_yields_empty_list(cfg):
    body = (
        "COMPLETION_STATUS: incomplete | COMPLETED_IDEA: partial | "
        "MECHANISM: unknown | WEAKEST_LINK: c1"
    )
    out = await complete_idea(ART, FakeLLM(lambda a, m: body), cfg)
    assert out is not None
    assert out.assumptions == []


def test_assumption_model_defaults():
    a = Assumption()
    assert a.text == ""
    assert a.status == "standard"


def test_idea_completion_assumptions_field_is_list_of_assumption():
    comp = IdeaCompletion(
        status="completed_candidate",
        completed_idea="x",
        mechanism="m",
        assumptions=[
            Assumption(text="a1", status="standard"),
            Assumption(text="a2", status="novel_load_bearing"),
        ],
        weakest_link="c1",
    )
    assert len(comp.assumptions) == 2
    assert comp.assumptions[1].status == "novel_load_bearing"


# ---- R4.2: Prediction detectability ----

@pytest.mark.asyncio
async def test_prediction_detectable_no_parses(cfg):
    body = (
        "OBSERVABLE: x measurement | EFFECT_SIZE: 0.01% | "
        "DISCRIMINATES_FROM: null | MEASURABLE: yes | DETECTABLE: no"
    )
    preds = await predict(FC, Novelty(delta="tiny effect"), FakeLLM(lambda a, m: body), cfg)
    assert len(preds) == 1
    assert preds[0].detectable == "no"


@pytest.mark.asyncio
async def test_prediction_detectable_unclear_default(cfg):
    body = (
        "OBSERVABLE: vague signal | EFFECT_SIZE: unknown | "
        "DISCRIMINATES_FROM: baseline | MEASURABLE: no | DETECTABLE: unclear"
    )
    preds = await predict(FC, None, FakeLLM(lambda a, m: body), cfg)
    assert preds[0].detectable == "unclear"


def test_prediction_detectable_field_default():
    p = Prediction(observable="o")
    assert p.detectable == "unclear"


# ---- R4.3: ValidationPlan discriminates_from + inferential_standard ----

@pytest.mark.asyncio
async def test_validation_plan_parses_discriminates_from_and_inferential_standard(cfg):
    body = (
        "TEST: benchmark run | CONFIRM_IF: delta > threshold | REFUTE_IF: delta < threshold | "
        "DISCRIMINATES_FROM: best prior model | INFERENTIAL_STANDARD: n=200 power=0.9 pre-registered | "
        "COST: medium"
    )
    plan = await design_validation(ART, FakeLLM(lambda a, m: body), cfg)
    assert plan is not None
    assert plan.discriminates_from == "best prior model"
    assert "power=0.9" in plan.inferential_standard
    assert plan.cost == "medium"


def test_validation_plan_new_fields_default_empty():
    plan = ValidationPlan(decisive_test="t")
    assert plan.discriminates_from == ""
    assert plan.inferential_standard == ""


# ---- _evaluate purity: none of the new display fields are read by the gate ----

def test_evaluate_does_not_read_new_rigor_fields():
    source = inspect.getsource(IdeaArtifact._evaluate)
    for forbidden in ("assumptions", "detectable", "discriminates_from", "inferential_standard"):
        assert forbidden not in source, (
            f"_evaluate must not read '{forbidden}' — it is a narrative/display field only"
        )
