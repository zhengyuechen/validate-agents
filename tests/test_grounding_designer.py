import json
from valagents.computation import ComputationPlan
from valagents.agents.magnitude_designer import design_magnitude
from valagents.prompts import MAGNITUDE_DESIGNER
from valagents.artifact import IdeaArtifact, FormalClaim, Prediction, AttackSurface
from valagents.config import Config
from tests.fake_llm import FakeLLM


def test_plan_has_grounding_fields():
    p = ComputationPlan(kind="magnitude", comparison_kind="bound_check",
                        source_quantity="Yb moment", claim_conditions="T < 1 K", source_unit="µB")
    assert p.source_quantity == "Yb moment" and p.claim_conditions == "T < 1 K" and p.source_unit == "µB"


def test_prompt_teaches_grounding_fields():
    for tok in ("SOURCE_QUANTITY", "CLAIM_CONDITIONS", "SOURCE_UNIT", "resolvable"):
        assert tok in MAGNITUDE_DESIGNER


BOUND_WITH_GROUNDING = (
    "COMPARISON_KIND: bound_check | PREDICTED_EFFECT: 1e-3 | BOUND: 1e-2 "
    "| BOUND_SOURCE: arXiv:2301.00000 "
    "| SOURCE_QUANTITY: Yb magnetic moment | CLAIM_CONDITIONS: T < 1 K | SOURCE_UNIT: µB "
    "| CONFIRM_IF: p<=bound | REFUTE_IF: p>bound"
)


def _art():
    return IdeaArtifact(
        raw_idea="seed",
        formal_claim=FormalClaim(statement="Yb moment < bound", falsifiable=True),
        predictions=[Prediction(observable="moment", effect_size="1e-3",
                                discriminates_from="", measurable=True)],
        attack_surface=AttackSurface(attempted=[]),
    )


def _cfg():
    return Config(default_model="fake")


def _router(body):
    return FakeLLM(lambda agent, messages: body if agent == "magnitude_designer" else "")


async def test_design_magnitude_populates_grounding_fields():
    art = _art()
    plan = await design_magnitude(art.predictions[0], art, _router(BOUND_WITH_GROUNDING), _cfg())
    assert plan is not None and plan.comparison_kind == "bound_check"
    assert plan.source_quantity == "Yb magnetic moment"
    assert plan.claim_conditions == "T < 1 K"
    assert plan.source_unit == "µB"
