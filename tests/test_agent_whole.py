import pytest
from valagents.agents.predictor import predict
from valagents.agents.redteam import red_team
from valagents.agents.validation_designer import design_validation
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim, Novelty
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="escape time falls with curl term", falsifiable=True)


@pytest.mark.asyncio
async def test_predict(cfg):
    body = "OBSERVABLE: mean escape time | EFFECT_SIZE: 2x faster | DISCRIMINATES_FROM: vanilla GD | MEASURABLE: yes"
    preds = await predict(FC, Novelty(delta="rotational term"), FakeLLM(lambda a, m: body), cfg)
    assert preds[0].measurable is True and "escape" in preds[0].observable


@pytest.mark.asyncio
async def test_red_team_records_surface_and_landed(cfg):
    body = ("ATTEMPTED: counterexample, magnitude\n"
            "ATTACK: counterexample | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: ok\n"
            "ATTACK: magnitude | SEVERITY: major | STATUS: landed | TARGET: c1 | BASIS: alpha saturates")
    art = IdeaArtifact(raw_idea="s", formal_claim=FC,
                       claim_graph=[AtomicClaim(id="c1", statement="alpha", type="mechanistic")])
    attacks, surface, per_claim = await red_team(art, FakeLLM(lambda a, m: body), cfg)
    assert "magnitude" in surface.attempted
    assert any(a.status == "landed" and a.severity == "major" for a in attacks)
    assert per_claim and per_claim[0][0] == "c1" and per_claim[0][1].verdict == "fail"


@pytest.mark.asyncio
async def test_design_validation(cfg):
    body = ("TEST: escape-time benchmark | CONFIRM_IF: scaling separates | "
            "REFUTE_IF: no separation | COST: low")
    art = IdeaArtifact(raw_idea="s", formal_claim=FC)
    plan = await design_validation(art, FakeLLM(lambda a, m: body), cfg)
    assert plan.cost == "low" and "benchmark" in plan.decisive_test
