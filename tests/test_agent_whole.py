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
    assert per_claim == []  # major is an open objection, not a per-claim refutation


@pytest.mark.asyncio
async def test_design_validation(cfg):
    body = ("TEST: escape-time benchmark | CONFIRM_IF: scaling separates | "
            "REFUTE_IF: no separation | COST: low")
    art = IdeaArtifact(raw_idea="s", formal_claim=FC)
    plan = await design_validation(art, FakeLLM(lambda a, m: body), cfg)
    assert plan.cost == "low" and "benchmark" in plan.decisive_test


@pytest.mark.asyncio
async def test_red_team_reask_path_parses_attacks(cfg):
    """First call has ATTEMPTED but no valid ATTACK rows; second call supplies the rows."""
    first_body = "ATTEMPTED: counterexample, magnitude\nNo valid attack lines here."
    second_body = ("ATTACK: counterexample | SEVERITY: major | STATUS: landed | TARGET: c1 | BASIS: found gap\n"
                   "ATTACK: magnitude | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: small effect")
    bodies = iter([first_body, second_body])
    art = IdeaArtifact(raw_idea="s", formal_claim=FC,
                       claim_graph=[AtomicClaim(id="c1", statement="alpha", type="mechanistic")])
    attacks, surface, per_claim = await red_team(art, FakeLLM(lambda a, m: next(bodies)), cfg)
    assert len(attacks) == 2
    assert any(a.status == "landed" and a.severity == "major" for a in attacks)
    assert per_claim == []


@pytest.mark.asyncio
async def test_red_team_skips_malformed_severity(cfg):
    """A line with bad severity is skipped; a valid line is returned without raising."""
    body = ("ATTACK: x | SEVERITY: critical | STATUS: landed | TARGET: c1 | BASIS: bad sev\n"
            "ATTACK: y | SEVERITY: fatal | STATUS: landed | TARGET: c1 | BASIS: valid")
    art = IdeaArtifact(raw_idea="s", formal_claim=FC,
                       claim_graph=[AtomicClaim(id="c1", statement="alpha", type="mechanistic")])
    attacks, surface, per_claim = await red_team(art, FakeLLM(lambda a, m: body), cfg)
    assert len(attacks) == 1
    assert attacks[0].severity == "fatal"
    assert per_claim and per_claim[0][1].verdict == "fail"
