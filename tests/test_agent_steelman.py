import inspect
import pytest
from valagents.agents.steelman import build_steelman_objection
from valagents.artifact import IdeaArtifact, FormalClaim, SteelmanObjection
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="curl term reduces escape time", falsifiable=True)

TAIL = (
    "STRONGEST_OBJECTION: the curl term vanishes at equilibrium so cannot drive escape"
    " | MECHANISM_OF_FAILURE: rotational flow requires non-equilibrium boundary conditions absent in the model"
    " | THREATENING_RESULT: Kramers' theorem proves escape rate depends only on barrier height not curl"
    " | WHAT_WOULD_KILL_IT: a numerical experiment showing escape time unchanged with and without the curl term"
    " | FAIR_SUMMARY: the proposed mechanism lacks a valid non-equilibrium regime and is likely subsumed by Kramers'"
)


@pytest.mark.asyncio
async def test_build_steelman_objection_returns_populated_model(cfg):
    art = IdeaArtifact(raw_idea="curl reduces escape time", formal_claim=FC)
    result = await build_steelman_objection(art, FakeLLM(lambda a, m: TAIL), cfg)
    assert result is not None
    assert isinstance(result, SteelmanObjection)
    assert "curl" in result.strongest_objection
    assert "Kramers" in result.threatening_result
    assert result.fair_summary != ""


@pytest.mark.asyncio
async def test_build_steelman_objection_returns_none_on_parse_failure(cfg):
    art = IdeaArtifact(raw_idea="curl reduces escape time", formal_claim=FC)
    result = await build_steelman_objection(art, FakeLLM(lambda a, m: "no valid tail here"), cfg)
    assert result is None


def test_steelman_objection_does_not_influence_gate():
    """steelman_objection must NOT appear in _evaluate(); setting it must not change status."""
    src = inspect.getsource(IdeaArtifact._evaluate)
    assert "steelman" not in src, "_evaluate() must not reference steelman_objection"

    base = IdeaArtifact(raw_idea="some idea")
    with_obj = IdeaArtifact(
        raw_idea="some idea",
        steelman_objection=SteelmanObjection(
            strongest_objection="it is wrong",
            mechanism_of_failure="fails here",
            threatening_result="Kramers",
            what_would_kill_it="null experiment",
            fair_summary="skeptic concludes: not proven",
        ),
    )
    assert base.status == with_obj.status
