import pytest

from valagents.agents.completer import complete_idea
from valagents.agents.convincing_case import build_convincing_case
from valagents.agents.known_limits import check_known_limits
from valagents.agents.positioning import position_prior_art
from valagents.agents.theory_bridge import build_theory_bridge
from valagents.artifact import AtomicClaim, FormalClaim, IdeaArtifact
from tests.fake_llm import FakeLLM


ART = IdeaArtifact(
    raw_idea="new idea",
    formal_claim=FormalClaim(statement="x changes y", falsifiable=True),
    claim_graph=[AtomicClaim(id="c1", statement="x changes y", type="mechanistic")],
)


@pytest.mark.asyncio
async def test_complete_idea(cfg):
    body = (
        "COMPLETION_STATUS: completed_candidate | COMPLETED_IDEA: x changes y by mechanism m | "
        "MECHANISM: m links x to y | WEAKEST_LINK: c1\n"
        "ASSUMPTION: a1 | STATUS: standard\n"
        "ASSUMPTION: a2 | STATUS: contested"
    )
    out = await complete_idea(ART, FakeLLM(lambda a, m: body), cfg)
    assert out.status == "completed_candidate"
    assert len(out.assumptions) == 2
    assert out.assumptions[0].text == "a1"
    assert out.assumptions[0].status == "standard"
    assert out.assumptions[1].text == "a2"
    assert out.assumptions[1].status == "contested"
    assert out.weakest_link == "c1"


@pytest.mark.asyncio
async def test_theory_bridge(cfg):
    body = (
        "THEORY_FAMILY: quantum foundations | NEAREST_THEORIES: CSL, GRW | "
        "EXTENDS: collapse models | CHALLENGES: unitary-only accounts | "
        "RECOVERS_KNOWN_LIMITS: standard QM when coupling vanishes | "
        "DEPARTURE_POINT: objective collapse channel | EXPERT_TRANSLATION: an added Lindblad term"
    )
    out = await build_theory_bridge(ART, FakeLLM(lambda a, m: body), cfg)
    assert out.nearest_theories == ["CSL", "GRW"]
    assert "Lindblad" in out.expert_translation


@pytest.mark.asyncio
async def test_prior_art_positioning(cfg):
    body = (
        "CLOSEST_PRIOR: CSL | SIMILARITY: stochastic collapse | "
        "DIFFERENCE: gravity-triggered channel | WHAT_IS_NEW: new trigger condition | "
        "MUST_CITE: CSL, Diosi-Penrose"
    )
    out = await position_prior_art(ART, FakeLLM(lambda a, m: body), cfg)
    assert out.closest_prior == "CSL"
    assert out.must_cite == ["CSL", "Diosi-Penrose"]


@pytest.mark.asyncio
async def test_known_limits(cfg):
    body = (
        "LIMIT: zero coupling | RECOVERED: yes | FAILURE_IF_NOT: violates standard limit | "
        "REPAIR_NEEDED: derive zero-coupling reduction\n"
        "LIMIT: no-signaling | RECOVERED: unclear | FAILURE_IF_NOT: superluminal channel | "
        "REPAIR_NEEDED: add no-signaling proof"
    )
    out = await check_known_limits(ART, FakeLLM(lambda a, m: body), cfg)
    assert [item.recovered for item in out] == ["yes", "unclear"]
    assert "no-signaling" in out[1].limit


@pytest.mark.asyncio
async def test_convincing_case(cfg):
    body = (
        "ELEVATOR_VERSION: x is a testable extension | "
        "TECHNICAL_VERSION: x adds an operator to the baseline equation | "
        "WHY_EXISTING_THEORY_LEAVES_ROOM: parameter is unconstrained | "
        "WHY_PLAUSIBLE: preserves the known limit | SKEPTIC_TESTS: recover limit, compare baseline"
    )
    out = await build_convincing_case(ART, FakeLLM(lambda a, m: body), cfg)
    assert out.skeptic_tests == ["recover limit", "compare baseline"]
    assert "operator" in out.technical_version
