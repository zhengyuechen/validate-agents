import pytest
from valagents.scheduler import run_claim_checks
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim
from tests.fake_llm import FakeLLM


def store_with(claims):
    return ArtifactStore(IdeaArtifact(raw_idea="s",
                         formal_claim=FormalClaim(statement="x", falsifiable=True), claim_graph=claims))


@pytest.mark.asyncio
async def test_empirical_claim_grounded_and_exhausted(cfg):
    s = store_with([AtomicClaim(id="c1", statement="effect exists", type="empirical")])
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1,A2 | BASIS: ok"
    await run_claim_checks(s, None, FakeLLM(lambda a, m: body), cfg)
    c = s.current.claim_graph[0]
    assert c.status == "pass" and c.exhausted is True


@pytest.mark.asyncio
async def test_fanout_runs_more_lenses_on_uncertain_loadbearing(cfg):
    # grounder returns uncertain → fan-out triggers a second diverse run; count lens calls
    s = store_with([AtomicClaim(id="c1", statement="alpha not saturated", type="mechanistic")])
    body = "CLAIM: c1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: unclear\n" \
           "DERIVATION: gapped | GAPS: c1 | FATAL_GAP: no"
    llm = FakeLLM(lambda a, m: body)
    await run_claim_checks(s, None, llm, cfg)
    c = s.current.claim_graph[0]
    assert c.status == "uncertain"
    assert len(c.checks) >= cfg.gate.fanout_N    # fan-out met before finalize
