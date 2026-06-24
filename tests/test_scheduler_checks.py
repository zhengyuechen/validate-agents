import pytest
from valagents.scheduler import run_claim_checks
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


class FakeBackend:
    def __init__(self, articles): self.articles = articles
    async def search(self, query, max_results=5): return self.articles


def store_with(claims):
    return ArtifactStore(IdeaArtifact(raw_idea="s",
                         formal_claim=FormalClaim(statement="x", falsifiable=True), claim_graph=claims))


@pytest.mark.asyncio
async def test_empirical_claim_grounded_and_exhausted(cfg):
    # A real backend returning one article with a URL lets the cap pass through:
    # matched_independent=1, min(1,1)=1 → map_support_to_verdict("supported",1) → "pass".
    s = store_with([AtomicClaim(id="c1", statement="effect exists", type="empirical")])
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | SOURCES: A1 | BASIS: ok"
    backend = FakeBackend([Article(title="T", summary="s", url="https://example.com/u")])
    await run_claim_checks(s, backend, FakeLLM(lambda a, m: body), cfg)
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
