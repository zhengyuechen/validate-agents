import json as _json
import pytest
from valagents.scheduler import run_claim_checks
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


def _grounder_body(tail: str, payload: dict) -> str:
    return tail + "\n```json\n" + _json.dumps(payload) + "\n```"


class FakeBackend:
    def __init__(self, articles): self.articles = articles
    async def search(self, query, max_results=5): return self.articles


def store_with(claims):
    return ArtifactStore(IdeaArtifact(raw_idea="s",
                         formal_claim=FormalClaim(statement="x", falsifiable=True), claim_graph=claims))


@pytest.mark.asyncio
async def test_empirical_claim_grounded_and_exhausted(cfg):
    # Tier-2: a code-witnessed on-property quote from a retrieved article → pass.
    # Two articles so "exists" token (in A1 only) stays below saturation threshold
    # (1/2=0.5 < 0.6) and remains distinctive.
    s = store_with([AtomicClaim(id="c1", statement="effect exists", type="empirical")])
    arts = [
        Article(title="T", summary="The effect exists and has been confirmed here.",
                url="https://example.com/u", published="2024"),
        Article(title="U", summary="A separate study examined the same phenomenon.",
                url="https://example.com/v", published="2024"),
    ]
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: ok"
    payload = {"asserted_property": "exists", "subject_phrase": "effect",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The effect exists and has been confirmed here."}]}
    body = _grounder_body(tail, payload)
    backend = FakeBackend(arts)
    await run_claim_checks(s, backend, FakeLLM(lambda a, m: body), cfg)
    c = s.current.claim_graph[0]
    assert c.status == "pass" and c.exhausted is True


@pytest.mark.asyncio
async def test_fanout_runs_more_lenses_on_uncertain_loadbearing(cfg):
    # empirical matrix = [grounder] only; grounder returns uncertain → fan-out adds the
    # diverse prover lens to reach fanout_N=2.  FakeLLM routes by agent name.
    s = store_with([AtomicClaim(id="c1", statement="alpha not saturated", type="empirical")])
    grounder_body = "CLAIM: c1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: unclear"
    prover_body = "DERIVATION: gapped | GAPS: c1 | FATAL_GAP: no"

    def router(agent, messages):
        if agent == "grounder":
            return grounder_body
        return prover_body

    llm = FakeLLM(router)
    await run_claim_checks(s, None, llm, cfg)
    c = s.current.claim_graph[0]
    assert c.status == "uncertain"
    assert len(c.checks) >= cfg.gate.fanout_N          # fan-out met before finalize
    assert any(ck.lens == "prover" for ck in c.checks) # diverse fan-out lens actually ran
