"""Spec-4: parallel scheduler equivalence — gathering claims/lenses must produce the SAME set of
checks/records as sequential (cap=1), changing only wall-clock and event-append order. FakeLLM is
instant, so the multiset/membership assertions are deterministic."""
import pytest

from valagents.scheduler import run_claim_checks, run
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim
from valagents.config import Config
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


class FakeBackend:
    async def search(self, query, max_results=10):
        return [Article(title="T", summary="The effect exists and has been confirmed here.",
                        url="http://arxiv.org/abs/x1v1", published="2024"),
                Article(title="U", summary="A separate study examined the same phenomenon.",
                        url="http://arxiv.org/abs/x2v1", published="2024")]


def _store():
    return ArtifactStore(IdeaArtifact(
        raw_idea="s", formal_claim=FormalClaim(statement="x", falsifiable=True),
        claim_graph=[AtomicClaim(id="c1", statement="effect exists", type="empirical", load_bearing=True),
                     AtomicClaim(id="c2", statement="recovers the limit", type="mathematical", load_bearing=True),
                     AtomicClaim(id="c3", statement="alpha not saturated", type="empirical", load_bearing=True)]))


def _router(agent, messages):
    if agent == "computation_designer":      # real passing plan -> c2 executor credit
        return ("EXPRESSION: G*M/r**2 | VARIABLES: G,M,r | LIMIT_VARIABLE: r | LIMIT_POINT: oo "
                "| EXPECTED: 0 | EXPECTED_SOURCE: textbook | CONFIRM_IF: limit is 0 | REFUTE_IF: differs")
    if agent == "prover":
        return "DERIVATION: gapped | GAPS: step unverified | FATAL_GAP: no"
    return "CLAIM: c | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | BASIS: unclear"   # grounder/other


def _checks_multiset(store):
    return sorted((c.id, ck.lens, ck.verdict)
                  for c in store.current.claim_graph for ck in c.checks)


def _all_ticks(store):
    return [ck.tick for c in store.current.claim_graph for ck in c.checks]


async def _run_claim_checks(cap):
    cfg = Config(default_model="fake", results_dir="")
    cfg.gate.max_concurrency = cap
    s = _store()
    await run_claim_checks(s, FakeBackend(), FakeLLM(_router), cfg)
    return s


async def test_parallel_claim_checks_equal_sequential():
    par = await _run_claim_checks(8)
    seq = await _run_claim_checks(1)            # cap=1 reproduces sequential exactly
    assert _checks_multiset(par) == _checks_multiset(seq)
    assert _checks_multiset(par)                # non-empty: all three claims actually checked


async def test_parallel_claim_check_ticks_unique():
    s = await _run_claim_checks(8)
    ticks = _all_ticks(s)
    assert len(ticks) == len(set(ticks))        # disjoint tick blocks -> no collisions across claims
