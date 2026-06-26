import json as _json
import pytest
from valagents.scheduler import run_claim_checks, _computations_dir
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim
from valagents.config import Config
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


def _grounder_body(tail: str, payload: dict) -> str:
    return tail + "\n```json\n" + _json.dumps(payload) + "\n```"


def test_computations_dir_grouped_by_run():
    cfg = Config(default_model="fake", results_dir="results")
    # with a run id -> computations live INSIDE the run's own folder (one folder per run holds everything)
    assert _computations_dir(cfg, "run-1", "magnitude") == "results/run-1/computations/magnitude"
    assert _computations_dir(cfg, "run-1", "simulation", "m1") == "results/run-1/computations/simulation/m1"
    assert _computations_dir(cfg, "run-1", "L1") == "results/run-1/computations/L1"
    # no run id -> falls back to the old category-only layout (backward-compatible for callers w/o a run id)
    assert _computations_dir(cfg, None, "magnitude") == "results/computations/magnitude"
    # no results_dir -> sandbox-artifacts disabled
    assert _computations_dir(Config(default_model="fake", results_dir=""), "run-1", "magnitude") is None


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
async def test_mathematical_claim_credited_by_executor_not_prover():
    # PC-1b: a mathematical claim earns its independent credit from a CODE-WITNESSED symbolic check
    # (the legit path the stripped prover say-so used to fake). results_dir="" -> no artifacts written.
    cfg = Config(default_model="fake", results_dir="")
    s = store_with([AtomicClaim(id="m1", statement="recovers Newtonian gravity", type="mathematical")])

    def router(agent, messages):
        if agent == "computation_designer":          # real passing plan: lim GM/r^2 as r->oo = 0
            return ("EXPRESSION: G*M/r**2 | VARIABLES: G,M,r | LIMIT_VARIABLE: r | LIMIT_POINT: oo "
                    "| EXPECTED: 0 | EXPECTED_SOURCE: textbook | CONFIRM_IF: limit is 0 | REFUTE_IF: differs")
        if agent == "prover":
            return "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
        return "CLAIM: m1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | BASIS: no literature"

    await run_claim_checks(s, None, FakeLLM(router), cfg)
    c = s.current.claim_graph[0]
    execs = [ck for ck in c.checks if ck.lens == "executor"]
    assert execs and execs[0].verdict == "pass" and execs[0].independent_sources == 1   # code-witnessed credit
    assert c.status == "pass"
    provers = [ck for ck in c.checks if ck.lens == "prover"]
    assert provers and all(ck.independent_sources == 0 for ck in provers)               # PC-1a: prover self-credits nothing


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
