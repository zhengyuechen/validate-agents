"""Tests for R3: limit-recovery claims checked by the Prover and gated."""
from __future__ import annotations
import inspect
import pytest

from valagents.artifact import (
    IdeaArtifact, AtomicClaim, FormalClaim, KnownLimit,
)
from valagents.artifact import IdeaArtifact as _IA
from valagents.scheduler import inject_limit_checks
from valagents.store import ArtifactStore
from tests.fake_llm import FakeLLM


def _make_store(known_limits: list[KnownLimit], extra_claims: list[AtomicClaim] | None = None):
    art = IdeaArtifact(
        raw_idea="seed",
        formal_claim=FormalClaim(statement="formal", falsifiable=True),
        known_limits=known_limits,
        claim_graph=extra_claims or [],
    )
    return ArtifactStore(art)


def _prover_llm(derivation: str, gaps: str, fatal_gap: str) -> FakeLLM:
    body = f"DERIVATION: {derivation} | GAPS: {gaps} | FATAL_GAP: {fatal_gap}"
    return FakeLLM(lambda agent, messages: body)


@pytest.mark.asyncio
async def test_complete_reduction_claim_pass(cfg):
    """A code-witnessed symbolic check → injected claim has status==pass (PC-1b: credit is the EXECUTED
    symbolic check, not the prover's say-so 'complete'). origin==limit_recovery, load_bearing."""
    limits = [KnownLimit(limit="thermodynamic limit recovers Gibbs ensemble")]
    store = _make_store(limits)

    def route(agent, messages):
        if agent == "computation_designer":      # a real passing plan: lim GM/r^2 as r->oo = 0
            return ("EXPRESSION: G*M/r**2 | VARIABLES: G,M,r | LIMIT_VARIABLE: r | LIMIT_POINT: oo "
                    "| EXPECTED: 0 | EXPECTED_SOURCE: textbook | CONFIRM_IF: limit is 0 | REFUTE_IF: differs")
        return "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    llm = FakeLLM(route)

    await inject_limit_checks(store, llm, cfg, tick=0)

    art = store.current
    limit_claims = [c for c in art.claim_graph if c.origin == "limit_recovery"]
    assert len(limit_claims) == 1
    c = limit_claims[0]
    assert c.id == "L1"
    assert c.status == "pass"
    assert c.origin == "limit_recovery"
    assert c.load_bearing is True
    assert c.exhausted is True


@pytest.mark.asyncio
async def test_contradiction_claim_fail_artifact_refuted(cfg):
    """A fatal gap → injected claim status==fail → artifact status==refuted."""
    limits = [KnownLimit(limit="must recover Boltzmann in high-T limit")]
    store = _make_store(limits)
    llm = _prover_llm("gapped", "CONTRADICTION: violates 2nd law", "yes")

    await inject_limit_checks(store, llm, cfg, tick=0)

    art = store.current
    limit_claims = [c for c in art.claim_graph if c.origin == "limit_recovery"]
    assert len(limit_claims) == 1
    assert limit_claims[0].status == "fail"
    assert art.status == "refuted"


@pytest.mark.asyncio
async def test_gapped_claim_uncertain_artifact_needs_experiment(cfg):
    """A gapped (non-fatal) derivation → claim uncertain → artifact needs_experiment."""
    limits = [KnownLimit(limit="must recover mean-field in d→∞")]
    store = _make_store(limits)
    llm = _prover_llm("gapped", "step 3 unverified", "no")

    await inject_limit_checks(store, llm, cfg, tick=0)

    art = store.current
    limit_claims = [c for c in art.claim_graph if c.origin == "limit_recovery"]
    assert len(limit_claims) == 1
    assert limit_claims[0].status == "uncertain"
    assert art.status == "needs_experiment"


@pytest.mark.asyncio
async def test_cap_at_three(cfg):
    """5 known limits → exactly 3 injected claims."""
    limits = [
        KnownLimit(limit=f"limit {i}") for i in range(5)
    ]
    store = _make_store(limits)
    llm = _prover_llm("complete", "none", "no")

    await inject_limit_checks(store, llm, cfg, tick=0)

    art = store.current
    limit_claims = [c for c in art.claim_graph if c.origin == "limit_recovery"]
    assert len(limit_claims) == 3

    cap_events = [e for e in store.events if e.get("event") == "limit_checks_capped"]
    assert len(cap_events) == 1
    assert cap_events[0]["total"] == 5
    assert cap_events[0]["kept"] == 3


def test_origin_not_in_evaluate():
    """'origin' must not appear in IdeaArtifact._evaluate source (gate must not read it)."""
    src = inspect.getsource(IdeaArtifact._evaluate)
    assert "origin" not in src


@pytest.mark.asyncio
async def test_no_known_limits_is_noop(cfg):
    """Empty known_limits → no claims injected, no errors."""
    store = _make_store([])
    llm = FakeLLM(lambda a, m: "DERIVATION: complete | GAPS: none | FATAL_GAP: no")

    await inject_limit_checks(store, llm, cfg, tick=0)

    art = store.current
    assert art.claim_graph == []
    assert llm.calls == []
