"""Task 8: say-so strip + per-path grounding gate tests.
Unit tests for verdict_to_check grounding parameter; end-to-end tests via run_magnitude_checks
with injected fake resolvers. Mirrors test_magnitude_integration.py's harness exactly."""
from __future__ import annotations
import pytest
from valagents.computation import ComputationPlan, ComputationVerdict, ComputationResult, verdict_to_check
from valagents.grounding import GroundingResult
from valagents.scheduler import run_magnitude_checks
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, Prediction, AttackSurface
from valagents.config import Config
from tests.fake_llm import FakeLLM


# ─── Unit: verdict_to_check ────────────────────────────────────────────────

def _bound_verdict(matched="confirm"):
    plan = ComputationPlan(kind="magnitude", comparison_kind="bound_check", bound="1e-3",
                           bound_source="arXiv:2104.01234")
    return ComputationVerdict(verdict="pass", measured="ok", plan=plan,
                              result=ComputationResult(ok=True, matched=matched))


def test_say_so_strip_no_grounding():
    rec = verdict_to_check(_bound_verdict(), grounding=None)        # grounding OFF
    assert rec.independent_sources == 0 and rec.sources == []       # the auto-credit is stripped


def test_grounding_supports_earns_credit():
    g = GroundingResult("supports", quote="…1e-3 µB…", converted_value=1e-3)
    rec = verdict_to_check(_bound_verdict(), grounding=g)
    assert rec.independent_sources == 1 and len(rec.sources) == 1


def test_symbolic_credit_unchanged():
    sym = ComputationVerdict(verdict="pass", measured="0", plan=ComputationPlan(kind="symbolic", expected="0"),
                             result=ComputationResult(ok=True, matched="confirm"))
    rec = verdict_to_check(sym)                                     # no grounding arg
    assert rec.independent_sources == 1                            # symbolic path untouched


def test_grounding_contradicts_still_sets_zero_credit():
    """A contradicts result on bound_check should yield 0 credit (suppress happens upstream in run_magnitude_checks,
    but if verdict_to_check is called directly with contradicts, the result must be 0)."""
    g = GroundingResult("contradicts", quote="…10 µB…", converted_value=10e-3)
    rec = verdict_to_check(_bound_verdict(), grounding=g)
    assert rec.independent_sources == 0


def test_grounding_inconclusive_sets_zero_credit():
    g = GroundingResult("inconclusive", quote="…1e-3 µB…", converted_value=1e-3)
    rec = verdict_to_check(_bound_verdict(), grounding=g)
    assert rec.independent_sources == 0 and rec.sources == []


def test_grounding_supports_attaches_source_with_locator():
    g = GroundingResult("supports", quote="…1e-3 µB…", converted_value=1e-3)
    rec = verdict_to_check(_bound_verdict(), grounding=g)
    assert rec.sources[0].locator == "arXiv:2104.01234"
    assert rec.sources[0].relation == "independent"


def test_bound_check_fail_verdict_no_credit():
    """Even with grounding supports, a fail verdict on bound_check keeps the verdict='fail' but
    grounding-supports still sets independent_sources=1 (the credit is about the grounding, not the verdict)."""
    plan = ComputationPlan(kind="magnitude", comparison_kind="bound_check", bound="1e-3",
                           bound_source="arXiv:2104.01234")
    v = ComputationVerdict(verdict="fail", measured="ok", plan=plan,
                           result=ComputationResult(ok=False, matched="refute"))
    g = GroundingResult("supports", quote="…1e-3 µB…", converted_value=1e-3)
    rec = verdict_to_check(v, grounding=g)
    # verdict propagated as-is; credit from grounding
    assert rec.verdict == "fail"
    assert rec.independent_sources == 1


# ─── End-to-end: run_magnitude_checks ─────────────────────────────────────

def cfg():
    return Config(default_model="fake")


def store_with_prediction():
    art = IdeaArtifact(raw_idea="seed", formal_claim=FormalClaim(statement="x", falsifiable=True),
                       predictions=[Prediction(observable="shift", effect_size="1e-3",
                                               discriminates_from="GR", measurable=True)],
                       attack_surface=AttackSurface(attempted=["counterexample"]))
    return ArtifactStore(art)


BOUND_OK = ("COMPARISON_KIND: bound_check | PREDICTED_EFFECT: 1e-3 | BOUND: 1e-2 "
            "| BOUND_SOURCE: arXiv:2104.01234 "
            "| SOURCE_QUANTITY: coupling constant | CLAIM_CONDITIONS: all energies | SOURCE_UNIT: dimensionless "
            "| CONFIRM_IF: p<=bound | REFUTE_IF: p>bound")

INERT = ("COMPARISON_KIND: sensitivity_ratio | PREDICTED_EFFECT: 1e-18 | BASELINE_OR_NULL: 0 "
         "| SENSITIVITY: 1e-12 | SENSITIVITY_SOURCE: arXiv:1234 "
         "| SOURCE_QUANTITY: noise floor | CLAIM_CONDITIONS: T < 1 K | SOURCE_UNIT: T/sqrt(Hz) "
         "| THRESHOLD: 3 | CONFIRM_IF: ratio>=3 | REFUTE_IF: ratio<3")
DETECT = INERT.replace("PREDICTED_EFFECT: 1e-18", "PREDICTED_EFFECT: 1e-9")


def router(body):
    return FakeLLM(lambda a, m: body if a == "magnitude_designer" else "")


# Sentinel resolver (non-None) so grounding is "on" — the pipeline is short-circuited
# by monkeypatching ground_plan directly in scheduler's local import scope.
class _SentinelResolver:
    """Non-None resolver that signals grounding is ON; never called in monkeypatched tests."""
    async def fetch(self, locator: str):
        raise AssertionError("fetch() should not be called in monkeypatched tests")


async def test_resolver_none_bound_check_independent_sources_zero():
    """resolver=None -> no grounding -> independent_sources == 0 (say-so strip)."""
    s = store_with_prediction()
    await run_magnitude_checks(s, router(BOUND_OK), cfg(), resolver=None)
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert bnd, "expected a BND claim to be injected"
    # With resolver=None, independent_sources must be 0 (stripped)
    check = bnd[0].checks[0]
    assert check.independent_sources == 0
    assert check.sources == []


async def test_grounding_supports_bound_check_earns_credit(monkeypatch):
    """resolver+LLM grounding to supports -> independent_sources == 1."""
    import valagents.agents.value_grounder as vg
    supports_result = GroundingResult("supports", quote="…1e-2 µB…", converted_value=1e-2)
    monkeypatch.setattr(vg, "ground_plan", lambda plan, resolver, llm, cfg: _async_return(supports_result))
    s = store_with_prediction()
    await run_magnitude_checks(s, router(BOUND_OK), cfg(), resolver=_SentinelResolver())
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert bnd, "expected a BND claim"
    check = bnd[0].checks[0]
    assert check.independent_sources == 1
    assert len(check.sources) == 1


async def test_grounding_contradicts_bound_check_skips_claim(monkeypatch):
    """resolver+LLM grounding to contradicts -> BND claim is NOT injected (suppressed)."""
    import valagents.agents.value_grounder as vg
    contradicts_result = GroundingResult("contradicts", quote="…1e2 µB…", converted_value=1e2)
    monkeypatch.setattr(vg, "ground_plan", lambda plan, resolver, llm, cfg: _async_return(contradicts_result))
    s = store_with_prediction()
    await run_magnitude_checks(s, router(BOUND_OK), cfg(), resolver=_SentinelResolver())
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert not bnd, "contradicts grounding must suppress the BND claim"


async def test_grounding_contradicts_sensitivity_no_attack(monkeypatch):
    """resolver+LLM contradicts for sensitivity_ratio -> no magnitude attack added."""
    import valagents.agents.value_grounder as vg
    contradicts_result = GroundingResult("contradicts", quote="…1e-6 T/sqrt(Hz)…", converted_value=1e-6)
    monkeypatch.setattr(vg, "ground_plan", lambda plan, resolver, llm, cfg: _async_return(contradicts_result))
    s = store_with_prediction()
    await run_magnitude_checks(s, router(DETECT), cfg(), resolver=_SentinelResolver())
    mags = [a for a in s.current.attacks if a.type == "magnitude"]
    assert not mags, "contradicts-grounded sensitivity must suppress the attack"


async def test_symbolic_verdict_to_check_independent_sources_unchanged():
    """The strip must NOT leak into the symbolic branch: a symbolic pass always gets independent_sources=1."""
    plan = ComputationPlan(kind="symbolic", expected="0", expected_source="arXiv:9999")
    v = ComputationVerdict(verdict="pass", measured="0", plan=plan,
                           result=ComputationResult(ok=True, matched="confirm"))
    rec = verdict_to_check(v)
    assert rec.independent_sources == 1
    # also works with explicit grounding=None (should not affect symbolic branch)
    rec2 = verdict_to_check(v, grounding=None)
    assert rec2.independent_sources == 1


async def _async_return(value):
    return value
