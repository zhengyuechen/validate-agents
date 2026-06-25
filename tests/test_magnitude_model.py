import inspect
from valagents.computation import (ComputationPlan, ComputationResult,
                                   ComputationVerdict, verdict_to_attack, verdict_to_check)

def test_magnitude_plan_omits_symbolic_fields():
    p = ComputationPlan(kind="magnitude", comparison_kind="sensitivity_ratio",
                        predicted_effect="1e-9", baseline_or_null="0", sensitivity="1e-12",
                        sensitivity_source="arXiv:1234", threshold="3")
    assert p.kind == "magnitude" and p.comparison_kind == "sensitivity_ratio"
    assert p.expression == "" and p.expected == ""   # symbolic fields now defaulted

def test_symbolic_plan_still_constructs():            # backward-compat
    p = ComputationPlan(expression="1/x", limit_variable="x", limit_point="oo", expected="0")
    assert p.kind == "symbolic"

def _verdict(matched, discriminating):
    p = ComputationPlan(kind="magnitude", comparison_kind="sensitivity_ratio",
                        predicted_effect="1e-9", baseline_or_null="0", sensitivity="1e-12",
                        sensitivity_source="arXiv:1234", threshold="3", discriminating=discriminating)
    r = ComputationResult(ok=True, computed="ratio=1000", matched=matched)
    v = ComputationVerdict(verdict=("pass" if matched == "confirm" else "fail"), measured="ratio=1000", plan=p, result=r)
    return v

def test_confirm_is_survived_attack():
    a = verdict_to_attack(_verdict("confirm", True), target_claim_id="c1", discriminating=True)
    assert a.type == "magnitude" and a.status == "survived"

def test_refute_discriminating_is_landed_fatal():
    a = verdict_to_attack(_verdict("refute", True), target_claim_id="c1", discriminating=True)
    assert a.status == "landed" and a.severity == "fatal" and a.target_claim_id == "c1"
    assert "sensitivity" in a.basis and "arXiv:1234" in a.basis   # loud source

def test_refute_nondiscriminating_is_landed_major():
    a = verdict_to_attack(_verdict("refute", False), target_claim_id=None, discriminating=False)
    assert a.status == "landed" and a.severity == "major"

def test_verdict_to_attack_takes_no_llm():
    assert "llm" not in inspect.signature(verdict_to_attack).parameters


def _bound_verdict(matched):
    p = ComputationPlan(kind="magnitude", comparison_kind="bound_check",
                        predicted_effect="1e-3", bound="1e-2", bound_source="PDG2024")
    r = ComputationResult(ok=True, computed="predicted=0.001, bound=0.01", matched=matched)
    v = ComputationVerdict(verdict=("pass" if matched == "confirm" else "fail"),
                           measured="predicted=0.001, bound=0.01", plan=p, result=r)
    return v

def test_bound_check_pass_is_independent_sourced_executor_check():
    # Say-so strip (G-D6/G-D10): bound_source alone no longer earns independent_sources credit.
    # With grounding=None (off), independent_sources == 0; the basis still carries the loud source.
    rec = verdict_to_check(_bound_verdict("confirm"))
    assert rec.lens == "executor" and rec.verdict == "pass"
    assert rec.independent_sources == 0 and rec.sources == []     # stripped: grounding=None
    assert "PDG2024" in rec.basis and "bound" in rec.basis        # loud source still in basis

def test_bound_check_fail_maps_to_fail_check():
    rec = verdict_to_check(_bound_verdict("refute"))
    assert rec.verdict == "fail" and rec.independent_sources == 0

def test_verdict_to_check_symbolic_unchanged():
    p = ComputationPlan(expression="1/x", limit_variable="x", limit_point="oo",
                        expected="0", expected_source="textbook")
    r = ComputationResult(ok=True, computed="0", matched="confirm")
    v = ComputationVerdict(verdict="pass", measured="0", plan=p, result=r)
    rec = verdict_to_check(v)
    assert "expected = 0" in rec.basis and rec.sources and rec.sources[0].locator == "textbook"


def test_closest_prior_source_field_exists():
    p = ComputationPlan(kind="magnitude", comparison_kind="discriminating_margin",
                        closest_prior_source="arXiv:5678")
    assert p.closest_prior_source == "arXiv:5678"

def test_discriminating_margin_basis_is_loud_sourced():
    p = ComputationPlan(kind="magnitude", comparison_kind="discriminating_margin",
                        predicted_effect="5e-9", closest_prior_effect="1e-9",
                        closest_prior_source="arXiv:5678", uncertainty="1e-9", threshold="3",
                        discriminating=True)
    r = ComputationResult(ok=True, computed="margin=4", matched="refute")
    v = ComputationVerdict(verdict="fail", measured="margin=4", plan=p, result=r)
    a = verdict_to_attack(v, target_claim_id="c1", discriminating=True)
    assert a.status == "landed" and a.severity == "fatal"
    assert "closest_prior" in a.basis and "arXiv:5678" in a.basis and "margin=4" in a.basis
