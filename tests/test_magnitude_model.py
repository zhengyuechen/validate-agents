import inspect
from valagents.computation import (ComputationPlan, ComputationResult,
                                   ComputationVerdict, verdict_to_attack)

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
