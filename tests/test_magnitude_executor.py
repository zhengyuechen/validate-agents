import inspect
from valagents.computation import ComputationPlan
from valagents.config import Config
from valagents.sandbox.executor import run_plan

def cfg():
    return Config(default_model="fake")

def mplan(**kw):
    base = dict(kind="magnitude", comparison_kind="sensitivity_ratio",
                predicted_effect="1e-9", baseline_or_null="0", sensitivity="1e-12",
                sensitivity_source="arXiv:1234", threshold="3")
    base.update(kw)
    return ComputationPlan(**base)

def test_detectable_is_confirm():
    v = run_plan(mplan(), cfg())                 # ratio = 1e-9/1e-12 = 1000 >= 3
    assert v.verdict == "pass" and v.result.matched == "confirm" and v.result.ok

def test_inert_is_refute():
    v = run_plan(mplan(predicted_effect="1e-18"), cfg())   # ratio = 1e-6 < 3
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_missing_sensitivity_source_is_uncertain():       # the anti-laundering core
    v = run_plan(mplan(sensitivity_source=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_missing_threshold_is_uncertain():
    v = run_plan(mplan(threshold=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_zero_sensitivity_is_uncertain():
    v = run_plan(mplan(sensitivity="0"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_dunder_quantity_is_uncertain_not_executed():
    v = run_plan(mplan(predicted_effect="x.__class__"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_run_plan_takes_no_llm():
    assert "llm" not in inspect.signature(run_plan).parameters


def bplan(**kw):
    base = dict(kind="magnitude", comparison_kind="bound_check",
                predicted_effect="1e-3", bound="1e-2", bound_source="PDG2024")
    base.update(kw)
    return ComputationPlan(**base)

def test_bound_complies_is_confirm():
    v = run_plan(bplan(), cfg())                 # 1e-3 <= 1e-2 -> comply
    assert v.verdict == "pass" and v.result.matched == "confirm" and v.result.ok

def test_bound_violates_is_refute():
    v = run_plan(bplan(predicted_effect="1e-1"), cfg())   # 1e-1 > 1e-2 -> violate
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_bound_missing_source_is_uncertain():    # anti-laundering (L2-D2): no source -> never pass/fail
    v = run_plan(bplan(bound_source=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_bound_missing_bound_is_uncertain():
    v = run_plan(bplan(bound=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_bound_dunder_is_uncertain_not_executed():
    v = run_plan(bplan(predicted_effect="x.__class__"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
