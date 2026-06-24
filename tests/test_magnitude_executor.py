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


def dplan(**kw):
    base = dict(kind="magnitude", comparison_kind="discriminating_margin",
                predicted_effect="5e-9", closest_prior_effect="1e-9",
                closest_prior_source="arXiv:5678", uncertainty="1e-9", threshold="3")
    base.update(kw)
    return ComputationPlan(**base)

def test_discriminating_clears_is_confirm():
    v = run_plan(dplan(), cfg())                 # |5e-9-1e-9|/1e-9 = 4 >= 3 -> distinguishable
    assert v.verdict == "pass" and v.result.matched == "confirm"

def test_indistinguishable_is_refute():
    v = run_plan(dplan(predicted_effect="2e-9"), cfg())   # |2e-9-1e-9|/1e-9 = 1 < 3
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_discriminating_missing_source_is_uncertain():    # L2-D10 anti-laundering of the alternative
    v = run_plan(dplan(closest_prior_source=""), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_discriminating_zero_uncertainty_is_uncertain():
    v = run_plan(dplan(uncertainty="0"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_discriminating_dunder_is_uncertain_not_executed():
    v = run_plan(dplan(closest_prior_effect="x.__class__"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok


def test_runner_rejects_pipe_spilled_source_backstop():   # defense-in-depth: sandbox catches a spilled source
    v = run_plan(bplan(bound_source="| CONFIRM_IF: p<=bound"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
    v2 = run_plan(dplan(closest_prior_source="| UNCERTAINTY: 1e-9"), cfg())
    assert v2.verdict == "uncertain" and not v2.result.ok
