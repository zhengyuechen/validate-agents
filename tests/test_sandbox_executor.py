from valagents.computation import ComputationPlan
from valagents.config import Config
from valagents.sandbox.executor import run_plan
import inspect

def cfg():
    return Config(default_model="fake")

def plan(**kw):
    base = dict(expression="1/x", variables=["x"], limit_variable="x", limit_point="oo", expected="0")
    base.update(kw)
    return ComputationPlan(**base)

def test_recovers_limit_passes():
    v = run_plan(plan(), cfg())                      # limit(1/x, x, oo) == 0
    assert v.verdict == "pass" and v.result.matched == "confirm" and v.result.ok

def test_newtonian_recovery_passes():
    v = run_plan(plan(expression="G*M/r**2*(1+a/c**2)", variables=["G","M","r","a","c"],
                      limit_variable="c", limit_point="oo", expected="G*M/r**2"), cfg())
    assert v.verdict == "pass"

def test_wrong_limit_fails():
    v = run_plan(plan(expected="1"), cfg())          # limit(1/x,...) == 0, not 1
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_unparseable_or_malicious_expression_is_uncertain_not_executed():
    v = run_plan(plan(expression="__import__('os').system('echo hacked')"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok      # '__' guard + suppressed builtins reject it pre-parse; nothing executes

def test_run_plan_takes_no_llm():                    # F3: code judges, no model in the loop
    sig = inspect.signature(run_plan)
    assert "llm" not in sig.parameters

def test_artifacts_saved(tmp_path):
    v = run_plan(plan(), cfg(), artifacts_dir=str(tmp_path / "c1"))
    assert (tmp_path / "c1" / "plan.json").exists() and (tmp_path / "c1" / "result.json").exists()

def test_disabled_sandbox_is_uncertain():
    c = cfg(); c.sandbox.enabled = False
    assert run_plan(plan(), c).verdict == "uncertain"

def test_dunder_expression_rejected():
    v = run_plan(plan(expression="x.__class__"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok   # rejected pre-parse by '__' guard

def test_subprocess_failure_is_uncertain(monkeypatch):
    import valagents.sandbox.executor as ex
    def boom(*a, **k): raise OSError("boom")
    monkeypatch.setattr(ex.subprocess, "run", boom)
    v = ex.run_plan(plan(), cfg())
    assert v.verdict == "uncertain" and "boom" in v.result.error
