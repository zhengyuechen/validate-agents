from valagents.computation import ComputationPlan
from valagents.config import Config, SimCfg
from valagents.sandbox.executor import run_plan

def cfg():
    return Config(default_model="fake")

def splan(**kw):
    base = dict(kind="simulation", primitive="ode_integrate", state_vars=["x"],
                rhs={"x": "-a*x"}, params={"a": "1.0"}, init={"x": "1.0"},
                t_span=["0", "5"], dt="0.01",
                param_sweep={"a": ["0.8", "1.2", "5"]},
                observable={"name": "final_value", "var": "x", "window_frac": "0.1"},
                sim_criterion={"op": "le", "threshold": ["0.2"]}, robust_frac="0.8",
                max_steps=2000, max_grid_points=50, max_state_vars=4, max_expr_nodes=50)
    base.update(kw)
    return ComputationPlan(**base)

def test_robust_pass():
    # every a in [0.8,1.2] decays x(5) well below 0.2 -> criterion holds across the whole grid
    v = run_plan(splan(), cfg())
    assert v.verdict == "pass" and v.result.matched == "confirm"
    assert "robust" in v.measured

def test_knife_edge_fail():
    # criterion final_value <= 0.2 fails for slow decay; make it fail across the grid
    v = run_plan(splan(sim_criterion={"op": "le", "threshold": ["1e-6"]}), cfg())   # never reached
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_missing_field_uncertain():
    v = run_plan(splan(rhs={}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_nonpositive_cap_uncertain():
    v = run_plan(splan(max_steps=0), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_cap_over_ceiling_uncertain():
    v = run_plan(splan(max_grid_points=10_000), cfg())   # exceeds SimCfg ceiling 400
    assert v.verdict == "uncertain" and not v.result.ok

def test_single_point_grid_uncertain():
    v = run_plan(splan(param_sweep={}, init_sweep={}), cfg())   # grid size 1 < min_grid_points 4
    assert v.verdict == "uncertain" and not v.result.ok

def test_total_work_cap_uncertain():
    # grid 400 * n_steps 10_000 = 4_000_000 > max_total_steps 2_000_000; EVERY per-axis cap is within ceiling
    # (so it fails on the total-work cap, not on a per-axis breach).
    v = run_plan(splan(param_sweep={"a": ["0", "1", "400"]}, t_span=["0", "10"], dt="0.001",
                       max_steps=200_000, max_grid_points=400), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_dunder_rhs_uncertain():
    v = run_plan(splan(rhs={"x": "x.__class__"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_blowup_uncertain():
    v = run_plan(splan(rhs={"x": "x**2"}, init={"x": "10.0"}, t_span=["0", "50"]), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_determinism():
    a = run_plan(splan(), cfg())
    b = run_plan(splan(), cfg())
    assert a.measured == b.measured and a.verdict == b.verdict

def test_missing_ceilings_fail_closed():
    # the two-layer cap guarantee must not silently depend on run_plan injecting ceilings
    from valagents.sandbox import runner
    plan = splan().model_dump()        # a valid plan dict, but with NO "_sim_ceilings" key
    out = runner._run_simulation(plan)
    assert out["ok"] is False and "ceiling" in out["error"].lower()

def test_reserved_name_shadowing_uncertain():
    # a state var named "E" would shadow Euler's number -> reject fail-closed, not silently mis-simulate
    v = run_plan(splan(state_vars=["E"], rhs={"E": "-a*E"}, init={"E": "1.0"},
                       observable={"name": "final_value", "var": "E", "window_frac": "0.1"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_single_arm_backward_compat():
    # no null_overrides -> v1 single-arm; computed keeps the v1 "robust: ... pass" format
    v = run_plan(splan(), cfg())
    assert v.verdict == "pass" and "robust" in v.measured and "pass" in v.measured
    assert "discriminating" not in v.measured

def test_discriminating_pass():
    # mechanism (a in [0.8,1.2]) decays x below 0.2; null (a=0) leaves x at 1.0 (criterion fails) -> discriminate
    v = run_plan(splan(null_overrides={"a": "0"}), cfg())
    assert v.verdict == "pass" and v.result.matched == "confirm"
    assert "discriminating" in v.measured

def test_behavior_without_mechanism_refutes():
    # loose criterion (<=2.0) is met in BOTH arms (x stays <2 even with a=0) -> not attributable -> refute
    v = run_plan(splan(null_overrides={"a": "0"}, sim_criterion={"op": "le", "threshold": ["2.0"]}), cfg())
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_behavior_absent_with_mechanism_refutes():
    # tight criterion (<=1e-9) is NOT met even in the mechanism arm -> not discriminating -> refute
    v = run_plan(splan(null_overrides={"a": "0"}, sim_criterion={"op": "le", "threshold": ["1e-9"]}), cfg())
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_null_override_undeclared_param_uncertain():
    v = run_plan(splan(null_overrides={"zzz": "0"}), cfg())   # zzz is not a declared param
    assert v.verdict == "uncertain" and not v.result.ok

def test_null_override_state_var_uncertain():
    v = run_plan(splan(null_overrides={"x": "0"}), cfg())     # x is a state var, not a param
    assert v.verdict == "uncertain" and not v.result.ok

def test_null_arm_blowup_uncertain():
    # null arm a=-1e6 -> dx/dt = 1e6*x -> blows up -> non-finite in the null arm -> uncertain
    v = run_plan(splan(null_overrides={"a": "-1e6"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
    assert "reference undeclared" not in (v.result.error or "")   # reached integration, not rejected at validation

def test_total_work_counts_both_arms():
    # grid 120 * n_steps 10_000 = 1.2M (< 2M for 1 arm) but 2.4M at x2 -> total-work cap fires in discrimination
    v = run_plan(splan(null_overrides={"a": "0"}, param_sweep={"a": ["0.8", "1.2", "120"]},
                       t_span=["0", "10"], dt="0.001", max_steps=200_000, max_grid_points=400), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_single_arm_within_double_budget_passes():
    # converse of the total-work x2 test: a single-arm plan where gsize*n_steps (150k) < max_total (200k)
    # but 150k*2 (300k) > max_total -- proving n_arms==1 for single-arm
    # (a hardcoded n_arms=2 would wrongly cap-reject this as uncertain)
    cfg2 = Config(default_model="fake", sim=SimCfg(max_total_steps=200_000))
    v = run_plan(splan(param_sweep={"a": ["0.8", "1.2", "10"]}, t_span=["0", "15"], dt="0.001",
                       max_steps=200_000, max_grid_points=400), cfg2)
    assert v.verdict == "pass"
