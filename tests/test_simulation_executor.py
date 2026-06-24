from valagents.computation import ComputationPlan
from valagents.config import Config
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

def test_reserved_name_shadowing_uncertain():
    # a state var named "E" would shadow Euler's number -> reject fail-closed, not silently mis-simulate
    v = run_plan(splan(state_vars=["E"], rhs={"E": "-a*E"}, init={"E": "1.0"},
                       observable={"name": "final_value", "var": "E", "window_frac": "0.1"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
