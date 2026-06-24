from valagents.computation import ComputationPlan
from valagents.config import Config, SimCfg
from valagents.sandbox.executor import run_plan

def cfg():
    return Config(default_model="fake")

def splan(**kw):
    base = dict(kind="simulation", primitive="ode_integrate", state_vars=["x"],
                rhs={"x": "-a*x"}, params={}, init={"x": "1.0"},
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

def test_simcfg_has_bounded_knobs():
    s = SimCfg()
    assert s.max_dt_halvings == 3 and s.conv_rtol == 0.1
    assert "max_dt_halvings" in s.model_dump() and "conv_rtol" in s.model_dump()

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

def test_robust_frac_zero_uncertain():
    # robust_frac=0 would be an unconditional rubber-stamp -> rejected as out of (0,1] -> uncertain
    v = run_plan(splan(robust_frac="0"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_name_collision_state_var_and_param_uncertain():
    # a name that is both a state var and a parameter is silently shadowed -> reject fail-closed
    v = run_plan(splan(state_vars=["a"], rhs={"a": "-a*a"}, init={"a": "1.0"},
                       observable={"name": "final_value", "var": "a", "window_frac": "0.1"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_param_in_both_params_and_param_sweep_rejected():
    # a name fixed AND swept is ambiguous (fixed value silently dead) -> fail-closed
    v = run_plan(splan(params={"a": "1.0"}, param_sweep={"a": ["0.8", "1.2", "5"]}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_projected_grid_cap_before_materializing_uncertain():
    # a sweep whose projected product exceeds max_grid_points -> uncertain (without building the huge list)
    v = run_plan(splan(param_sweep={"a": ["0.8", "1.2", "100000"]}, max_grid_points=50), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_duplicate_state_vars_uncertain():
    v = run_plan(splan(state_vars=["x", "x"], rhs={"x": "-a*x"}, init={"x": "1.0"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def lsplan(**kw):
    base = dict(kind="simulation", primitive="linear_stability", state_vars=["x"],
                rhs={"x": "-a*x"}, params={}, fixed_point={"x": "0"},
                param_sweep={"a": ["0.5", "2.0", "6"]},
                sim_criterion={"op": "lt", "threshold": ["0"]}, robust_frac="1",
                max_grid_points=50, max_state_vars=4, max_expr_nodes=50)
    base.update(kw)
    return ComputationPlan(**base)

def test_linstab_stable_confirm():
    # dx/dt = -a*x, a in [0.5,2], equilibrium x*=0, alpha = -a < 0 -> stable everywhere -> confirm
    v = run_plan(lsplan(), cfg())
    assert v.verdict == "pass" and v.result.matched == "confirm"
    assert "linear_stability" in v.measured and "alpha" in v.measured

def test_linstab_instability_via_gt():
    # criterion gt 0 with a>0 (alpha=-a<0) -> NOT > 0 -> refute (the homogeneous state is NOT unstable)
    v = run_plan(lsplan(sim_criterion={"op": "gt", "threshold": ["0"]}), cfg())
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_linstab_instability_onset_confirm():
    # dx/dt = a*x with a>0 -> alpha = +a > 0 -> instability claim (gt 0) confirms
    v = run_plan(lsplan(rhs={"x": "a*x"}, sim_criterion={"op": "gt", "threshold": ["0"]}), cfg())
    assert v.verdict == "pass" and v.result.matched == "confirm"

def test_linstab_parametric_fixed_point():
    # f = a - b*x^2 -> x* = sqrt(a/b); Jacobian d f/dx = -2 b x = -2 sqrt(a b) < 0 -> stable
    v = run_plan(lsplan(state_vars=["x"], rhs={"x": "a - b*x**2"}, params={"b": "1.0"},
                        fixed_point={"x": "sqrt(a/b)"},
                        param_sweep={"a": ["1.0", "4.0", "6"]}), cfg())
    assert v.verdict == "pass" and v.result.matched == "confirm"

def test_linstab_not_an_equilibrium_uncertain():
    # x*=1 is NOT a root of -a*x (residual = -a*1 = -a != 0) -> uncertain
    v = run_plan(lsplan(fixed_point={"x": "1"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_linstab_fixed_point_keys_mismatch_uncertain():
    v = run_plan(lsplan(state_vars=["x", "y"], rhs={"x": "-a*x", "y": "-a*y"}, fixed_point={"x": "0"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok   # missing 'y' coordinate

def test_linstab_fixed_point_references_state_var_uncertain():
    v = run_plan(lsplan(fixed_point={"x": "x"}), cfg())    # references a state var -> circular
    assert v.verdict == "uncertain" and not v.result.ok

def test_linstab_init_sweep_rejected_uncertain():
    v = run_plan(lsplan(init_sweep={"x": ["0", "1", "5"]}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_linstab_per_axis_floor_uncertain():
    # only 3 points on the swept axis, below min_points_per_axis (5) -> uncertain
    v = run_plan(lsplan(param_sweep={"a": ["0.5", "2.0", "3"]}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_linstab_dunder_fixed_point_uncertain():
    v = run_plan(lsplan(fixed_point={"x": "x.__class__"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

# ---------------------------------------------------------------------------
# Task-5 tests: _bounded_observe and helpers
# ---------------------------------------------------------------------------

def _bexprs(rhs_by_var, state_vars, params=()):
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    local = {n: sympy.Symbol(n) for n in list(state_vars) + list(params)}
    return [(v, parse_expr(rhs_by_var[v], local_dict=local, evaluate=True)) for v in state_vars]

def _bounded(rhs_by_var, state_vars, env, y0, n_steps, dt, bound, t_end,
            params=(), max_halvings=3, conv_rtol=0.1, per_refine_max_steps=200_000):
    import sympy, numpy as np
    from valagents.sandbox.runner import _bounded_observe, _npfuncs
    rhs = _bexprs(rhs_by_var, state_vars, params)
    vi = {v: i for i, v in enumerate(state_vars)}
    obs = {"name": "max_abs", "var": state_vars[0], "window_frac": "1.0"}
    return _bounded_observe(rhs, vi, env, y0, n_steps, dt, obs, bound, t_end,
                            max_halvings, conv_rtol, per_refine_max_steps, np, _npfuncs(sympy, np))

def test_bounded_observe_bounded_accepts():
    # decaying x stays well under bound 2 -> bounded at base dt, no refinement
    verdict, info, steps = _bounded({"x": "-x"}, ["x"], {}, [1.0], 500, 0.01, bound=2.0, t_end=5.0)
    assert verdict == "bounded" and info["refinements"] == 0

def test_bounded_observe_confirmed_divergence_refutes():
    # x' = x^2, x0=1 -> singularity at t*=1; t_of converges across refinements -> unbounded
    verdict, info, steps = _bounded({"x": "x**2"}, ["x"], {}, [1.0], 3000, 0.001, bound=10.0, t_end=2.0)
    assert verdict == "unbounded" and info["max_abs"] == "diverged"
    assert 0.9 < info["t_star"] < 1.1

def test_bounded_observe_stiff_artifact_uncertain():
    # x' = -lam*x with base dt*lam above the RK4 stability boundary (~2.78): numerically unstable but TRULY
    # bounded. Either a finer dt becomes bounded, or t_of recedes -> NOT a refutation.
    # lam=200, dt=0.02 -> lam*dt=4.0 > 2.78 (unstable); halving once: lam*(dt/2)=2.0 < 2.78 (stable).
    # Base diverges (kind="div"); first refinement is bounded -> "diverged_unconfirmed" -> uncertain.
    verdict, info, steps = _bounded({"x": "-200.0*x"}, ["x"], {}, [1.0], 200, 0.02, bound=2.0, t_end=4.0)
    assert verdict == "uncertain"           # the load-bearing soundness test: stiff != divergent

def test_bounded_observe_tstar_near_tend_uncertain():
    # same x'=x^2 singularity at t*=1, but t_span ends right at ~1 -> t* within conv_rtol of t_end -> uncertain
    verdict, info, steps = _bounded({"x": "x**2"}, ["x"], {}, [1.0], 1000, 0.001, bound=10.0, t_end=1.0)
    assert verdict == "uncertain"

def test_bounded_observe_budget_exhausted_uncertain():
    # a refuting (diverging) point but per_refine_max_steps too small to take >=2 refinements -> budget exhausted
    verdict, info, steps = _bounded({"x": "x**2"}, ["x"], {}, [1.0], 3000, 0.001, bound=10.0, t_end=2.0,
                                    per_refine_max_steps=4000)   # base 3000 ok, but 2x=6000 > 4000 -> stop
    assert verdict == "uncertain" and info["max_abs"] == "refine_budget_exhausted"

def test_converged_monotone_from_below():
    # direction-agnostic pinning test: convergence FROM BELOW is accepted
    # (the div branch relies on _converged_monotone NOT assuming an approach side)
    from valagents.sandbox.runner import _converged_monotone
    assert _converged_monotone([0.9, 0.99, 1.0], 0.1) is True

def test_bounded_observe_breach_converging_unbounded():
    # A system whose finite max_abs exceeds bound AND converges across refinements -> unbounded.
    # x' = 0, x(0) = 3.0; bound = 2.0. max_abs = 3.0 at every refinement level (constant trajectory).
    # seq = [3.0, 3.0, 3.0, 3.0] (base + 3 halvings). Deltas all zero -> monotone; mags non-increasing.
    # But last delta is 0 and last = 3.0, so mags[-1]/|last| = 0/3.0 = 0.0 < conv_rtol=0.1 -> converged.
    # seq[-1]=3.0 > bound=2.0 -> unbounded.
    verdict, info, steps = _bounded({"x": "0"}, ["x"], {}, [3.0], 100, 0.01, bound=2.0, t_end=1.0,
                                    max_halvings=3, conv_rtol=0.1)
    assert verdict == "unbounded" and info["max_abs"] == 3.0

def test_bounded_observe_breach_artifact_uncertain():
    # A coarse-dt numerical artifact breach that vanishes at finer dt -> uncertain.
    # x' = -5*x, x(0)=0.3, bound=0.5. True trajectory decays from 0.3 -> 0; true max_abs=0.3 < 0.5 (bounded).
    # lam=5, dt=0.6: lam*dt=3.0 > 2.78 (RK4 stability boundary). The stiff-unstable scheme causes the coarse
    # trajectory to grow to ~175 (a finite numerical artifact, below _DIVERGENCE_MAG=1e100 -> classified as
    # "breach" not "div"). At k=1 (dt=0.3, lam*dt=1.5 < 2.78) the scheme is stable and max_abs=0.3 < 0.5
    # (bounded) -> refutation vanished -> uncertain. y0=0.3 < bound=0.5 ensures initial condition is not itself
    # the breach source.
    verdict, info, steps = _bounded({"x": "-5.0*x"}, ["x"], {}, [0.3], 20, 0.6, bound=0.5, t_end=12.0,
                                    max_halvings=3, conv_rtol=0.1)
    assert verdict == "uncertain"
