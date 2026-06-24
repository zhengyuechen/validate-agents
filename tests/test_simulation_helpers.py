import numpy as np
import pytest
from valagents.sandbox import runner

def _traj():
    # x decays 1 -> ~0 over 101 samples; y oscillates between -1 and 1
    t = np.linspace(0, 10, 101)
    x = np.exp(-t)
    y = np.sin(t)
    return np.stack([x, y], axis=1), {"x": 0, "y": 1}

def test_final_value():
    traj, vi = _traj()
    v = runner._extract_observable(traj, vi, {"name": "final_value", "var": "x", "window_frac": "1.0"}, np)
    assert abs(v - float(np.exp(-10))) < 1e-9

def test_amplitude_window():
    traj, vi = _traj()
    v = runner._extract_observable(traj, vi, {"name": "amplitude", "var": "y", "window_frac": "1.0"}, np)
    assert abs(v - 1.0) < 0.05         # ~ (max-min)/2 of sin

def test_settle_std_small_for_decay():
    traj, vi = _traj()
    v = runner._extract_observable(traj, vi, {"name": "settle_std", "var": "x", "window_frac": "0.2"}, np)
    assert v < 1e-2                    # x has settled near 0

def test_observable_window_too_small_raises():
    traj, vi = _traj()
    with pytest.raises(Exception):     # window_frac so small the window has < 2 samples
        runner._extract_observable(traj, vi, {"name": "amplitude", "var": "y", "window_frac": "0.001"}, np)

def test_converged_monotone_true():
    from valagents.sandbox.runner import _converged_monotone
    # t_of converging to ~1.0: deltas -0.1, -0.01 (same sign, shrinking, last rel delta 0.01/0.99<0.1)
    assert _converged_monotone([1.11, 1.01, 1.0], 0.1) is True

def test_converged_monotone_single_lucky_pair_rejected():
    from valagents.sandbox.runner import _converged_monotone
    # overall receding (increasing) with one coincidentally-close pair (5.0,5.1): deltas 4.0, 0.1, 3.9
    # -> NOT shrinking (|0.1| then |3.9| grows) -> reject (a pure 2-value rtol gate would be fooled)
    assert _converged_monotone([1.0, 5.0, 5.1, 9.0], 0.1) is False

def test_converged_monotone_receding_rejected():
    from valagents.sandbox.runner import _converged_monotone
    # steadily receding, deltas grow -> not shrinking -> reject
    assert _converged_monotone([1.0, 2.0, 4.0, 8.0], 0.1) is False

def test_converged_monotone_too_few_samples():
    from valagents.sandbox.runner import _converged_monotone
    assert _converged_monotone([1.0, 1.0], 0.1) is False     # <3 samples (need >=2 refinements)

def test_converged_monotone_last_delta_too_large():
    from valagents.sandbox.runner import _converged_monotone
    # monotone + shrinking but last relative delta 0.5/1.5 = 0.33 > 0.1 -> not yet converged
    assert _converged_monotone([3.0, 2.0, 1.5], 0.1) is False

def test_observable_unknown_var_raises():
    traj, vi = _traj()
    with pytest.raises(Exception):
        runner._extract_observable(traj, vi, {"name": "final_value", "var": "z", "window_frac": "1.0"}, np)

def test_criterion_ops():
    assert runner._eval_criterion(0.5, {"op": "le", "threshold": ["1.0"]}) is True
    assert runner._eval_criterion(2.0, {"op": "le", "threshold": ["1.0"]}) is False
    assert runner._eval_criterion(0.95, {"op": "in", "threshold": ["0.9", "1.1"]}) is True
    assert runner._eval_criterion(0.9, {"op": "in", "threshold": ["0.9", "1.1"]}) is True   # inclusive
    assert runner._eval_criterion(1.2, {"op": "in", "threshold": ["0.9", "1.1"]}) is False

def test_build_grid_product_and_axes():
    pn = lambda s: float(s)
    grid = runner._build_grid({"a": ["0", "1", "3"]}, {"x": ["0", "1", "2"]}, pn)
    assert len(grid) == 6                              # 3 * 2
    pov, iov = grid[0]
    assert set(pov) == {"a"} and set(iov) == {"x"}     # axes split correctly
    avals = sorted({p["a"] for p, _ in grid})
    assert avals == [0.0, 0.5, 1.0]                    # linspace(0,1,3)

def test_build_grid_projected_cap_raises():
    pn = lambda s: float(s)
    with pytest.raises(ValueError):
        runner._build_grid({"a": ["0", "1", "1000"]}, {"b": ["0", "1", "1000"]}, pn, max_grid_points=100)

def test_build_grid_at_cap_ok():
    pn = lambda s: float(s)
    grid = runner._build_grid({"a": ["0", "1", "10"]}, {}, pn, max_grid_points=10)   # projected == cap -> allowed
    assert len(grid) == 10

def test_build_grid_within_cap_ok():
    pn = lambda s: float(s)
    grid = runner._build_grid({"a": ["0", "1", "3"]}, {}, pn, max_grid_points=100)
    assert len(grid) == 3   # under the cap, builds normally

def test_extract_observable_max_abs():
    import numpy as np
    from valagents.sandbox.runner import _extract_observable
    # traj column for var "x": values [-3, 1, 2, -5]; peak |x| over full window = 5
    traj = np.array([[-3.0], [1.0], [2.0], [-5.0]])
    obs = {"name": "max_abs", "var": "x", "window_frac": "1.0"}
    assert _extract_observable(traj, {"x": 0}, obs, np) == 5.0

def test_extract_observable_max_abs_windowed():
    import numpy as np
    from valagents.sandbox.runner import _extract_observable
    # window_frac 0.5 -> last 2 of 4 samples [2, -5] -> peak |x| = 5
    traj = np.array([[-30.0], [1.0], [2.0], [-5.0]])
    obs = {"name": "max_abs", "var": "x", "window_frac": "0.5"}
    assert _extract_observable(traj, {"x": 0}, obs, np) == 5.0

def _exprs(src_by_var, state_vars):
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    local = {n: sympy.Symbol(n) for n in state_vars}
    return [(v, parse_expr(src_by_var[v], local_dict=local, evaluate=True)) for v in state_vars]

def test_capturing_integrator_finite_trajectory():
    import sympy, numpy as np
    from valagents.sandbox.runner import _rk4_integrate_capturing, _npfuncs
    rhs = _exprs({"x": "-x"}, ["x"])                 # decays, never diverges
    traj, overflow = _rk4_integrate_capturing(rhs, {"x": 0}, {}, [1.0], 100, 0.01, np, _npfuncs(sympy, np))
    assert overflow is None and traj.shape == (101, 1)
    assert abs(traj[-1, 0]) < 1.0                    # decayed

def test_capturing_integrator_captures_divergence():
    import sympy, numpy as np
    from valagents.sandbox.runner import _rk4_integrate_capturing, _npfuncs
    rhs = _exprs({"x": "x**2"}, ["x"])               # x0=1 -> singularity at t*=1
    traj, overflow = _rk4_integrate_capturing(rhs, {"x": 0}, {}, [1.0], 2000, 0.001, np, _npfuncs(sympy, np))
    assert overflow is not None                       # captured, did NOT raise
    t_of = overflow * 0.001
    assert 0.9 < t_of < 1.05                          # overflow time near the singularity t*=1
    assert np.all(np.isfinite(traj))                  # the returned prefix is finite

def test_capturing_integrator_nan_domain_error_raises():
    import sympy, numpy as np
    from valagents.sandbox.runner import _rk4_integrate_capturing, _npfuncs, _DomainError
    rhs = _exprs({"x": "log(x)"}, ["x"])             # x crosses below 0 -> log(neg) -> nan -> domain error
    try:
        _rk4_integrate_capturing(rhs, {"x": 0}, {}, [-1.0], 10, 0.1, np, _npfuncs(sympy, np))
        assert False, "expected _DomainError (domain error -> uncertain, NOT a captured divergence)"
    except _DomainError as e:
        assert "domain_error" in str(e)              # distinct diagnostic label, not a generic ValueError


def test_capturing_integrator_nonfinite_initial_raises():
    import sympy, numpy as np
    from valagents.sandbox.runner import _rk4_integrate_capturing, _npfuncs
    rhs = _exprs({"x": "-x"}, ["x"])
    try:
        _rk4_integrate_capturing(rhs, {"x": 0}, {}, [float("inf")], 10, 0.1, np, _npfuncs(sympy, np))
        assert False, "expected ValueError on non-finite initial condition"
    except ValueError:
        pass
