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
