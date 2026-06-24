import math
import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
import pytest
from valagents.sandbox import runner

def _parse(s, syms):
    glob = {n: getattr(sympy, n) for n in runner._ALLOWED}
    glob["__builtins__"] = {}
    local = {n: sympy.Symbol(n) for n in syms}
    return parse_expr(s, local_dict=local, global_dict=glob, evaluate=True)

def _np():
    return runner._npfuncs(sympy, np)

def test_eval_arithmetic_and_function():
    e = _parse("a*x + sin(y)", ["a", "x", "y"])
    val = runner._eval_expr(e, {"a": 2.0, "x": 3.0, "y": 0.0}, np, _np())
    assert abs(val - 6.0) < 1e-12

def test_eval_unbound_symbol_raises():
    e = _parse("x + z", ["x", "z"])
    with pytest.raises(ValueError):
        runner._eval_expr(e, {"x": 1.0}, np, _np())   # z unbound

def test_eval_complex_pow_raises():
    e = _parse("x**0.5", ["x"])
    with pytest.raises(ValueError):
        runner._eval_expr(e, {"x": -1.0}, np, _np())   # negative base, fractional exp -> complex

def test_eval_non_whitelisted_node_raises():
    e = sympy.Derivative(sympy.Symbol("x"), sympy.Symbol("x"))
    with pytest.raises(ValueError):
        runner._eval_expr(e, {"x": 1.0}, np, _np())

def test_rk4_matches_exponential_decay():
    # dx/dt = -x  ->  x(t) = x0 * e^-t
    rhs = [("x", _parse("-x", ["x"]))]
    vi = {"x": 0}
    traj = runner._rk4_integrate(rhs, vi, {}, np.array([1.0]), n_steps=500, dt=0.01, np=np, npfuncs=_np())
    t_end = 5.0
    assert abs(float(traj[-1, 0]) - math.exp(-t_end)) < 1e-3

def test_rk4_is_deterministic():
    rhs = [("x", _parse("-x", ["x"]))]
    vi = {"x": 0}
    a = runner._rk4_integrate(rhs, vi, {}, np.array([1.0]), 500, 0.01, np, _np())
    b = runner._rk4_integrate(rhs, vi, {}, np.array([1.0]), 500, 0.01, np, _np())
    assert np.array_equal(a, b)

def test_rk4_blowup_raises():
    # dx/dt = x**2, x0=10 -> finite-time blow-up -> non-finite -> raise
    rhs = [("x", _parse("x**2", ["x"]))]
    vi = {"x": 0}
    with pytest.raises(ValueError):
        runner._rk4_integrate(rhs, vi, {}, np.array([10.0]), 100000, 0.01, np, _np())

def test_rk4_trajectory_blowup_raises():
    # finite (huge) constant derivative -> RK4 update overflows y to inf -> step-level trajectory guard fires
    rhs = [("x", _parse("k", ["x", "k"]))]
    vi = {"x": 0}
    with pytest.raises(ValueError):
        runner._rk4_integrate(rhs, vi, {"k": 1e308}, np.array([1e308]), n_steps=50, dt=10.0, np=np, npfuncs=_np())

def test_eval_expr_allow_nonfinite_returns_inf():
    import sympy, numpy as np, math
    from valagents.sandbox.runner import _eval_expr, _npfuncs
    x = sympy.Symbol("x")
    expr = x ** 2
    env = {"x": 1e200}                       # 1e200**2 = 1e400 -> OverflowError in Pow
    # default: raises
    try:
        _eval_expr(expr, env, np, _npfuncs(sympy, np))
        assert False, "expected ValueError"
    except ValueError:
        pass
    # allow_nonfinite: returns inf, does not raise
    val = _eval_expr(expr, env, np, _npfuncs(sympy, np), allow_nonfinite=True)
    assert math.isinf(val)

def test_eval_expr_allow_nonfinite_still_rejects_complex():
    import sympy, numpy as np
    from valagents.sandbox.runner import _eval_expr, _npfuncs
    x = sympy.Symbol("x")
    expr = x ** 0.5                           # sqrt of a negative -> complex
    env = {"x": -4.0}
    for flag in (False, True):                # complex rejected in BOTH modes
        try:
            _eval_expr(expr, env, np, _npfuncs(sympy, np), allow_nonfinite=flag)
            assert False, "complex must always raise"
        except ValueError:
            pass
