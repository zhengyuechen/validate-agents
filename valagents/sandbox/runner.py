"""Sandbox runner (subprocess entry point). Reads a frozen ComputationPlan JSON on stdin,
runs the SymPy computation the plan describes, prints a result JSON on stdout.
Imports ONLY json + sympy (+ numpy for magnitude). No network, no filesystem writes.
NEVER execs LLM-provided code: expressions are parsed with parse_expr over a restricted
namespace, not sympify/eval."""
import json
import sys

_ALLOWED = ("sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "sign",
            "pi", "E", "oo", "Rational", "Integer", "Float")

def _run_symbolic(plan: dict) -> dict:
    # Defense-in-depth: reject dunder strings pre-parse (blocks __import__, __class__, etc.)
    for field in (plan["expression"], plan["expected"], str(plan["limit_point"])):
        if "__" in field:
            raise ValueError("rejected: '__' not allowed in plan expressions")
    for name in plan.get("variables", []):
        if "__" in str(name):
            raise ValueError("rejected: '__' not allowed in variable names")

    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    syms = {name: sympy.Symbol(name) for name in plan.get("variables", [])}
    glob = {n: getattr(sympy, n) for n in _ALLOWED}
    glob["__builtins__"] = {}   # suppress auto-injected builtins (incl. __import__)
    local = dict(syms)
    expr = parse_expr(plan["expression"], local_dict=local, global_dict=glob, evaluate=True)
    expected = parse_expr(plan["expected"], local_dict=local, global_dict=glob, evaluate=True)
    lv = plan["limit_variable"]
    lvar = syms.get(lv, sympy.Symbol(lv))
    pt_raw = str(plan["limit_point"]).strip()
    point = {"oo": sympy.oo, "+oo": sympy.oo, "-oo": -sympy.oo}.get(
        pt_raw, parse_expr(pt_raw, local_dict=local, global_dict=glob, evaluate=True))
    computed = sympy.limit(expr, lvar, point)
    diff = sympy.simplify(computed - expected)
    holds = bool(diff == 0)
    return {"ok": True, "computed": str(computed),
            "matched": "confirm" if holds else "refute"}

_MAG_REQUIRED = {
    "sensitivity_ratio": ["predicted_effect", "baseline_or_null", "sensitivity",
                          "sensitivity_source", "threshold"],
    "bound_check": ["predicted_effect", "bound", "bound_source"],
    "discriminating_margin": ["predicted_effect", "closest_prior_effect",
                              "closest_prior_source", "uncertainty", "threshold"],
}

def _parse_number(s, glob) -> float:
    from sympy.parsing.sympy_parser import parse_expr
    if "__" in str(s):
        raise ValueError("rejected: '__' not allowed")
    return float(parse_expr(str(s), local_dict={}, global_dict=glob, evaluate=True).evalf())

def _npfuncs(sympy, np):
    # Unary whitelisted functions only. NOTE: sqrt(x) arrives as Pow(x, 1/2) after parse_expr(evaluate=True),
    # so it is handled by the _eval_expr Pow branch (with the narrow-Pow complex guard) — not listed here.
    return {sympy.sin: np.sin, sympy.cos: np.cos, sympy.tan: np.tan, sympy.exp: np.exp,
            sympy.log: np.log, sympy.Abs: np.abs, sympy.sign: np.sign, sympy.tanh: np.tanh}

def _eval_expr(node, env, np, npfuncs):
    """Eval a restricted-parsed SymPy Expr over env (symbol->float). Whitelisted node TYPES only;
    anything else (or unbound symbol / non-finite / complex) raises -> uncertain. No eval/lambdify."""
    import math
    if node.is_Symbol:
        name = node.name
        if name not in env:
            raise ValueError(f"unbound symbol: {name}")
        val = float(env[name])
        if not math.isfinite(val):                       # finite-real check at the symbol leaf too
            raise ValueError(f"non-finite symbol value: {name}")
        return val
    if node.is_Number or node.is_NumberSymbol:       # Integer/Float/Rational/pi/E
        return float(node)
    if node.is_Add:
        s = 0.0
        for a in node.args:
            s += _eval_expr(a, env, np, npfuncs)
        return s
    if node.is_Mul:
        p = 1.0
        for a in node.args:
            p *= _eval_expr(a, env, np, npfuncs)
        return p
    if node.is_Pow:
        base = _eval_expr(node.args[0], env, np, npfuncs)
        expo = _eval_expr(node.args[1], env, np, npfuncs)
        val = base ** expo
        if isinstance(val, complex) or not math.isfinite(val):   # narrow Pow: no complex continuation
            raise ValueError(f"non-finite/complex power: {base}**{expo}")
        return float(val)
    fn = npfuncs.get(node.func)
    if fn is not None and len(node.args) == 1:
        val = float(fn(_eval_expr(node.args[0], env, np, npfuncs)))
        if not math.isfinite(val):
            raise ValueError(f"non-finite function value: {node.func}")
        return val
    raise ValueError(f"unsupported expression node: {type(node).__name__}")

def _rk4_integrate(rhs_exprs, var_index, env_base, y0, n_steps, dt, np, npfuncs):
    """Deterministic fixed-step RK4. rhs_exprs: list of (var, Expr); var_index: var->row.
    env_base: fixed params (symbol->float). Returns trajectory (n_steps+1, n_vars). Raises on non-finite."""
    nvars = len(rhs_exprs)
    y = np.array(y0, dtype=float)
    if not np.all(np.isfinite(y)) or np.iscomplexobj(y):     # finite-real INITIAL condition (before any step)
        raise ValueError("non-finite initial condition")
    def deriv(yv):
        env = dict(env_base)
        for var, i in var_index.items():
            env[var] = float(yv[i])
        d = np.empty(nvars, dtype=float)
        for k, (_, expr) in enumerate(rhs_exprs):
            d[k] = _eval_expr(expr, env, np, npfuncs)
        if not np.all(np.isfinite(d)):
            raise ValueError("non-finite derivative")
        return d
    traj = np.empty((n_steps + 1, nvars), dtype=float)
    traj[0] = y
    for step in range(n_steps):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(y)) or np.iscomplexobj(y):
            raise ValueError(f"non-finite trajectory at step {step + 1}")
        traj[step + 1] = y
    return traj

def _run_magnitude(plan: dict) -> dict:
    import sympy
    import numpy as np
    ck = plan.get("comparison_kind")
    required = _MAG_REQUIRED.get(ck)
    if required is None:
        return {"ok": False, "matched": "neither", "error": f"unsupported comparison_kind: {ck}"}
    for field in required:                        # fail-closed: quantity AND source must be present...
        val = str(plan.get(field, "")).strip()
        if not val:
            return {"ok": False, "matched": "neither", "error": f"missing required field: {field}"}
        if "|" in val:                            # ...and a '|' means an empty field spilled in parse_tail (defense-in-depth)
            return {"ok": False, "matched": "neither", "error": f"separator leaked into field: {field}"}
    glob = {n: getattr(sympy, n) for n in _ALLOWED}
    glob["__builtins__"] = {}
    try:
        if ck == "sensitivity_ratio":
            predicted = _parse_number(plan["predicted_effect"], glob)
            baseline = _parse_number(plan["baseline_or_null"], glob)
            sensitivity = _parse_number(plan["sensitivity"], glob)
            threshold = _parse_number(plan["threshold"], glob)
            if sensitivity == 0:
                return {"ok": False, "matched": "neither", "error": "sensitivity is zero"}
            ratio = float(np.abs(predicted - baseline) / sensitivity)
            detectable = ratio >= threshold
            return {"ok": True, "computed": f"ratio={ratio:.6g}",
                    "matched": "confirm" if detectable else "refute"}
        if ck == "bound_check":
            predicted = _parse_number(plan["predicted_effect"], glob)
            bound = _parse_number(plan["bound"], glob)   # bound_source is presence-checked above, never parsed
            compliant = predicted <= bound
            return {"ok": True, "computed": f"predicted={predicted:.6g}, bound={bound:.6g}",
                    "matched": "confirm" if compliant else "refute"}
        if ck == "discriminating_margin":
            predicted = _parse_number(plan["predicted_effect"], glob)
            closest = _parse_number(plan["closest_prior_effect"], glob)
            uncertainty = _parse_number(plan["uncertainty"], glob)
            threshold = _parse_number(plan["threshold"], glob)
            if uncertainty == 0:
                return {"ok": False, "matched": "neither", "error": "uncertainty is zero"}
            margin = float(np.abs(predicted - closest) / uncertainty)
            distinguishable = margin >= threshold
            return {"ok": True, "computed": f"margin={margin:.6g}",
                    "matched": "confirm" if distinguishable else "refute"}
    except Exception as e:
        return {"ok": False, "matched": "neither", "error": f"{type(e).__name__}: {e}"}
    return {"ok": False, "matched": "neither", "error": "no computation performed"}

def _run(plan: dict) -> dict:
    if plan.get("kind") == "magnitude":
        return _run_magnitude(plan)
    return _run_symbolic(plan)

def main() -> None:
    try:
        plan = json.load(sys.stdin)
        out = _run(plan)
    except Exception as e:                       # parse error, sympy failure, etc.
        out = {"ok": False, "matched": "neither", "error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout)

if __name__ == "__main__":
    main()
