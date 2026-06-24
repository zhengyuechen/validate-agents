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
}

def _parse_number(s, glob) -> float:
    from sympy.parsing.sympy_parser import parse_expr
    if "__" in str(s):
        raise ValueError("rejected: '__' not allowed")
    return float(parse_expr(str(s), local_dict={}, global_dict=glob, evaluate=True).evalf())

def _run_magnitude(plan: dict) -> dict:
    import sympy
    import numpy as np
    ck = plan.get("comparison_kind")
    required = _MAG_REQUIRED.get(ck)
    if required is None:
        return {"ok": False, "matched": "neither", "error": f"unsupported comparison_kind: {ck}"}
    for field in required:                        # fail-closed: quantity AND source must be present
        if not str(plan.get(field, "")).strip():
            return {"ok": False, "matched": "neither", "error": f"missing required field: {field}"}
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
