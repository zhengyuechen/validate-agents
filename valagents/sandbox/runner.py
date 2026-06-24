"""Sandbox runner (subprocess entry point). Reads a frozen ComputationPlan JSON on stdin,
runs the SymPy computation the plan describes, prints a result JSON on stdout.
Imports ONLY json + sympy. No network, no filesystem writes. NEVER execs LLM-provided code:
expressions are parsed with parse_expr over a restricted namespace, not sympify/eval."""
import json
import sys

_ALLOWED = ("sin", "cos", "tan", "exp", "log", "sqrt", "Abs", "sign",
            "pi", "E", "oo", "Rational", "Integer", "Float")

def _run(plan: dict) -> dict:
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

def main() -> None:
    try:
        plan = json.load(sys.stdin)
        out = _run(plan)
    except Exception as e:                       # parse error, sympy failure, etc.
        out = {"ok": False, "matched": "neither", "error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout)

if __name__ == "__main__":
    main()
