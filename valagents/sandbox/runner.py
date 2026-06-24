"""Sandbox runner (subprocess entry point). Reads a frozen ComputationPlan JSON on stdin,
runs the SymPy computation the plan describes, prints a result JSON on stdout.
Imports ONLY json + sympy (+ numpy for magnitude). No network, no filesystem writes.
NEVER execs LLM-provided code: expressions are parsed with parse_expr over a restricted
namespace, not sympify/eval."""
import json
import math
import sys

_ALLOWED = ("sin", "cos", "tan", "tanh", "exp", "log", "sqrt", "Abs", "sign",
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

def _eval_expr(node, env, np, npfuncs, allow_nonfinite=False):
    """Eval a restricted-parsed SymPy Expr over env (symbol->float). Whitelisted node TYPES only;
    anything else (unbound symbol / non-whitelisted node / complex) raises ValueError.
    With allow_nonfinite=False (default) a non-finite real also raises and every return is a finite real float.
    With allow_nonfinite=True, a non-finite REAL (inf/nan) is returned as a float (complex still raises) — used
    ONLY by the overflow-capturing integrator so a divergence/domain-error can be classified, not swallowed.
    No eval/lambdify."""
    if node.is_Symbol:
        name = node.name
        if name not in env:
            raise ValueError(f"unbound symbol: {name}")
        val = float(env[name])
    elif node.is_Number or node.is_NumberSymbol:
        val = float(node)
    elif node.is_Add:
        val = 0.0
        for a in node.args:
            val += _eval_expr(a, env, np, npfuncs, allow_nonfinite)
    elif node.is_Mul:
        val = 1.0
        for a in node.args:
            val *= _eval_expr(a, env, np, npfuncs, allow_nonfinite)
    elif node.is_Pow:
        base = _eval_expr(node.args[0], env, np, npfuncs, allow_nonfinite)
        expo = _eval_expr(node.args[1], env, np, npfuncs, allow_nonfinite)
        try:
            val = base ** expo                          # narrow Pow: no complex continuation
        except OverflowError as e:
            if not allow_nonfinite:
                raise ValueError(f"invalid power {base}**{expo}: {e}")
            val = math.inf                              # magnitude overflow -> +inf (sign irrelevant downstream)
        except (ZeroDivisionError, ValueError) as e:
            if not allow_nonfinite:
                raise ValueError(f"invalid power {base}**{expo}: {e}")
            val = math.nan                              # e.g. 0**negative -> domain error -> nan
    else:
        fn = npfuncs.get(node.func)
        if fn is None or len(node.args) != 1:
            raise ValueError(f"unsupported expression node: {type(node).__name__}")
        val = fn(_eval_expr(node.args[0], env, np, npfuncs, allow_nonfinite))
    if isinstance(val, complex):                        # complex ALWAYS rejected (both modes)
        raise ValueError(f"complex value at {type(node).__name__}")
    if not allow_nonfinite and not math.isfinite(val):
        raise ValueError(f"non-finite value at {type(node).__name__}")
    return float(val)

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

class _DomainError(ValueError):
    """Raised when the RHS becomes undefined at a finite, in-bounds state (a NaN: e.g. log/sqrt of a negative).
    A ValueError subclass, so every existing except-path still maps it to uncertain; the distinct type +
    'domain_error' message let an inconclusive sweep label a well-posedness defect, not a missed divergence."""

_DIVERGENCE_MAG = 1e100   # |state| past this is treated as a numerical divergence. Verdict-invariant (BP-1):
                          # it only sets the t_of scale; t_of's converge/recede behaviour is the same for any
                          # large threshold, so the bounded verdict does not depend on the exact value.

def _rk4_integrate_capturing(rhs_exprs, var_index, env_base, y0, n_steps, dt, np, npfuncs):
    """Like _rk4_integrate but RETURNS (traj, overflow_step) instead of raising on a divergent state.
    overflow_step = first step (1..n_steps) at which the state diverged (an infinity, or |state| > _DIVERGENCE_MAG),
    or None if the whole trajectory stays finite and within _DIVERGENCE_MAG. On divergence, traj is the finite
    prefix traj[0:overflow_step]. A non-finite INITIAL condition raises ValueError; a NaN mid-run (a domain error
    such as log of a negative) raises _DomainError -> uncertain (a domain error is about DOMAIN, not magnitude,
    so it can neither confirm nor refute a boundedness claim; BP-1). Deterministic, fixed-step (L3-D11 holds).
    Used ONLY by the bounded honesty check; derivatives are evaluated with allow_nonfinite=True so a blow-up
    yields inf (classified here as divergence) rather than a swallowed ValueError.
    Known edge (errs safe): a blow-up that first manifests as inf-inf -> NaN INSIDE one RK4 stage, before that
    step's magnitude check fires, is raised as a domain error (uncertain) rather than a divergence. It needs a
    contrived RHS (two individually-overflowing terms subtracting) and errs toward uncertain, so it is acceptable."""
    nvars = len(rhs_exprs)
    y = np.array(y0, dtype=float)
    if not np.all(np.isfinite(y)) or np.iscomplexobj(y):
        raise ValueError("non-finite initial condition")
    def deriv(yv):
        env = dict(env_base)
        for var, i in var_index.items():
            env[var] = float(yv[i])
        d = np.empty(nvars, dtype=float)
        for k, (_, expr) in enumerate(rhs_exprs):
            d[k] = _eval_expr(expr, env, np, npfuncs, allow_nonfinite=True)   # may be inf/nan; complex still raises
        return d
    traj = np.empty((n_steps + 1, nvars), dtype=float)
    traj[0] = y
    for step in range(n_steps):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if np.iscomplexobj(y) or np.any(np.isnan(y)):
            raise _DomainError(f"domain_error: RHS undefined at a reachable finite state (step {step + 1})")
        if np.any(np.isinf(y)) or np.max(np.abs(y)) > _DIVERGENCE_MAG:
            return traj[:step + 1], step + 1                                   # divergence captured
        traj[step + 1] = y
    return traj, None

def _extract_observable(traj, var_index, observable, np):
    import math
    name = observable.get("name"); var = observable.get("var")
    wf = float(observable.get("window_frac", "1.0"))
    if not (0.0 < wf <= 1.0):
        raise ValueError(f"invalid window_frac: {wf}")
    if var not in var_index:
        raise ValueError(f"observable var not a state var: {var}")
    series = traj[:, var_index[var]]
    n = len(series)
    w = max(1, int(np.ceil(wf * n)))
    window = series[-w:]
    if name in ("amplitude", "settle_std") and len(window) < 2:
        raise ValueError(f"{name} needs >= 2 samples (window={len(window)})")
    if name == "final_value":
        val = float(series[-1])
    elif name == "mean_window":
        val = float(np.mean(window))
    elif name == "amplitude":
        val = float((np.max(window) - np.min(window)) / 2.0)
    elif name == "settle_std":
        val = float(np.std(window))
    elif name == "max_value":
        val = float(np.max(window))
    elif name == "min_value":
        val = float(np.min(window))
    elif name == "max_abs":
        val = float(np.max(np.abs(window)))
    else:
        raise ValueError(f"unknown observable: {name}")
    if not math.isfinite(val):
        raise ValueError(f"non-finite observable: {name}({var})")
    return val

def _eval_criterion(val, crit):
    op = crit.get("op"); thr = crit.get("threshold")
    if not isinstance(thr, list) or not thr:
        raise ValueError(f"criterion threshold must be a non-empty list: {thr}")
    if op == "in":
        if len(thr) < 2:
            raise ValueError("op 'in' needs [lo, hi]")
        lo, hi = float(thr[0]), float(thr[1])
        return lo <= val <= hi
    t = float(thr[0])
    if op == "ge":
        return val >= t
    if op == "le":
        return val <= t
    if op == "gt":
        return val > t
    if op == "lt":
        return val < t
    raise ValueError(f"unknown criterion op: {op}")

def _build_grid(param_sweep, init_sweep, parse_num, max_grid_points=None):
    """Cartesian product of param_sweep x init_sweep axes. Each [lo, hi, n] -> n evenly spaced values in
    [lo, hi] inclusive. Returns [(param_overrides, init_overrides), ...]. If max_grid_points is given,
    rejects (raises) when the PROJECTED product of the axis counts exceeds it, BEFORE building any axis."""
    import itertools
    specs = []   # (kind, name, lo_str, hi_str, npts)
    for kind, sweep in (("param", param_sweep), ("init", init_sweep)):
        for name, spec in sweep.items():
            npts = int(float(spec[2]))
            if npts < 1:
                raise ValueError(f"sweep '{name}' needs n >= 1")
            specs.append((kind, name, spec[0], spec[1], npts))
    if max_grid_points is not None:                       # cap on the projected product BEFORE materializing any axis
        projected = 1
        for s in specs:
            projected *= s[4]
        if projected > int(max_grid_points):
            raise ValueError(f"projected grid {projected} exceeds max_grid_points {max_grid_points}")
    axes = []
    for (kind, name, lo_s, hi_s, npts) in specs:
        lo, hi = parse_num(lo_s), parse_num(hi_s)
        step = 0.0 if npts == 1 else (hi - lo) / (npts - 1)
        axes.append((kind, name, [lo + step * i for i in range(npts)]))
    grid = []
    for combo in itertools.product(*[ax[2] for ax in axes]) if axes else [()]:
        pov, iov = {}, {}
        for (kind, name, _), value in zip(axes, combo):
            (pov if kind == "param" else iov)[name] = value
        grid.append((pov, iov))
    return grid

def _spectral_pass(jac_exprs, rhs_exprs, state_vars, fp_exprs, params_env, fp_tol, sim_criterion, np, npfuncs):
    """One linear_stability grid point: evaluate the parametric fixed point under params_env, verify rhs~=0,
    evaluate the Jacobian there, and read the spectral abscissa alpha=max(Re eigvals). Raises (-> uncertain
    upstream) on an off-equilibrium point or a non-finite alpha. Returns (x_star, max_residual, alpha, stable)."""
    import math
    x_star = {var: _eval_expr(fp_exprs[var], params_env, np, npfuncs) for var in state_vars}
    env = {**params_env, **x_star}
    max_res = max(abs(_eval_expr(expr, env, np, npfuncs)) for (_, expr) in rhs_exprs)
    if max_res > fp_tol:
        raise ValueError(f"declared point is not an equilibrium: residual {max_res:.3g} > tol {fp_tol:.3g}")
    n = len(state_vars)
    jac = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            jac[i, j] = _eval_expr(jac_exprs[i][j], env, np, npfuncs)
    alpha = float(np.max(np.linalg.eigvals(jac).real))
    if not math.isfinite(alpha):
        raise ValueError("non-finite spectral abscissa")
    return x_star, max_res, alpha, _eval_criterion(alpha, sim_criterion)

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

_SIM_REQUIRED = {
    "ode_integrate":    ["state_vars", "rhs", "init", "t_span", "dt", "observable", "sim_criterion", "robust_frac"],
    "linear_stability": ["state_vars", "rhs", "fixed_point", "sim_criterion", "robust_frac", "param_sweep"],
}
_SIM_CAPS = {
    "ode_integrate":    ["max_steps", "max_grid_points", "max_state_vars", "max_expr_nodes"],
    "linear_stability": ["max_grid_points", "max_state_vars", "max_expr_nodes"],
}

def _u(msg):
    return {"ok": False, "matched": "neither", "error": msg}

def _run_simulation(plan: dict) -> dict:
    import math
    import sympy
    import numpy as np
    from sympy.parsing.sympy_parser import parse_expr
    primitive = plan.get("primitive")
    required = _SIM_REQUIRED.get(primitive)
    caps = _SIM_CAPS.get(primitive)
    if required is None:
        return _u(f"unsupported primitive: {primitive}")
    for f in required:
        if not plan.get(f):
            return _u(f"missing required field: {f}")
    ceil = plan.get("_sim_ceilings", {})
    _REQUIRED_CEILINGS = ("max_state_vars", "max_expr_nodes", "max_grid_points",
                          "max_steps", "max_total_steps", "min_grid_points",
                          "fixed_point_tol", "min_points_per_axis")
    if not all(k in ceil for k in _REQUIRED_CEILINGS):
        return _u("missing/incomplete sandbox ceilings (no run_plan injection)")
    # caps: positive, and not exceeding config ceilings
    for cap in caps:
        v = int(plan.get(cap, 0))
        if v <= 0:
            return _u(f"non-positive cap: {cap}")
        if ceil and v > int(ceil.get(cap, 0)):
            return _u(f"cap {cap}={v} exceeds ceiling {ceil.get(cap)}")
    state_vars = list(plan["state_vars"])
    if len(state_vars) > int(plan["max_state_vars"]):
        return _u("too many state_vars")
    glob = {n: getattr(sympy, n) for n in _ALLOWED}
    glob["__builtins__"] = {}
    parse_num = lambda s: _parse_number(s, glob)
    npfuncs = _npfuncs(sympy, np)
    try:
        # The restricted parser is aware of state vars AND parameter names (fixed params + swept params),
        # so RHS like "-a*x" parses; any free symbol OUTSIDE that declared set is rejected up front
        # (forbids arbitrary undeclared names — fail-closed, not deferred to an eval-time unbound error).
        allowed = list(state_vars) + list(plan.get("params", {}).keys()) + list(plan.get("param_sweep", {}).keys())
        reserved = set(allowed) & set(_ALLOWED)
        if reserved:
            return _u(f"declared symbol(s) shadow reserved math names: {sorted(reserved)}")
        param_names_set = set(plan.get("params", {})) | set(plan.get("param_sweep", {}))
        collisions = set(state_vars) & param_names_set          # a name can't be BOTH a state var and a parameter
        if collisions:
            return _u(f"name(s) used as both state var and parameter: {sorted(collisions)}")
        if len(state_vars) != len(set(state_vars)):
            return _u(f"duplicate state_vars: {sorted({v for v in state_vars if state_vars.count(v) > 1})}")
        param_overlap = set(plan.get("params", {})) & set(plan.get("param_sweep", {}))
        if param_overlap:
            return _u(f"name(s) both fixed (params) and swept (param_sweep): {sorted(param_overlap)}")
        null_overrides = plan.get("null_overrides", {})
        bad_null = set(null_overrides) - param_names_set
        if bad_null:                                  # NC-D4: a null override may only touch a declared coupling param
            return _u(f"null_overrides reference undeclared/non-param names: {sorted(bad_null)}")
        n_arms = 2 if null_overrides else 1
        local = {n: sympy.Symbol(n) for n in allowed}
        allowed_syms = set(local.values())
        rhs_exprs = []
        for var in state_vars:
            src = str(plan["rhs"].get(var, ""))
            if not src:
                return _u(f"missing rhs for state var: {var}")
            if "__" in src:
                return _u("rejected: '__' in rhs")
            expr = parse_expr(src, local_dict=local, global_dict=glob, evaluate=True)
            if not expr.free_symbols <= allowed_syms:
                return _u(f"rhs '{var}' references undeclared symbol(s): {expr.free_symbols - allowed_syms}")
            if expr.count_ops() + 1 > int(plan["max_expr_nodes"]):
                return _u(f"rhs '{var}' exceeds max_expr_nodes")
            rhs_exprs.append((var, expr))
        var_index = {v: i for i, v in enumerate(state_vars)}
        if primitive == "ode_integrate":
            # grid + caps
            grid = _build_grid(plan.get("param_sweep", {}), plan.get("init_sweep", {}), parse_num,
                               max_grid_points=min(int(plan["max_grid_points"]), int(ceil["max_grid_points"])))
            gsize = len(grid)
            if gsize > int(plan["max_grid_points"]):
                return _u("grid exceeds max_grid_points")
            if ceil and gsize < int(ceil.get("min_grid_points", 0)):
                return _u(f"grid size {gsize} < min_grid_points (not a sweep)")
            t0, t1 = parse_num(plan["t_span"][0]), parse_num(plan["t_span"][1])
            dt = parse_num(plan["dt"])
            if dt <= 0 or t1 <= t0:
                return _u("invalid t_span/dt")
            n_steps = int(math.ceil((t1 - t0) / dt))
            if n_steps > int(plan["max_steps"]):
                return _u("n_steps exceeds max_steps")
            if ceil and gsize * n_steps * n_arms > int(ceil.get("max_total_steps", 0)):
                return _u(f"total work {gsize}*{n_steps}*{n_arms} exceeds max_total_steps")
            # fixed params/init
            base_params = {k: parse_num(v) for k, v in plan.get("params", {}).items()}
            base_init = {k: parse_num(v) for k, v in plan["init"].items()}
            null_parsed = {k: parse_num(v) for k, v in null_overrides.items()} if null_overrides else {}
            rf = parse_num(plan["robust_frac"])
            if not (0.0 < rf <= 1.0):
                return _u(f"robust_frac must be in (0, 1]: {rf}")
            passes = 0
            detail = []                                 # per-grid-point audit table (persisted via stdout.txt)
            for pov, iov in grid:                       # swept overrides fixed
                env_base = {**base_params, **pov}
                init_vals = {**base_init, **iov}
                y0 = np.array([init_vals[v] for v in state_vars], dtype=float)
                traj_m = _rk4_integrate(rhs_exprs, var_index, env_base, y0, n_steps, dt, np, npfuncs)
                obs_m = _extract_observable(traj_m, var_index, plan["observable"], np)
                crit_m = _eval_criterion(obs_m, plan["sim_criterion"])
                if null_overrides:                      # discrimination: behavior present WITH, absent WITHOUT
                    env_null = {**env_base, **null_parsed}
                    traj_n = _rk4_integrate(rhs_exprs, var_index, env_null, y0, n_steps, dt, np, npfuncs)
                    obs_n = _extract_observable(traj_n, var_index, plan["observable"], np)
                    crit_n = _eval_criterion(obs_n, plan["sim_criterion"])
                    point_pass = bool(crit_m and not crit_n)
                    detail.append({"params": pov, "init": iov, "obs_mech": obs_m, "crit_mech": crit_m,
                                   "obs_null": obs_n, "crit_null": crit_n, "discriminate": point_pass})
                else:
                    point_pass = crit_m
                    detail.append({"params": pov, "init": iov, "observable": obs_m, "pass": crit_m})
                if point_pass:
                    passes += 1
            frac = passes / gsize
            robust = frac >= rf
            if null_overrides:
                computed = f"discriminating: {passes}/{gsize} ({frac:.2f} >= {plan['robust_frac']})"
            else:
                computed = f"robust: {passes}/{gsize} pass ({frac:.2f} >= {plan['robust_frac']})"
            return {"ok": True, "computed": computed,
                    "matched": "confirm" if robust else "refute",
                    "detail": detail}
        if primitive == "linear_stability":
            if plan.get("init_sweep"):
                return _u("linear_stability: init_sweep not allowed (param_sweep only)")
            if plan.get("null_overrides"):
                return _u("linear_stability: null_overrides not supported (single-arm only)")
            min_axis = int(ceil.get("min_points_per_axis", 0))
            for name, spec in plan["param_sweep"].items():
                if int(float(spec[2])) < min_axis:
                    return _u(f"param_sweep axis '{name}' has < min_points_per_axis ({min_axis})")
            fixed_point = plan["fixed_point"]
            if set(fixed_point) != set(state_vars):
                return _u("fixed_point keys must equal state_vars")
            param_local = {n: sympy.Symbol(n) for n in
                           (list(plan.get("params", {})) + list(plan.get("param_sweep", {})))}
            param_syms = set(param_local.values())
            fp_exprs = {}
            for var in state_vars:                                  # parse over PARAM symbols only (no state vars)
                src = str(fixed_point[var])
                if "__" in src:
                    return _u("rejected: '__' in fixed_point")
                fe = parse_expr(src, local_dict=param_local, global_dict=glob, evaluate=True)
                if not fe.free_symbols <= param_syms:
                    return _u(f"fixed_point '{var}' references non-param symbol(s): {sorted(map(str, fe.free_symbols - param_syms))}")
                if fe.count_ops() + 1 > int(plan["max_expr_nodes"]):
                    return _u(f"fixed_point '{var}' exceeds max_expr_nodes")
                fp_exprs[var] = fe
            sv_syms = [sympy.Symbol(s) for s in state_vars]
            jac_exprs = []                                          # symbolic Jacobian, once; post-diff node cap
            for (_, expr) in rhs_exprs:
                row = []
                for sj in sv_syms:
                    d = sympy.diff(expr, sj)
                    if d.count_ops() + 1 > int(plan["max_expr_nodes"]):
                        return _u("jacobian entry exceeds max_expr_nodes")
                    row.append(d)
                jac_exprs.append(row)
            grid = _build_grid(plan["param_sweep"], {}, parse_num,
                               max_grid_points=min(int(plan["max_grid_points"]), int(ceil["max_grid_points"])))
            gsize = len(grid)
            if gsize < int(ceil.get("min_grid_points", 0)):
                return _u(f"grid size {gsize} < min_grid_points")
            rf = parse_num(plan["robust_frac"])
            if not (0.0 < rf <= 1.0):
                return _u(f"robust_frac must be in (0, 1]: {rf}")
            fp_tol = float(ceil["fixed_point_tol"])
            base_params = {k: parse_num(val) for k, val in plan.get("params", {}).items()}
            passes, detail, alphas = 0, [], []
            for pov, _iov in grid:
                params_env = {**base_params, **pov}
                x_star, max_res, alpha, stable = _spectral_pass(
                    jac_exprs, rhs_exprs, state_vars, fp_exprs, params_env, fp_tol, plan["sim_criterion"], np, npfuncs)
                alphas.append(alpha)
                if stable:
                    passes += 1
                detail.append({"params": pov, "fixed_point": x_star, "max_residual": max_res,
                               "alpha": alpha, "pass": stable})
            frac = passes / gsize
            robust = frac >= rf
            computed = (f"linear_stability: {passes}/{gsize} points satisfy criterion "
                        f"(frac >= {plan['robust_frac']}); alpha in [{min(alphas):.4g}, {max(alphas):.4g}]")
            return {"ok": True, "computed": computed,
                    "matched": "confirm" if robust else "refute", "detail": detail}
    except Exception as e:                          # parse error, non-finite/complex, bad window, etc.
        return _u(f"{type(e).__name__}: {e}")

def _run(plan: dict) -> dict:
    kind = plan.get("kind")
    if kind == "simulation":
        return _run_simulation(plan)
    if kind == "magnitude":
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
