# Spec 2 Lens 3 — `linear_stability` Primitive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `linear_stability` simulation primitive — linearize a mechanism at its preregistered **parametric** fixed point and read the Jacobian spectral abscissa α(J)=maxᵢ Re λᵢ against a structured criterion, robustly across a parameter sweep — plus two shared front-half fixes the new primitive rides on.

**Architecture:** Two shared fail-closed fixes first (name-uniqueness, projected-grid-cap in `_build_grid`); then the model field + config knobs + the `verdict_to_sim_attack` basis branch; then `_run_simulation` is refactored to dispatch on `primitive` after a shared front half (parse/caps/guards/grid), with the existing RK4 path kept **byte-identical** (guarded by the ode regression suite) and a new `linear_stability` branch (parametric fixed point → verify `rhs≈0` → symbolic Jacobian → eigvals → α → criterion → sweep); then the designer learns `fixed_point`. The gate status mapping is unchanged; only the attack **basis** gains a primitive branch.

**Tech Stack:** Python, Pydantic v2, SymPy (`parse_expr` + `diff`), numpy (`linalg.eigvals`), subprocess sandbox, pytest (`conda run -n cosci-reproduce python -m pytest`).

## Global Constraints

Copied from `docs/2026-06-24-validate-agents-spec2-lens3-linstab-design.md` (LS-D1..D8). Every task implicitly includes these:

- **LS-D1 — parametric fixed point, code-verified.** `fixed_point: dict[str,str]`, values are restricted expressions over the **parameters** (not state vars); evaluated per grid point and verified `|rhs_i| ≤ fixed_point_tol` before the spectrum. A coordinate referencing a state var, or `set(fixed_point) != set(state_vars)`, or residual > tol anywhere → uncertain.
- **LS-D2 — spectral abscissa.** α = `max(Re(numpy.linalg.eigvals(J)))` of the `sympy.diff` Jacobian evaluated at the equilibrium through `_eval_expr`. The runner is **general**: `_eval_criterion(α, sim_criterion)` (`lt`→stability, `gt`→instability-onset). F1/F3 intact.
- **LS-D3/§5 — soundness in the prompt**, not the runner; the loud caveat (hyperbolicity; α≈0 inconclusive).
- **LS-D4 — gate status mapping UNCHANGED; only `verdict_to_sim_attack`'s `basis` gains a `linear_stability` branch** (fixed point + α). `_evaluate`, `run_simulation_checks` untouched. PASS → discounted `survived`; FAIL → `landed`→`challenged`.
- **LS-D5 — fail-closed:** name-uniqueness (state_vars disjoint from `params ∪ param_sweep`); `init_sweep` non-empty → uncertain; projected grid `∏ₖ nₖ > max_grid_points` → uncertain before materializing; per-axis `< min_points_per_axis` → uncertain; `"__"` in `fixed_point`; post-diff `J_ij` node cap; non-finite/non-whitelisted → uncertain; primitive-specific required-fields/caps.
- **LS-D6 — single-arm v1** (no `null_overrides` for `linear_stability`).
- **LS-D8 — knobs:** `SimCfg.fixed_point_tol = 1e-6` (absolute); `SimCfg.min_points_per_axis = 5`; a stability (`lt`) claim should use `robust_frac = 1` (prompt-enforced).
- **`ode_integrate` byte-identical:** the refactor must not change any `ode_integrate` behavior; the full existing simulation suite is the guard.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `valagents/sandbox/runner.py` | projected-grid-cap in `_build_grid`; name-uniqueness; `_run_simulation` primitive dispatch + `linear_stability` branch + `_spectral_pass` | 1, 3 |
| `valagents/computation.py` | `fixed_point` field; `verdict_to_sim_attack` `linear_stability` basis branch | 2 |
| `valagents/config.py` | `SimCfg.fixed_point_tol`, `SimCfg.min_points_per_axis` | 2 |
| `valagents/agents/simulation_designer.py` | `_FIELDS += "fixed_point"` | 4 |
| `valagents/prompts.py` | `SIMULATION_DESIGNER` teaches `linear_stability` | 4 |
| `tests/test_simulation_*.py` | shared-guard, executor, model/basis, integration tests | 1–4 |

**Interfaces that already exist (consume verbatim):**
- `_run_simulation(plan: dict)` (ode-only, `_SIM_REQUIRED`/`_SIM_CAPS` flat lists); helpers `_eval_expr(node, env, np, npfuncs)`, `_rk4_integrate`, `_extract_observable`, `_eval_criterion`, `_build_grid(param_sweep, init_sweep, parse_num)`, `_parse_number`, `_npfuncs`, `_u`, `_ALLOWED` — `valagents/sandbox/runner.py`. `run_plan` injects `cfg.sim.model_dump()` as `plan["_sim_ceilings"]` for `kind=="simulation"`; the runner verifies a `_REQUIRED_CEILINGS` set is present.
- `verdict_to_sim_attack(v, target_claim_id, fatal_eligible, tick=0)`, `run_simulation_checks` — UNCHANGED except the basis branch in Task 2.
- Test fixtures: `tests/test_simulation_executor.py` (`cfg()`, `splan(**kw)`, `run_plan`); `tests/test_simulation_integration.py` (`cfg()`, `_store`, `router`, `PLAN`, `design_simulation`, `run_simulation_checks`).

**Test command (all tasks):** `conda run -n cosci-reproduce python -m pytest tests/ -q`

---

### Task 1: Shared front-half fixes — name-uniqueness + projected-grid-cap

**Files:**
- Modify: `valagents/sandbox/runner.py` (`_build_grid` projected-cap; `_run_simulation` name-uniqueness + the `_build_grid` call)
- Test: `tests/test_simulation_executor.py`, `tests/test_simulation_helpers.py`

**Interfaces:**
- Produces: `_build_grid(param_sweep, init_sweep, parse_num, max_grid_points=None)` (raises `ValueError` if the projected product `∏ len(axis)` exceeds `max_grid_points`, before materializing); `_run_simulation` rejects a name used as both a state var and a parameter → uncertain.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_simulation_executor.py`:

```python
def test_name_collision_state_var_and_param_uncertain():
    # a name that is both a state var and a parameter is silently shadowed -> reject fail-closed
    v = run_plan(splan(state_vars=["a"], rhs={"a": "-a*a"}, init={"a": "1.0"},
                       observable={"name": "final_value", "var": "a", "window_frac": "0.1"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_param_in_both_params_and_param_sweep_is_ok():
    # the LEGIT pattern (fixed default + swept range for the same param) must NOT be flagged
    v = run_plan(splan(), cfg())   # splan has params={"a":"1.0"}, param_sweep={"a":[...]}
    assert v.verdict in ("pass", "fail")   # runs (not uncertain from a false collision)

def test_projected_grid_cap_before_materializing_uncertain():
    # a sweep whose projected product exceeds max_grid_points -> uncertain (without building the huge list)
    v = run_plan(splan(param_sweep={"a": ["0.8", "1.2", "100000"]}, max_grid_points=50), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
```

Add to `tests/test_simulation_helpers.py`:

```python
def test_build_grid_projected_cap_raises():
    import pytest
    pn = lambda s: float(s)
    with pytest.raises(Exception):
        runner._build_grid({"a": ["0", "1", "1000"]}, {"b": ["0", "1", "1000"]}, pn, max_grid_points=100)

def test_build_grid_within_cap_ok():
    pn = lambda s: float(s)
    grid = runner._build_grid({"a": ["0", "1", "3"]}, {}, pn, max_grid_points=100)
    assert len(grid) == 3   # under the cap, builds normally
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py tests/test_simulation_helpers.py -q`
Expected: the collision + projected-cap tests FAIL (no guard yet); `test_param_in_both_params_and_param_sweep_is_ok` already passes (no false collision today).

- [ ] **Step 3: Add the projected-product cap to `_build_grid`**

In `valagents/sandbox/runner.py`, change `_build_grid`'s signature and add the cap before the product is materialized. Replace the function header and add the cap right before the `grid = []` / product line:

```python
def _build_grid(param_sweep, init_sweep, parse_num, max_grid_points=None):
    """Cartesian product of param_sweep x init_sweep axes. Each [lo, hi, n] -> n evenly spaced
    values in [lo, hi] inclusive. Returns [(param_overrides, init_overrides), ...]. If max_grid_points
    is given, rejects (raises) when the PROJECTED product exceeds it, before materializing the list."""
    import itertools
    axes = []   # (kind, name, [values])
    for kind, sweep in (("param", param_sweep), ("init", init_sweep)):
        for name, spec in sweep.items():
            lo, hi, npts = parse_num(spec[0]), parse_num(spec[1]), int(float(spec[2]))
            if npts < 1:
                raise ValueError(f"sweep '{name}' needs n >= 1")
            step = 0.0 if npts == 1 else (hi - lo) / (npts - 1)
            values = [lo + step * i for i in range(npts)]
            axes.append((kind, name, values))
    if max_grid_points is not None:                              # projected-product cap BEFORE materializing
        projected = 1
        for ax in axes:
            projected *= len(ax[2])
        if projected > int(max_grid_points):
            raise ValueError(f"projected grid {projected} exceeds max_grid_points {max_grid_points}")
    grid = []
    for combo in itertools.product(*[ax[2] for ax in axes]) if axes else [()]:
        pov, iov = {}, {}
        for (kind, name, _), value in zip(axes, combo):
            (pov if kind == "param" else iov)[name] = value
        grid.append((pov, iov))
    return grid
```

- [ ] **Step 4: Add the name-uniqueness guard + pass the cap in `_run_simulation`**

In `valagents/sandbox/runner.py` `_run_simulation`, inside the `try`, right after the `allowed = ...` / `reserved` shadow check block, insert the name-uniqueness guard:

```python
        param_names_set = set(plan.get("params", {})) | set(plan.get("param_sweep", {}))
        collisions = set(state_vars) & param_names_set          # a name can't be BOTH a state var and a parameter
        if collisions:
            return _u(f"name(s) used as both state var and parameter: {sorted(collisions)}")
```

And change the existing `_build_grid` call to pass the effective cap:

```python
        grid = _build_grid(plan.get("param_sweep", {}), plan.get("init_sweep", {}), parse_num,
                           max_grid_points=min(int(plan["max_grid_points"]), int(ceil["max_grid_points"])))
```

- [ ] **Step 5: Run to verify pass + no regression**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (the new guards + the full existing simulation suite — the params∩param_sweep pattern still runs; the projected cap fires before materializing).

- [ ] **Step 6: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_executor.py tests/test_simulation_helpers.py
git commit -m "fix(simulation): shared front-half guards — reject state-var/param name collision; cap projected grid size before materializing"
```

(Plain message — NO Co-Authored-By/Claude-Session/attribution trailer.)

---

### Task 2: `fixed_point` model field + `SimCfg` knobs + `verdict_to_sim_attack` basis branch

**Files:**
- Modify: `valagents/computation.py` (`fixed_point` field; basis branch)
- Modify: `valagents/config.py` (`SimCfg.fixed_point_tol`, `SimCfg.min_points_per_axis`)
- Modify: `valagents/sandbox/runner.py` (add the two knobs to `_REQUIRED_CEILINGS`)
- Test: `tests/test_simulation_model.py`

**Interfaces:**
- Produces: `ComputationPlan.fixed_point: dict[str,str] = {}`; `SimCfg.fixed_point_tol: float = 1e-6`, `SimCfg.min_points_per_axis: int = 5` (both injected into `_sim_ceilings`); `verdict_to_sim_attack` renders a `linear_stability` basis (fixed point + α via `v.measured`) instead of `observable = ?(?)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_simulation_model.py`:

```python
from valagents.config import SimCfg

def test_fixed_point_field_and_simcfg_knobs():
    p = ComputationPlan(kind="simulation", primitive="linear_stability", fixed_point={"x": "sqrt(a/b)"})
    assert p.fixed_point == {"x": "sqrt(a/b)"}
    c = SimCfg()
    assert c.fixed_point_tol == 1e-6 and c.min_points_per_axis == 5

def test_sim_attack_basis_linear_stability_branch():
    p = ComputationPlan(kind="simulation", primitive="linear_stability",
                        fixed_point={"x": "0"}, sim_criterion={"op": "lt", "threshold": ["0"]}, robust_frac="1")
    r = ComputationResult(ok=True, computed="linear_stability: 5/5 points satisfy criterion; alpha in [-0.5, -0.2]",
                          matched="confirm")
    v = ComputationVerdict(verdict="pass", measured=r.computed, plan=p, result=r)
    a = verdict_to_sim_attack(v, target_claim_id="m1", fatal_eligible=True)
    assert a.type == "simulation" and a.status == "survived"
    assert "linear_stability" in a.basis and "alpha" in a.basis
    assert "fixed_point" in a.basis and "?(?)" not in a.basis    # NOT the ode observable rendering
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_model.py -q`
Expected: FAIL (`fixed_point` not a field; `SimCfg` lacks the knobs; basis renders `observable = ?(?)`).

- [ ] **Step 3: Add the `fixed_point` model field**

In `valagents/computation.py`, in the `ComputationPlan` simulation block, add immediately after `null_overrides`:

```python
    fixed_point: dict[str, str] = {}   # linear_stability: state var -> equilibrium expression over the params (LS-D1)
```

- [ ] **Step 4: Add the `SimCfg` knobs + the `_REQUIRED_CEILINGS` entries**

In `valagents/config.py`, in `SimCfg`, add after `min_grid_points`:

```python
    fixed_point_tol: float = 1e-6      # linear_stability equilibrium residual tolerance (absolute, LS-D8)
    min_points_per_axis: int = 5       # linear_stability per-swept-axis density floor (LS-D8)
```

In `valagents/sandbox/runner.py`, extend the `_REQUIRED_CEILINGS` tuple inside `_run_simulation` to include the two new keys (they are always injected via `cfg.sim.model_dump()`):

```python
    _REQUIRED_CEILINGS = ("max_state_vars", "max_expr_nodes", "max_grid_points",
                          "max_steps", "max_total_steps", "min_grid_points",
                          "fixed_point_tol", "min_points_per_axis")
```

- [ ] **Step 5: Add the `linear_stability` basis branch to `verdict_to_sim_attack`**

In `valagents/computation.py`, in `verdict_to_sim_attack`, replace the single `basis = (...)` assignment with a primitive-aware branch (status/severity logic above and the `return Attack(...)` below are unchanged):

```python
    crit = v.plan.sim_criterion or {}
    thr_raw = crit.get("threshold", [])
    thr = " ".join(str(x) for x in thr_raw) if thr_raw else "?"
    if v.plan.primitive == "linear_stability":
        basis = (f"simulation/linear_stability: {v.measured or '?'}; "
                 f"fixed_point = {v.plan.fixed_point}; "
                 f"criterion = {crit.get('op', '?')} {thr} (on spectral abscissa alpha); "
                 f"robust_frac = {v.plan.robust_frac or 'n/a'}")
    else:
        obs = v.plan.observable or {}
        basis = (f"simulation/{v.plan.primitive}: {v.measured or '?'}; "
                 f"observable = {obs.get('name', '?')}({obs.get('var', '?')}); "
                 f"criterion = {crit.get('op', '?')} {thr}; "
                 f"robust_frac = {v.plan.robust_frac or 'n/a'}")
```

(Remove the now-superseded `obs = v.plan.observable or {}` / `thr_raw`/`thr` lines that preceded the old single basis, so they're computed once as above.)

- [ ] **Step 6: Run to verify pass + no regression**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_model.py tests/ -q`
Expected: PASS (the new model/basis tests + the full suite; the ode/negcontrol basis still renders `observable = …` via the `else` branch).

- [ ] **Step 7: Commit**

```bash
git add valagents/computation.py valagents/config.py valagents/sandbox/runner.py tests/test_simulation_model.py
git commit -m "feat(simulation): fixed_point field + SimCfg fixed_point_tol/min_points_per_axis + linear_stability basis branch in verdict_to_sim_attack"
```

(Plain message — NO attribution trailer.)

---

### Task 3: `linear_stability` executor — per-primitive dispatch + spectral compute

**Files:**
- Modify: `valagents/sandbox/runner.py` (`_SIM_REQUIRED`/`_SIM_CAPS` → per-primitive dicts; `_run_simulation` dispatch + `linear_stability` branch; `_spectral_pass` helper)
- Test: `tests/test_simulation_executor.py`

**Interfaces:**
- Consumes: Task 1's guards/`_build_grid`; Task 2's `fixed_point` field + `fixed_point_tol`/`min_points_per_axis` ceilings; `_eval_expr`, `_eval_criterion`, `_build_grid`, `_npfuncs`, `_parse_number`.
- Produces: `_run_simulation` dispatches `primitive=="linear_stability"`; verdict `confirm` iff ≥ `robust_frac` of grid points satisfy `criterion(α)`, with `computed = "linear_stability: P/G … alpha in [lo, hi]"`; ode path unchanged.

- [ ] **Step 1: Write the failing executor tests**

Add to `tests/test_simulation_executor.py`:

```python
def lsplan(**kw):
    base = dict(kind="simulation", primitive="linear_stability", state_vars=["x"],
                rhs={"x": "-a*x"}, params={"a": "1.0"}, fixed_point={"x": "0"},
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
    v = run_plan(lsplan(state_vars=["x"], rhs={"x": "a - b*x**2"}, params={"a": "1.0", "b": "1.0"},
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -q`
Expected: FAIL — `primitive=="linear_stability"` hits `_u("unsupported primitive")` (current `_run_simulation` gates on `ode_integrate`).

- [ ] **Step 3: Convert `_SIM_REQUIRED`/`_SIM_CAPS` to per-primitive dicts**

In `valagents/sandbox/runner.py`, replace the two module constants:

```python
_SIM_REQUIRED = {
    "ode_integrate":    ["state_vars", "rhs", "init", "t_span", "dt", "observable", "sim_criterion", "robust_frac"],
    "linear_stability": ["state_vars", "rhs", "fixed_point", "sim_criterion", "robust_frac", "param_sweep"],
}
_SIM_CAPS = {
    "ode_integrate":    ["max_steps", "max_grid_points", "max_state_vars", "max_expr_nodes"],
    "linear_stability": ["max_grid_points", "max_state_vars", "max_expr_nodes"],
}
```

- [ ] **Step 4: Dispatch on `primitive` in `_run_simulation`; branch the back half**

In `valagents/sandbox/runner.py`, change the top of `_run_simulation` to select per-primitive required/caps (replacing the `if plan.get("primitive") != "ode_integrate"` gate and the `for f in _SIM_REQUIRED` / `for cap in _SIM_CAPS` loops):

```python
    primitive = plan.get("primitive")
    required = _SIM_REQUIRED.get(primitive)
    caps = _SIM_CAPS.get(primitive)
    if required is None:
        return _u(f"unsupported primitive: {primitive}")
    for f in required:
        if not plan.get(f):
            return _u(f"missing required field: {f}")
```

and the cap loop to iterate `caps` (the per-primitive list):

```python
    for cap in caps:
        v = int(plan.get(cap, 0))
        if v <= 0:
            return _u(f"non-positive cap: {cap}")
        if ceil and v > int(ceil.get(cap, 0)):
            return _u(f"cap {cap}={v} exceeds ceiling {ceil.get(cap)}")
```

Then, the shared front (`state_vars` count, `glob`/`parse_num`/`npfuncs`, the `try` with name-uniqueness + reserved + RHS parse + `var_index`) is **unchanged**. After `var_index = ...` and before the existing ode body (the `grid = _build_grid(...)` line onward), wrap the existing ode body in `if primitive == "ode_integrate":` and add the `linear_stability` branch. Concretely, the existing lines from `grid = _build_grid(...)` through the `return {"ok": True, ...}` become the body of:

```python
        if primitive == "ode_integrate":
            <the existing ode body, unchanged>
```

and immediately after it add:

```python
        if primitive == "linear_stability":
            if plan.get("init_sweep"):
                return _u("linear_stability: init_sweep not allowed (param_sweep only)")
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
```

- [ ] **Step 5: Add the `_spectral_pass` helper**

In `valagents/sandbox/runner.py`, add (next to the other sim helpers, after `_build_grid`):

```python
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
```

- [ ] **Step 6: Run the executor suite + the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -q`
Expected: PASS (the 10 new `linear_stability` tests + every existing `ode_integrate`/negative-control test — the ode path is byte-identical, just wrapped in `if primitive == "ode_integrate":`).
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (whole suite — the refactor preserves ode behavior).

- [ ] **Step 7: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_executor.py
git commit -m "feat(simulation): linear_stability primitive — parametric fixed point + Jacobian spectral abscissa criterion (per-primitive dispatch; ode path unchanged)"
```

(Plain message — NO attribution trailer.)

---

### Task 4: Designer `fixed_point` support + integration

**Files:**
- Modify: `valagents/agents/simulation_designer.py` (`_FIELDS += "fixed_point"`)
- Modify: `valagents/prompts.py` (`SIMULATION_DESIGNER` teaches `linear_stability`)
- Test: `tests/test_simulation_integration.py`

**Interfaces:**
- Consumes: Task 3's `linear_stability` executor; the unchanged `run_simulation_checks`/gate.
- Produces: `design_simulation` accepts a `fixed_point` key; end-to-end `linear_stability` via the unchanged attack path.

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_simulation_integration.py`:

```python
LS_PLAN = {
    "primitive": "linear_stability", "state_vars": ["x"], "rhs": {"x": "-a*x"},
    "params": {"a": "1.0"}, "fixed_point": {"x": "0"},
    "param_sweep": {"a": ["0.5", "2.0", "6"]},
    "sim_criterion": {"op": "lt", "threshold": ["0"]}, "robust_frac": "1",
    "max_grid_points": 50, "max_state_vars": 4, "max_expr_nodes": 50,
}
LS_BODY = "```json\n" + json.dumps(LS_PLAN) + "\n```"
LS_UNSTABLE = {**LS_PLAN, "rhs": {"x": "a*x"}}   # alpha=+a>0, criterion lt 0 -> NOT stable -> refute -> challenged
LS_UNSTABLE_BODY = "```json\n" + json.dumps(LS_UNSTABLE) + "\n```"

async def test_ls_designer_emits_fixed_point():
    s = _store()
    p = await design_simulation(s.current.claim_graph[0], s.current, router(LS_BODY), cfg())
    assert p is not None and p.primitive == "linear_stability" and p.fixed_point == {"x": "0"}

async def test_ls_stable_is_discounted_survived():
    s = _store()
    await run_simulation_checks(s, router(LS_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived"
    assert s.current.claim_graph[0].checks == []                  # discounted: no CheckRecord
    assert "linear_stability" in sims[0].basis                    # the basis branch (not observable=?(?))

async def test_ls_unstable_challenges():
    s = _store(role="novel_core", load_bearing=True)
    await run_simulation_checks(s, router(LS_UNSTABLE_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "landed" and sims[0].severity == "fatal"
    assert s.current.verdict_class == "challenged"
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py -q`
Expected: FAIL — `design_simulation` drops `fixed_point` (not in `_FIELDS`), so the plan is missing its equilibrium → uncertain → no attack (the stable/unstable assertions fail; `fixed_point == {"x":"0"}` fails).

- [ ] **Step 3: Add `fixed_point` to the designer whitelist**

In `valagents/agents/simulation_designer.py`, add `"fixed_point"` to `_FIELDS` (append after `"null_overrides"`):

```python
_FIELDS = ("primitive", "state_vars", "rhs", "params", "init", "param_sweep", "init_sweep",
           "null_overrides", "t_span", "dt", "observable", "sim_criterion", "robust_frac",
           "max_steps", "max_grid_points", "max_state_vars", "max_expr_nodes", "fixed_point")
```

(`fixed_point` is a `dict[str,str]`; the existing `_NO_COERCE`/`_stringify_scalars` correctly stringifies numeric coordinates like `{"y": 0}` → `{"y": "0"}` and leaves formula strings unchanged.)

- [ ] **Step 4: Teach the `SIMULATION_DESIGNER` prompt about `linear_stability`**

In `valagents/prompts.py`, in `SIMULATION_DESIGNER`, add a paragraph after the `ode_integrate` modeling instructions (before the JSON-output line), and add `fixed_point` to the key list:

```python
Alternatively, for a LINEAR-STABILITY claim (primitive "linear_stability"): give the RHS and the parameters, \
DERIVE the equilibrium and preregister it as fixed_point — one coordinate per state var, each a formula in the \
PARAMETERS only (e.g. fixed_point {{"x": "sqrt(a/b)", "y": "0"}}); give sim_criterion on the spectral abscissa \
alpha (max real eigenvalue of the Jacobian): "lt" a non-positive margin for a STABILITY claim (set robust_frac \
to 1 — any sampled point above the margin refutes), or "gt" a non-negative margin for an INSTABILITY-ONSET \
claim. Prefer a strict margin (a hyperbolic equilibrium). Give a param_sweep with enough points per axis. No \
t_span/dt/init/observable for linear_stability.
```

And add `fixed_point` (optional) to the key list line:

```python
... max_steps, max_grid_points, max_state_vars, max_expr_nodes, and (optional) null_overrides / fixed_point."""
```

- [ ] **Step 5: Run the integration suite + the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py -q`
Expected: PASS (the 3 new `linear_stability` integration tests + all existing ode/negcontrol integration tests + the teeth/gate-purity/malformed-JSON pins).
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (whole suite).

- [ ] **Step 6: Commit**

```bash
git add valagents/agents/simulation_designer.py valagents/prompts.py tests/test_simulation_integration.py
git commit -m "feat(simulation): designer linear_stability support (fixed_point + parametric-equilibrium/margin prompt) + integration"
```

(Plain message — NO attribution trailer.)

---

## Notes for the executor

- **Gate untouched (except the basis branch).** No task edits `valagents/artifact.py` or `run_simulation_checks`; `verdict_to_sim_attack`'s status/severity mapping is unchanged — only its `basis` gains a `linear_stability` branch (Task 2). `test_simulation_does_not_satisfy_magnitude_teeth` and `test_evaluate_ignores_simulation_fields` must stay green.
- **ode byte-identity is the Task-3 guard.** The existing simulation suite (executor + integration) must pass unchanged through the refactor; the ode body moves verbatim into the `if primitive == "ode_integrate":` branch. The **refactor boundary** is the `grid = _build_grid(...)` line: everything before it (the `try`, name-uniqueness, reserved-name, the existing `null_overrides`/`n_arms` block, the RHS parse, `var_index`) stays in the **shared front**; everything from `grid = …` through the `return` becomes the ode branch. The `null_overrides`/`n_arms` block stays shared and is harmless for `linear_stability` (empty `null_overrides` → `n_arms=1`, unused by the spectral branch which never references it).
- **Name-uniqueness is state-var↔param only.** A param appearing in BOTH `params` and `param_sweep` is the legit fixed-default+sweep pattern and must NOT be flagged (Task 1).
- **F1/F3:** `sympy.diff` operates only on restricted-parsed Exprs; evaluation is the whitelist-node `_eval_expr` (a non-whitelisted derivative node → raises → uncertain); `numpy.linalg.eigvals` does float linear algebra on a numeric matrix. The designer emits a structured plan only.
- **Soundness (margin sign, `robust_frac=1` for stability) is prompt-guided, not runner-enforced** — the runner is a general α-vs-criterion checker.
