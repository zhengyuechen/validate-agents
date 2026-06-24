# Spec 2 Lens 3 — Toy-Model / Simulation Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 toy-model/simulation executor — a structured `SimulationPlan` (`primitive="ode_integrate"`) run by trusted code (safe expression-tree evaluator + deterministic RK4 + robustness sweep), producing a code verdict mapped conservatively onto the gate's attack path.

**Architecture:** Extend `ComputationPlan` with `kind="simulation"` fields (reusing the existing subprocess `run_plan`/`ComputationVerdict` pipeline). A new `_run_simulation` branch in `runner.py` restricted-parses each RHS, evaluates it via an `eval`-free expression-tree walk, integrates with a fixed-step RK4, sweeps a parameter × initial-condition grid, extracts a code-computed observable, and returns PASS iff a structured criterion holds across ≥ `robust_frac` of the grid. A new `Simulation-Designer` (LLM, structured plan only) and `run_simulation_checks` wire it to the gate via `Attack(type="simulation")` — fail → challenged, pass → discounted survived, uncertain → no-op.

**Tech Stack:** Python, Pydantic v2, SymPy restricted `parse_expr` + a custom expression-tree evaluator, numpy (RK4 + observables), subprocess sandbox, pytest (`conda run -n cosci-reproduce python -m pytest`).

## Global Constraints

Copied verbatim from `docs/2026-06-24-validate-agents-spec2-lens3-design.md` (§2, §3, §4, decision log). Every task's requirements implicitly include these:

- **F1 — no arbitrary code.** Each `rhs[var]` and every numeric scalar is parsed with the restricted `parse_expr` (whitelisted namespace, `__builtins__={}`, `"__"`-rejected). Evaluation is **`eval`-free and `lambdify`-free** — a trusted expression-tree walk over whitelisted node *types* only. numpy does float arithmetic only, never executes plan strings. The designer emits a **structured plan only** — never a verdict, never code, never sees the result.
- **F3 — code judges, never the LLM after execution.** `design_simulation(llm)` → `run_plan` (no llm) → `verdict_to_sim_attack` (no llm). The observable is a fixed code vocabulary; the `sim_criterion` is a structured comparison; `confirm_if`/`refute_if` are display-only glosses.
- **Finite-real-float everywhere.** Every evaluator output and trajectory value must be a finite real float. Complex, object-dtype, NaN, Inf → `uncertain`. `Pow` is narrow (no complex continuation: a negative base to a fractional exponent → `uncertain`).
- **Fail-closed.** A missing required field, a non-positive/over-ceiling cap, a cap breach (`max_steps`/`max_grid_points`/`max_state_vars`/`max_expr_nodes`/`max_total_steps`), a grid below `min_grid_points`, an invalid `window_frac`, a parse/`"__"` rejection, an unbound symbol, a non-whitelisted node, or any non-finite/complex value → `uncertain` (never pass/land). Source strings are presence-only; numbers are restricted-parsed.
- **Determinism.** Fixed-step RK4, no adaptive stepping, no RNG → a frozen plan yields a bit-reproducible verdict.
- **Conservative gate mapping (attack path).** A robust **FAIL** → landed `Attack(type="simulation")` → `challenged` (severity `fatal` iff target claim is `load_bearing` AND `role=="novel_core"`, else `major`); **never refuted**. A robust **PASS** → `survived` attack, severity `minor`, **discounted** — no `CheckRecord`, no `independent_sources`, no injected claim, no route to `internally_validated`. **UNCERTAIN** → no attack (F2).
- **Anti-laundering (L2-D9 carried).** `"simulation"` is added to `AttackSurface.attempted` **only on a decisive verdict** (survived/landed); never on `uncertain`. `"simulation"` is an **additional** category and **does not** satisfy the mandatory `"magnitude"` teeth requirement.
- **No-op when no mechanistic claim.** Lens 3 never invents a mechanism target.
- **Gate purity.** `IdeaArtifact._evaluate()` references neither `"simulation"` nor `"primitive"`; it is **not changed**. Simulation flows entirely through `attacks` and `attack_surface.attempted`.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `valagents/computation.py` | `ComputationPlan` simulation fields (`kind="simulation"`); `verdict_to_sim_attack` | 1 |
| `valagents/config.py` | `SimCfg` ceilings + `Config.sim` | 4 |
| `valagents/sandbox/runner.py` | `_eval_expr`, `_rk4_integrate`, observable/criterion/grid helpers, `_run_simulation`, `_run` dispatch | 2,3,4 |
| `valagents/sandbox/executor.py` | inject `cfg.sim` ceilings into the simulation subprocess payload (guarded) | 4 |
| `valagents/agents/simulation_designer.py` | `design_simulation` — structured plan only | 5 |
| `valagents/prompts.py` | `SIMULATION_DESIGNER` prompt | 5 |
| `valagents/scheduler.py` | `run_simulation_checks` (per mechanistic claim, attack path, no-op rule) | 5 |
| `tests/test_simulation_*.py` | model, evaluator/RK4, helpers, executor, integration | 1–5 |

**Test command (all tasks):** `conda run -n cosci-reproduce python -m pytest tests/ -q` (sympy + numpy installed).

**Interfaces that already exist (consume verbatim):**
- `ComputationResult(ok, computed, matched, ...)`, `ComputationVerdict(verdict, measured, plan, result)`, and `run_plan(plan, cfg, artifacts_dir=None) -> ComputationVerdict` mapping `ok=False→uncertain`, `confirm→pass`, `refute→fail`, `measured=result.computed` — `valagents/sandbox/executor.py`, `valagents/computation.py`.
- `runner._run(plan: dict)` dispatches on `plan["kind"]` (`"magnitude"` → `_run_magnitude`, else `_run_symbolic`); `_ALLOWED` whitelist; `_parse_number(s, glob)` (restricted parse → float, rejects `"__"`) — `valagents/sandbox/runner.py`.
- `Attack(type:str, severity:Literal["fatal","major","minor"], status:Literal["survived","landed"], target_claim_id, basis)`; `AtomicClaim(id, statement, type, role, load_bearing, checks, exhausted, origin)` with `role` including `"novel_core"`; the gate rule "landed fatal → severe_objection → needs_experiment → challenged", "landed major (finalized) → open_objection → challenged" — `valagents/artifact.py`.
- Scheduler patterns: `run_magnitude_checks` (per-item design→run→decisive-attack+mark) and `inject_limit_checks` (per-claim cap at 3) — `valagents/scheduler.py`. Test fixtures: `tests/fake_llm.py` `FakeLLM(lambda agent, messages: body)`.

---

### Task 1: Data model (`kind="simulation"`) + `verdict_to_sim_attack`

**Files:**
- Modify: `valagents/computation.py` (`ComputationPlan` Literal + simulation fields; new `verdict_to_sim_attack`)
- Test: `tests/test_simulation_model.py`

**Interfaces:**
- Produces: `ComputationPlan(kind="simulation", primitive="ode_integrate", state_vars, rhs, params, init, param_sweep, init_sweep, t_span, dt, observable, sim_criterion, robust_frac, max_steps, max_grid_points, max_state_vars, max_expr_nodes, target_claim_id)`; `verdict_to_sim_attack(v, target_claim_id, fatal_eligible: bool, tick=0) -> Attack` (confirm→survived/minor; refute→landed/(fatal if fatal_eligible else major); `type="simulation"`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simulation_model.py`:

```python
import inspect
from valagents.computation import (ComputationPlan, ComputationResult,
                                   ComputationVerdict, verdict_to_sim_attack)

def splan(**kw):
    base = dict(kind="simulation", primitive="ode_integrate", state_vars=["x"],
                rhs={"x": "-x"}, init={"x": "1.0"}, t_span=["0", "5"], dt="0.01",
                param_sweep={"a": ["0.3", "0.7", "3"]},
                observable={"name": "final_value", "var": "x", "window_frac": "0.2"},
                sim_criterion={"op": "le", "threshold": ["0.1"]}, robust_frac="0.8",
                max_steps=1000, max_grid_points=50, max_state_vars=4, max_expr_nodes=50)
    base.update(kw)
    return ComputationPlan(**base)

def test_simulation_plan_constructs():
    p = splan()
    assert p.kind == "simulation" and p.primitive == "ode_integrate"
    assert p.rhs == {"x": "-x"} and p.param_sweep == {"a": ["0.3", "0.7", "3"]}
    assert p.expression == ""        # symbolic fields stay defaulted

def _sv(matched):
    p = splan()
    r = ComputationResult(ok=True, computed="robust: 3/3 pass (1.00 >= 0.80)", matched=matched)
    v = ComputationVerdict(verdict=("pass" if matched == "confirm" else "fail"),
                           measured=r.computed, plan=p, result=r)
    return v

def test_confirm_is_survived_minor():
    a = verdict_to_sim_attack(_sv("confirm"), target_claim_id="c1", fatal_eligible=True)
    assert a.type == "simulation" and a.status == "survived" and a.severity == "minor"

def test_refute_eligible_is_landed_fatal():
    a = verdict_to_sim_attack(_sv("refute"), target_claim_id="c1", fatal_eligible=True)
    assert a.status == "landed" and a.severity == "fatal" and a.target_claim_id == "c1"
    assert "simulation" in a.basis and "final_value" in a.basis

def test_refute_not_eligible_is_landed_major():
    a = verdict_to_sim_attack(_sv("refute"), target_claim_id="c1", fatal_eligible=False)
    assert a.status == "landed" and a.severity == "major"

def test_verdict_to_sim_attack_takes_no_llm():
    assert "llm" not in inspect.signature(verdict_to_sim_attack).parameters
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_model.py -q`
Expected: FAIL (`kind="simulation"` not a valid Literal; `verdict_to_sim_attack` undefined).

- [ ] **Step 3: Extend `ComputationPlan`**

In `valagents/computation.py`, change the `kind` line and add the simulation block **after** the magnitude fields (after `discriminating: bool = False`), before the `# --- criterion / glosses (shared) ---` block:

```python
    kind: Literal["symbolic", "magnitude", "simulation"] = "symbolic"
```

```python
    # --- simulation (kind="simulation") ---
    primitive: Literal["ode_integrate", "iterated_map", "monte_carlo", "linear_stability"] | None = None
    state_vars: list[str] = []
    rhs: dict[str, str] = {}
    params: dict[str, str] = {}
    init: dict[str, str] = {}
    param_sweep: dict[str, list[str]] = {}
    init_sweep: dict[str, list[str]] = {}
    t_span: list[str] = []
    dt: str = ""
    observable: dict = {}
    sim_criterion: dict = {}        # structured pass/fail rule (criterion Literal is taken by symbolic/magnitude)
    robust_frac: str = ""
    max_steps: int = 0
    max_grid_points: int = 0
    max_state_vars: int = 0
    max_expr_nodes: int = 0
    # NOTE: max_total_steps is CONFIG-ONLY (SimCfg, Task 4) — a derived ceiling on grid_size x n_steps.
    #       Do NOT add it to ComputationPlan; it is never a plan-declared field.
```

- [ ] **Step 4: Add `verdict_to_sim_attack`**

In `valagents/computation.py`, append after `verdict_to_attack`:

```python
def verdict_to_sim_attack(v: "ComputationVerdict", target_claim_id, fatal_eligible: bool, tick: int = 0):
    """Map an executed simulation ComputationVerdict to an Attack(type='simulation'). No LLM (F3).
    Call ONLY on a decisive verdict. confirm -> survived/minor (DISCOUNTED positive); refute -> landed,
    fatal iff fatal_eligible (target claim load_bearing AND role=='novel_core') else major. Never refutes."""
    from valagents.artifact import Attack
    if v.result.matched == "confirm":
        status, severity = "survived", "minor"
    else:  # "refute" — the mechanism failed its own preregistered toy demonstration
        status, severity = "landed", ("fatal" if fatal_eligible else "major")
    obs = v.plan.observable or {}
    crit = v.plan.sim_criterion or {}
    basis = (f"simulation/{v.plan.primitive}: {v.measured or '?'}; "
             f"observable = {obs.get('name', '?')}({obs.get('var', '?')}); "
             f"criterion = {crit.get('op', '?')} {crit.get('threshold', '?')}; "
             f"robust_frac = {v.plan.robust_frac or 'n/a'}")
    return Attack(type="simulation", severity=severity, status=status,
                  target_claim_id=target_claim_id, basis=basis)
```

- [ ] **Step 5: Run to verify pass + no regression**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_model.py tests/ -q`
Expected: PASS (new file + the full suite, since the model change is additive).

- [ ] **Step 6: Commit**

```bash
git add valagents/computation.py tests/test_simulation_model.py
git commit -m "feat(simulation): ComputationPlan kind=simulation fields + verdict_to_sim_attack (attack path, fatal iff eligible)"
```

---

### Task 2: Safe expression-tree evaluator + deterministic RK4

**Files:**
- Modify: `valagents/sandbox/runner.py` (add `_eval_expr`, `_rk4_integrate`, the numpy-func map)
- Test: `tests/test_simulation_evaluator.py`

**Interfaces:**
- Consumes: `_ALLOWED` (existing), `parse_expr` (existing usage pattern).
- Produces: `_eval_expr(node, env, np, npfuncs) -> float` (whitelisted node types only; raises on anything else / unbound / non-finite / complex); `_npfuncs(sympy, np) -> dict` (sympy function class → numpy ufunc); `_rk4_integrate(rhs_exprs, var_index, env_base, y0, n_steps, dt, np, npfuncs) -> np.ndarray` shape `(n_steps+1, n_vars)` (deterministic; raises on any non-finite step).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simulation_evaluator.py`:

```python
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
    with pytest.raises(Exception):
        runner._eval_expr(e, {"x": 1.0}, np, _np())   # z unbound

def test_eval_complex_pow_raises():
    e = _parse("x**0.5", ["x"])
    with pytest.raises(Exception):
        runner._eval_expr(e, {"x": -1.0}, np, _np())   # negative base, fractional exp -> complex

def test_eval_non_whitelisted_node_raises():
    e = sympy.Derivative(sympy.Symbol("x"), sympy.Symbol("x"))
    with pytest.raises(Exception):
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
    with pytest.raises(Exception):
        runner._rk4_integrate(rhs, vi, {}, np.array([10.0]), 100000, 0.01, np, _np())
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_evaluator.py -q`
Expected: FAIL (`_npfuncs`/`_eval_expr`/`_rk4_integrate` undefined).

- [ ] **Step 3: Implement the evaluator + RK4**

In `valagents/sandbox/runner.py`, add (after `_parse_number`, before `_run`):

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_evaluator.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_evaluator.py
git commit -m "feat(simulation): safe expression-tree evaluator (no eval/lambdify, narrow Pow) + deterministic RK4"
```

---

### Task 3: Observable extractors, structured criterion, sweep grid builder

**Files:**
- Modify: `valagents/sandbox/runner.py` (`_extract_observable`, `_eval_criterion`, `_build_grid`)
- Test: `tests/test_simulation_helpers.py`

**Interfaces:**
- Consumes: nothing from Task 2 directly (pure trajectory/dict helpers); `_parse_number` (existing) is used in Task 4, not here.
- Produces: `_extract_observable(traj, var_index, observable, np) -> float` (fixed vocabulary; raises on bad var / window<2 for amplitude|settle_std / non-finite); `_eval_criterion(val, crit) -> bool` (`ge/le/gt/lt/in`, `in` inclusive); `_build_grid(param_sweep, init_sweep, parse_num) -> list[tuple[dict, dict]]` (Cartesian product of param × init axes; each entry `(param_overrides, init_overrides)`; swept names override fixed at apply time).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simulation_helpers.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_helpers.py -q`
Expected: FAIL (helpers undefined).

- [ ] **Step 3: Implement the helpers**

In `valagents/sandbox/runner.py`, add (after `_rk4_integrate`):

```python
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

def _build_grid(param_sweep, init_sweep, parse_num):
    """Cartesian product of param_sweep x init_sweep axes. Each [lo, hi, n] -> n evenly spaced
    values in [lo, hi] inclusive. Returns [(param_overrides, init_overrides), ...]."""
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
    grid = []
    for combo in itertools.product(*[ax[2] for ax in axes]) if axes else [()]:
        pov, iov = {}, {}
        for (kind, name, _), value in zip(axes, combo):
            (pov if kind == "param" else iov)[name] = value
        grid.append((pov, iov))
    return grid
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_helpers.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_helpers.py
git commit -m "feat(simulation): observable vocabulary + structured criterion + sweep grid builder"
```

---

### Task 4: `_run_simulation` orchestration + two-layer caps + dispatch + `SimCfg` + ceiling injection

**Files:**
- Modify: `valagents/config.py` (`SimCfg` + `Config.sim`)
- Modify: `valagents/sandbox/executor.py` (`run_plan` injects `cfg.sim` ceilings into the simulation payload, guarded)
- Modify: `valagents/sandbox/runner.py` (`_run_simulation`; `_run` dispatch on `kind=="simulation"`)
- Test: `tests/test_simulation_executor.py`

**Interfaces:**
- Consumes: Task 2 (`_eval_expr`, `_rk4_integrate`, `_npfuncs`), Task 3 (`_extract_observable`, `_eval_criterion`, `_build_grid`), `_parse_number`/`_ALLOWED` (existing), `run_plan` (existing).
- Produces: `Config.sim: SimCfg`; `_run_simulation(plan: dict) -> dict` returning `{"ok": True, "computed": "robust: P/G pass (frac >= K)", "matched": "confirm"|"refute"}` on a decisive sweep, else `{"ok": False, "matched": "neither", "error": ...}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simulation_executor.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -q`
Expected: FAIL (`kind="simulation"` reaches `_run_symbolic` → KeyError → uncertain for the pass case; ceilings absent).

- [ ] **Step 3: Add `SimCfg` to config**

In `valagents/config.py`, after `SandboxCfg`:

```python
class SimCfg(BaseModel):
    max_state_vars: int = 8
    max_expr_nodes: int = 200
    max_grid_points: int = 400
    max_steps: int = 200_000
    max_total_steps: int = 2_000_000
    min_grid_points: int = 4
```

And add to `Config` (after `sandbox: SandboxCfg = SandboxCfg()`):

```python
    sim: SimCfg = SimCfg()
```

- [ ] **Step 4: Inject ceilings in `run_plan` (guarded)**

In `valagents/sandbox/executor.py`, in `run_plan`, replace the `input=plan.model_dump_json()` argument. Build the payload first (just before the `subprocess.run` call):

```python
    payload = plan.model_dump_json()
    if plan.kind == "simulation":
        d = json.loads(payload)
        d["_sim_ceilings"] = cfg.sim.model_dump()   # ceilings reach the subprocess; saved artifact stays the frozen plan
        payload = json.dumps(d)
    try:
        proc = subprocess.run(
            [sys.executable, _RUNNER],
            input=payload, capture_output=True, text=True,
            ...
```

(`_save(artifacts_dir, plan, result)` continues to persist `plan.model_dump_json()` — the original frozen plan, without the injected ceilings.) Symbolic/magnitude payloads are byte-unchanged (the `if` is the only new code). **`json` is already imported at the top of `executor.py`** — no new import needed.

- [ ] **Step 5: Implement `_run_simulation` + dispatch**

In `valagents/sandbox/runner.py`, add `_run_simulation` (after the Task-3 helpers) and extend `_run`:

```python
_SIM_REQUIRED = ["state_vars", "rhs", "init", "t_span", "dt", "observable", "sim_criterion", "robust_frac"]
_SIM_CAPS = ["max_steps", "max_grid_points", "max_state_vars", "max_expr_nodes"]

def _u(msg):
    return {"ok": False, "matched": "neither", "error": msg}

def _run_simulation(plan: dict) -> dict:
    import math
    import sympy
    import numpy as np
    from sympy.parsing.sympy_parser import parse_expr
    if plan.get("primitive") != "ode_integrate":
        return _u(f"unsupported primitive: {plan.get('primitive')}")
    for f in _SIM_REQUIRED:
        if not plan.get(f):
            return _u(f"missing required field: {f}")
    ceil = plan.get("_sim_ceilings", {})
    # caps: positive, and not exceeding config ceilings
    for cap in _SIM_CAPS:
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
        # grid + caps
        grid = _build_grid(plan.get("param_sweep", {}), plan.get("init_sweep", {}), parse_num)
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
        if ceil and gsize * n_steps > int(ceil.get("max_total_steps", 0)):
            return _u(f"total work {gsize}*{n_steps} exceeds max_total_steps")
        # fixed params/init
        base_params = {k: parse_num(v) for k, v in plan.get("params", {}).items()}
        base_init = {k: parse_num(v) for k, v in plan["init"].items()}
        passes = 0
        detail = []                                 # per-grid-point audit table (persisted via stdout.txt)
        for pov, iov in grid:                       # swept overrides fixed
            env_base = {**base_params, **pov}
            init_vals = {**base_init, **iov}
            y0 = np.array([init_vals[v] for v in state_vars], dtype=float)
            traj = _rk4_integrate(rhs_exprs, var_index, env_base, y0, n_steps, dt, np, npfuncs)
            obs = _extract_observable(traj, var_index, plan["observable"], np)
            ok_pt = _eval_criterion(obs, plan["sim_criterion"])
            if ok_pt:
                passes += 1
            detail.append({"params": pov, "init": iov, "observable": obs, "pass": ok_pt})
        frac = passes / gsize
        robust = frac >= parse_num(plan["robust_frac"])
        return {"ok": True,
                "computed": f"robust: {passes}/{gsize} pass ({frac:.2f} >= {plan['robust_frac']})",
                "matched": "confirm" if robust else "refute",
                "detail": detail}                   # ignored by ComputationResult; saved in stdout.txt artifact
    except Exception as e:                          # parse error, non-finite/complex, bad window, etc.
        return _u(f"{type(e).__name__}: {e}")
```

Then extend `_run`:

```python
def _run(plan: dict) -> dict:
    kind = plan.get("kind")
    if kind == "simulation":
        return _run_simulation(plan)
    if kind == "magnitude":
        return _run_magnitude(plan)
    return _run_symbolic(plan)
```

- [ ] **Step 6: Run to verify pass + no regression**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py tests/ -q`
Expected: PASS (the new executor tests + the full suite; symbolic/magnitude payloads unchanged).

- [ ] **Step 7: Commit**

```bash
git add valagents/config.py valagents/sandbox/executor.py valagents/sandbox/runner.py tests/test_simulation_executor.py
git commit -m "feat(simulation): _run_simulation orchestration + two-layer caps (positive, ceiling, total-work, min-grid) + SimCfg + ceiling injection"
```

---

### Task 5: Simulation-Designer + `run_simulation_checks` wiring (attack path, no-op rule, anti-laundering)

**Files:**
- Create: `valagents/agents/simulation_designer.py` (`design_simulation`)
- Modify: `valagents/prompts.py` (`SIMULATION_DESIGNER`)
- Modify: `valagents/scheduler.py` (`run_simulation_checks`; call it in `_whole_artifact_lenses`)
- Test: `tests/test_simulation_integration.py`

**Interfaces:**
- Consumes: Task 1 (`ComputationPlan(kind="simulation")`, `verdict_to_sim_attack`), Task 4 (`_run_simulation` via `run_plan`), `FakeLLM`.
- Produces: `design_simulation(claim, art, llm, cfg) -> ComputationPlan | None` (emits a structured plan via a JSON tail; no verdict, no code); `run_simulation_checks(store, llm, cfg, tick=0)` (per `type=="mechanistic"` claim, capped at 3; attack path; mark `"simulation"` decisive-only; no-op when no mechanistic claim).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_simulation_integration.py`:

```python
import inspect
import json
from valagents.scheduler import run_simulation_checks
from valagents.agents.simulation_designer import design_simulation
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim, AttackSurface
from valagents.config import Config
from tests.fake_llm import FakeLLM

def cfg():
    return Config(default_model="fake")

PLAN = {
    "primitive": "ode_integrate", "state_vars": ["x"], "rhs": {"x": "-a*x"},
    "params": {"a": "1.0"}, "init": {"x": "1.0"}, "t_span": ["0", "5"], "dt": "0.01",
    "param_sweep": {"a": ["0.8", "1.2", "5"]},
    "observable": {"name": "final_value", "var": "x", "window_frac": "0.1"},
    "sim_criterion": {"op": "le", "threshold": ["0.2"]}, "robust_frac": "0.8",
    "max_steps": 2000, "max_grid_points": 50, "max_state_vars": 4, "max_expr_nodes": 50,
}
PASS_BODY = "Here is the plan.\n```json\n" + json.dumps(PLAN) + "\n```"
FAIL_PLAN = {**PLAN, "sim_criterion": {"op": "le", "threshold": ["1e-9"]}}   # never met -> robust fail
FAIL_BODY = "```json\n" + json.dumps(FAIL_PLAN) + "\n```"

def _store(role="novel_core", load_bearing=True, mechanistic=True):
    claim = AtomicClaim(id="m1", statement="mechanism M produces oscillation",
                        type=("mechanistic" if mechanistic else "empirical"),
                        role=role, load_bearing=load_bearing)
    art = IdeaArtifact(raw_idea="seed", formal_claim=FormalClaim(statement="x", falsifiable=True),
                       claim_graph=[claim], attack_surface=AttackSurface(attempted=["counterexample"]))
    return ArtifactStore(art)

def router(body):
    return FakeLLM(lambda a, m: body if a == "simulation_designer" else "")

async def test_designer_emits_plan_only():
    s = _store()
    p = await design_simulation(s.current.claim_graph[0], s.current, router(PASS_BODY), cfg())
    assert p is not None and p.kind == "simulation" and p.primitive == "ode_integrate"
    assert p.target_claim_id == "m1"
    assert "ComputationVerdict" not in inspect.getsource(design_simulation)   # F1: no verdict

async def test_designer_malformed_json_returns_none():
    s = _store()
    assert await design_simulation(s.current.claim_graph[0], s.current, router("no json here at all"), cfg()) is None
    bad = router("```json\n{not valid json,,}\n```")
    assert await design_simulation(s.current.claim_graph[0], s.current, bad, cfg()) is None

async def test_robust_fail_lands_fatal_and_challenges():
    s = _store(role="novel_core", load_bearing=True)
    await run_simulation_checks(s, router(FAIL_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "landed" and sims[0].severity == "fatal"
    assert "simulation" in s.current.attack_surface.attempted
    assert s.current.verdict_class == "challenged"

async def test_fail_non_novelcore_is_major():
    s = _store(role="background", load_bearing=True)
    await run_simulation_checks(s, router(FAIL_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].severity == "major"

async def test_robust_pass_is_discounted_survived():
    s = _store()
    await run_simulation_checks(s, router(PASS_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived"
    # discounted: PASS creates NO CheckRecord and sets NO independent_sources on the claim
    assert s.current.claim_graph[0].checks == []
    assert "simulation" in s.current.attack_surface.attempted

async def test_uncertain_no_attack_not_marked(monkeypatch):
    import valagents.sandbox.executor as ex
    from valagents.computation import ComputationVerdict, ComputationResult
    def fake(plan, cfg, artifacts_dir=None):
        return ComputationVerdict(verdict="uncertain", measured="", plan=plan,
                                  result=ComputationResult(ok=False, error="x"))
    monkeypatch.setattr(ex, "run_plan", fake)
    s = _store()
    await run_simulation_checks(s, router(PASS_BODY), cfg())
    assert not [a for a in s.current.attacks if a.type == "simulation"]
    assert "simulation" not in s.current.attack_surface.attempted

async def test_no_mechanistic_claim_is_noop():
    s = _store(mechanistic=False)
    await run_simulation_checks(s, router(PASS_BODY), cfg())
    assert not [a for a in s.current.attacks if a.type == "simulation"]
    assert "simulation" not in s.current.attack_surface.attempted

def test_simulation_does_not_satisfy_magnitude_teeth():
    art = IdeaArtifact(raw_idea="x", attack_surface=AttackSurface(attempted=["simulation", "counterexample"]))
    assert art._thin_attack_surface() is True        # "simulation" != mandatory "magnitude"

def test_evaluate_ignores_simulation_fields():
    assert "simulation" not in inspect.getsource(IdeaArtifact._evaluate)
    assert "primitive" not in inspect.getsource(IdeaArtifact._evaluate)
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py -q`
Expected: FAIL (`design_simulation`/`run_simulation_checks` undefined).

- [ ] **Step 3: Add the `SIMULATION_DESIGNER` prompt**

In `valagents/prompts.py`, add:

```python
SIMULATION_DESIGNER = """You DESIGN a toy-model simulation; you do NOT run or judge it — code does that, \
and you will never see the result. Given a mechanistic claim, produce a structured plan that tests whether \
the proposed dynamics ROBUSTLY produce the claimed behavior. Output NO code — only the structured plan.

FORMAL CLAIM: {formal}
MECHANISTIC CLAIM: {statement}

Model the mechanism as a set of first-order ODEs (primitive "ode_integrate"): give state_vars, the rhs for \
each (plain SymPy-parseable expressions in the state vars + parameters, e.g. "-a*x + b*y"), fixed params and \
initial conditions (init), and — REQUIRED for robustness — a param_sweep and/or init_sweep of ranges \
[lo, hi, n]. Give the observable (one of: final_value, mean_window, amplitude, settle_std, max_value, \
min_value) with its var and window_frac in (0,1], a structured criterion {{"op": ge|le|gt|lt|in, \
"threshold": [..]}} that the observable must satisfy for the claimed behavior, and robust_frac in [0,1] (the \
fraction of the swept grid on which the criterion must hold). Include positive caps: max_steps, \
max_grid_points, max_state_vars, max_expr_nodes.

Output the plan as a SINGLE JSON object in a ```json fenced block, with exactly these keys: primitive, \
state_vars, rhs, params, init, param_sweep, init_sweep, t_span, dt, observable, sim_criterion, robust_frac, \
max_steps, max_grid_points, max_state_vars, max_expr_nodes."""
```

- [ ] **Step 4: Implement `design_simulation`**

Create `valagents/agents/simulation_designer.py`:

```python
"""Simulation-Designer: emits a structured SimulationPlan (kind='simulation') for a mechanistic claim.
It DESIGNS the toy model only — returns no verdict and never sees the execution result (F1/F3)."""
from __future__ import annotations
import json
import re
from valagents.computation import ComputationPlan
from valagents.prompts import SIMULATION_DESIGNER
from valagents.agents.base import build_messages

_FIELDS = ("primitive", "state_vars", "rhs", "params", "init", "param_sweep", "init_sweep",
           "t_span", "dt", "observable", "sim_criterion", "robust_frac",
           "max_steps", "max_grid_points", "max_state_vars", "max_expr_nodes")

def _extract_json(text: str):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"(\{.*\})", text, re.DOTALL)   # fall back to the last bare object
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None

async def design_simulation(claim, art, llm, cfg) -> ComputationPlan | None:
    user = SIMULATION_DESIGNER.format(
        formal=art.formal_claim.statement if art.formal_claim else "",
        statement=claim.statement)
    body = await llm.complete("simulation_designer", build_messages("You design toy-model simulations.", user))
    data = _extract_json(body)
    if not isinstance(data, dict):
        return None
    fields = {k: data[k] for k in _FIELDS if k in data}     # accept only known keys (ignore extras)
    try:
        return ComputationPlan(kind="simulation", target_claim_id=claim.id, **fields)
    except Exception:
        return None
```

- [ ] **Step 5: Implement `run_simulation_checks` + call it**

In `valagents/scheduler.py`, add the import near the other agent imports:

```python
from valagents.agents.simulation_designer import design_simulation
```

Add the function (place it next to `run_magnitude_checks`):

```python
async def run_simulation_checks(store, llm, cfg, tick: int = 0) -> None:
    art = store.current
    claims = [c for c in art.claim_graph if c.type == "mechanistic"][:3]   # no-op when none; cap at 3
    for claim in claims:
        plan = await design_simulation(claim, art, llm, cfg)
        if plan is None:
            continue
        from valagents.sandbox.executor import run_plan
        from valagents.computation import verdict_to_sim_attack
        adir = f"{cfg.results_dir}/computations/simulation/{claim.id}" if getattr(cfg, "results_dir", None) else None
        verdict = run_plan(plan, cfg, artifacts_dir=adir)
        store.record({"event": "simulation_executed", "claim": claim.id,
                      "verdict": verdict.verdict, "computed": verdict.measured})
        if verdict.verdict == "uncertain":
            continue                                   # FAIL-CLOSED: no attack, no mark (L2-D9 / F2)
        fatal_eligible = bool(claim.load_bearing and claim.role == "novel_core")
        attack = verdict_to_sim_attack(verdict, claim.id, fatal_eligible, tick=tick)
        art.attacks = art.attacks + [attack]
        if art.attack_surface is not None and "simulation" not in art.attack_surface.attempted:
            art.attack_surface.attempted = art.attack_surface.attempted + ["simulation"]
        tick += 1
```

In `_whole_artifact_lenses`, add the call right after `run_magnitude_checks(...)`:

```python
    await run_simulation_checks(store, llm, cfg, tick=tick + 700)
```

- [ ] **Step 6: Add the router entry to the repair test fixture (if present)**

If `tests/test_scheduler_repair.py` has a FakeLLM BASE dict that maps agent names to bodies, add `"simulation_designer": ""` to it (an empty body → `design_simulation` returns `None` → no-op). Run `conda run -n cosci-reproduce python -m pytest tests/test_scheduler_repair.py -q` and, only if it fails for a missing `simulation_designer` key, add that entry.

- [ ] **Step 7: Run to verify pass + full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py tests/ -q`
Expected: PASS (new integration tests + the whole suite, including the gate-purity and magnitude-teeth pins).

- [ ] **Step 8: Commit**

```bash
git add valagents/agents/simulation_designer.py valagents/prompts.py valagents/scheduler.py tests/test_simulation_integration.py
git commit -m "feat(simulation): Simulation-Designer + run_simulation_checks (attack path, fatal iff novel_core load-bearing, no-op rule, L2-D9)"
```

---

## Notes for the executor

- **Gate untouched.** No task edits `valagents/artifact.py`. `test_evaluate_ignores_simulation_fields` and `test_simulation_does_not_satisfy_magnitude_teeth` (Task 5) are the gate-purity and teeth-requirement guards — they must stay green.
- **Two-layer expression safety, both required.** The restricted `parse_expr` (whitelist + `__builtins__={}` + `"__"`-reject) builds the tree; `_eval_expr` then executes only whitelisted node *types* and enforces finite-real-float + narrow `Pow`. Do not remove either layer.
- **Ceilings reach the subprocess only via the guarded `run_plan` injection.** The saved artifact (`plan.json`) is the frozen scientific plan without ceilings; symbolic/magnitude payloads are byte-unchanged.
- **L2-D9 line not to cross:** `"simulation"` is added to `attempted` only in the decisive branch of `run_simulation_checks`; an `uncertain` run (or a `None` plan, or no mechanistic claim) marks nothing.
- **`measured = result.computed`** for simulation is the `"robust: P/G pass (frac >= K)"` summary; `verdict_to_sim_attack` surfaces it plus the observable/criterion in the attack basis.
