# `bounded` observable (Spec 2 Lens 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `max_abs` observable and an honesty-checked **bounded claim** (`max_abs le bound`) to the Lens-3 simulation executor, so a confirmed dt-independent divergence (or an over-`bound` breach surviving dt-refinement) **refutes** boundedness, while a step-size artifact stays **uncertain**.

**Architecture:** All execution lives in the existing single-file subprocess runner `valagents/sandbox/runner.py` (imports only json/math/sympy/numpy; never execs LLM code). The new work is: a relaxed-finiteness mode on the safe evaluator `_eval_expr`, an overflow-capturing RK4 variant `_rk4_integrate_capturing`, a `max_abs` extractor, a pure convergence predicate `_converged_monotone`, the honesty-check orchestrator `_bounded_observe`, and a branch in the `ode_integrate` sweep loop that uses them. The gate, the attack mapping (`verdict_to_sim_attack`), and `artifact.py` are NOT touched — a bounded REFUTE rides the existing `refute → landed → challenged` path.

**Tech Stack:** Python, Pydantic v2, SymPy + NumPy. Tests run under conda env `cosci-reproduce`: `conda run -n cosci-reproduce python -m pytest tests/ -q`.

## Global Constraints

- **Commit messages carry NO attribution trailer** — no `Co-Authored-By: Claude`, no `Claude-Session:`, no "Generated with Claude", nothing. Plain messages only. Audit every commit.
- **NEVER modify `valagents/artifact.py`** (gate purity). The gate invariants (I1 verdicts gate not narrate; I2 validated = survived an independent code check; I3 gate is total) must be untouched.
- **Do NOT modify `valagents/computation.py`'s `verdict_to_sim_attack`, `_evaluate`, or the scheduler `run_simulation_checks`.** The bounded REFUTE reuses the existing mapping: `confirm → survived/minor` (discounted), `refute → landed` (fatal iff target claim is `load_bearing` AND `role=='novel_core'`, else major) `→ challenged`; `uncertain → no attack`. The existing `verdict_to_sim_attack` else-branch already renders `observable = max_abs(var)` + `criterion = le <bound>` correctly — confirm via test, change nothing.
- **The honesty check is convergence, not persistence.** A refuting point (overflow OR finite `max_abs > bound`) refutes ONLY if its refuting quantity converges **monotone AND shrinking** (§3a) across `dt → dt/2 → …`; anything else (vanish, recede, type-morph, non-converge, budget-exhaust) → **uncertain**. Err toward uncertain.
- **`t*` near `t_span_end` is uncertain** (§3b): a converged `t*` within `conv_rtol` of `t_span_end` → uncertain, never refute.
- **Bit-reproducible verdict** (§3c): refinement sequence is always `dt → dt/2 → dt/4`; a point's verdict depends only on its own `(params, init)` (traversal-order-independent); the cumulative budget is a deterministic integer step count, never wall-clock.
- **Overflow trigger (BP-1):** divergence = state reaching `inf` or crossing `_DIVERGENCE_MAG = 1e100`; a `nan` (domain error) → uncertain, never a divergence. `_eval_expr`'s `allow_nonfinite` relaxes ONLY the finiteness raise; the complex guard and all structure/whitelist guards stay unconditional.
- **Scoping:** the inverted/honesty path fires ONLY for `observable.name == "max_abs"` AND `sim_criterion.op == "le"`. `max_abs` with any other op uses the v1 rule (finite → value; blow-up → uncertain).
- **`ode_integrate` only.** No `linear_stability` trajectory. No `gt` "exceeds" mirror (deferred, B-D6).
- Result detail rows must stay finite/JSON-valid: a confirmed divergence writes the string `"diverged"`, a budget-exhausted refuting point writes `"refine_budget_exhausted"` — never `Infinity`/`+inf`.

---

## File Structure

- `valagents/config.py` — `SimCfg` gains `max_dt_halvings: int = 3`, `conv_rtol: float = 0.1` (Task 1).
- `valagents/sandbox/runner.py` — all executor work (Tasks 2–6): `_eval_expr` flag, `_rk4_integrate_capturing`, `max_abs` in `_extract_observable`, `_converged_monotone`, `_bounded_observe`, the `ode_integrate` loop branch, `_REQUIRED_CEILINGS` additions.
- `valagents/prompts.py` — `SIMULATION_DESIGNER` gains bounded-claim guidance (Task 7).
- `tests/test_simulation_helpers.py` — unit tests for the integrator + `max_abs` + `_converged_monotone` (Tasks 2–4).
- `tests/test_simulation_evaluator.py` — `_eval_expr` `allow_nonfinite` unit tests (Task 2's eval part may live here; keep the existing finite-real tests green).
- `tests/test_simulation_executor.py` — `_bounded_observe` + `run_plan` end-to-end executor tests (Tasks 5–6).
- `tests/test_simulation_integration.py` — designer + gate integration tests (Task 7).

The implementer should `cat` the relevant existing file before editing, and follow its style (terse comments tying code to decision-log IDs, `_u(msg)` for uncertain returns, `parse_num` lambdas, etc.).

---

## Task 1: Config knobs (`max_dt_halvings`, `conv_rtol`)

**Files:**
- Modify: `valagents/config.py:22-30` (the `SimCfg` class)
- Test: `tests/test_simulation_executor.py` (add one test near the existing `test_missing_ceilings_fail_closed`)

**Interfaces:**
- Consumes: nothing.
- Produces: `SimCfg.max_dt_halvings: int = 3` and `SimCfg.conv_rtol: float = 0.1`. These flow to the subprocess automatically via `run_plan`'s existing `d["_sim_ceilings"] = cfg.sim.model_dump()` (executor.py:52), so Task 6 reads them from `ceil["max_dt_halvings"]` / `ceil["conv_rtol"]`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_simulation_executor.py`:

```python
def test_simcfg_has_bounded_knobs():
    s = SimCfg()
    assert s.max_dt_halvings == 3 and s.conv_rtol == 0.1
    assert "max_dt_halvings" in s.model_dump() and "conv_rtol" in s.model_dump()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py::test_simcfg_has_bounded_knobs -v`
Expected: FAIL — `AttributeError: 'SimCfg' object has no attribute 'max_dt_halvings'`.

- [ ] **Step 3: Add the two fields**

In `valagents/config.py`, inside `class SimCfg(BaseModel)`, after the `min_points_per_axis` line (config.py:30), add:

```python
    max_dt_halvings: int = 3           # bounded honesty check: dt-refinement depth (BP / B-D4)
    conv_rtol: float = 0.1             # bounded honesty check: convergence tolerance on t*/max_abs (B-D7)
```

- [ ] **Step 4: Run the new test and the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py::test_simcfg_has_bounded_knobs -v`
Expected: PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass (the new ceilings are now in every injected `_sim_ceilings`; nothing yet requires them, so existing behavior is unchanged).

- [ ] **Step 5: Commit**

```bash
git add valagents/config.py tests/test_simulation_executor.py
git commit -m "feat(sim): add bounded honesty-check config knobs (max_dt_halvings, conv_rtol)"
```

---

## Task 2: `max_abs` observable + `_eval_expr` `allow_nonfinite` mode

Two small, independent low-level additions. `max_abs` serves the scoping fallback (op != le) and is reused by `_bounded_observe`; `allow_nonfinite` is needed by the capturing integrator (Task 3).

**Files:**
- Modify: `valagents/sandbox/runner.py:61-94` (`_eval_expr`), `runner.py:126-156` (`_extract_observable`)
- Test: `tests/test_simulation_evaluator.py` (eval flag), `tests/test_simulation_helpers.py` (max_abs)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `_eval_expr(node, env, np, npfuncs, allow_nonfinite=False)` — when `allow_nonfinite=True`, a non-finite *real* value (`inf`/`nan`) is RETURNED as a float instead of raising; a complex value STILL raises; all unbound-symbol/structure/whitelist guards STILL raise; a Pow `OverflowError` returns `inf`, a Pow `ZeroDivisionError`/`ValueError` returns `nan`. Default `False` → byte-identical to current behavior.
  - `_extract_observable` recognizes `name == "max_abs"` → `float(np.max(np.abs(window)))`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_simulation_evaluator.py` (it already imports `_eval_expr`, `sympy`, `numpy as np`, `_npfuncs`; match its existing fixture style — check the top of the file):

```python
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
```

Add to `tests/test_simulation_helpers.py` (it already imports the runner helpers + numpy; match its style):

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_evaluator.py::test_eval_expr_allow_nonfinite_returns_inf tests/test_simulation_helpers.py::test_extract_observable_max_abs -v`
Expected: FAIL — `_eval_expr() got an unexpected keyword argument 'allow_nonfinite'` and `unknown observable: max_abs`.

- [ ] **Step 3a: Add `allow_nonfinite` to `_eval_expr`**

Replace the `_eval_expr` signature and its Pow branch and final guard (runner.py:61-94). New full function:

```python
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
```

- [ ] **Step 3b: Add `max_abs` to `_extract_observable`**

In `_extract_observable` (runner.py:148-153), after the `min_value` branch and before the `else`, add a `max_abs` branch:

```python
    elif name == "max_value":
        val = float(np.max(window))
    elif name == "min_value":
        val = float(np.min(window))
    elif name == "max_abs":
        val = float(np.max(np.abs(window)))
    else:
        raise ValueError(f"unknown observable: {name}")
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_evaluator.py tests/test_simulation_helpers.py -v`
Expected: the four new tests PASS; all pre-existing tests in both files still PASS (default `allow_nonfinite=False` is byte-identical).
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_evaluator.py tests/test_simulation_helpers.py
git commit -m "feat(sim): max_abs observable + _eval_expr allow_nonfinite mode (complex guard intact)"
```

---

## Task 3: Overflow-capturing integrator `_rk4_integrate_capturing`

**Files:**
- Modify: `valagents/sandbox/runner.py` — add `_DIVERGENCE_MAG` constant + `_rk4_integrate_capturing` right after `_rk4_integrate` (runner.py:124). Leave `_rk4_integrate` UNCHANGED.
- Test: `tests/test_simulation_helpers.py`

**Interfaces:**
- Consumes: `_eval_expr(..., allow_nonfinite=True)` (Task 2).
- Produces: `_rk4_integrate_capturing(rhs_exprs, var_index, env_base, y0, n_steps, dt, np, npfuncs) -> (traj, overflow_step)`. `overflow_step` is the int step (1..n_steps) at which the state first diverged (`inf` or `|state| > _DIVERGENCE_MAG`), or `None` if the trajectory stayed finite and `<= _DIVERGENCE_MAG`. On divergence, `traj` is the finite prefix `traj[0:overflow_step]` (rows 0..overflow_step-1). A non-finite **initial** condition raises plain `ValueError`; a `nan` mid-run (a *domain* error — the RHS became undefined at a finite, in-bounds state, e.g. `log` of a negative) raises **`_DomainError(ValueError)`** with a `"domain_error"` message so the inconclusive sweep can tell "model ill-posed at a reachable state" apart from "couldn't confirm divergence". Module-level: `class _DomainError(ValueError): pass`, constant `_DIVERGENCE_MAG = 1e100`. (`_DomainError` is a `ValueError` subclass, so every existing `except ValueError`/`except Exception` path still maps it to uncertain — it only adds a recognizable label.)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_simulation_helpers.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_helpers.py -k capturing -v`
Expected: FAIL — `cannot import name '_rk4_integrate_capturing'`.

- [ ] **Step 3: Implement the capturing integrator**

In `valagents/sandbox/runner.py`, immediately after `_rk4_integrate` ends (after runner.py:124), add:

```python
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
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_helpers.py -k capturing -v`
Expected: all four PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass (`_rk4_integrate` unchanged; no existing path calls the new function yet).

- [ ] **Step 5: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_helpers.py
git commit -m "feat(sim): overflow-capturing RK4 (returns overflow step; nan domain-error -> raise)"
```

---

## Task 4: Convergence predicate `_converged_monotone` (§3a)

**Files:**
- Modify: `valagents/sandbox/runner.py` — add `_converged_monotone` near the other sim helpers (e.g. right after `_rk4_integrate_capturing`).
- Test: `tests/test_simulation_helpers.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `_converged_monotone(seq, rtol) -> bool`. `seq` is the list of refining quantities `[q0, q1, q2, ...]` (q0 = base dt, then each halving). Returns `True` iff: (i) `len(seq) >= 3` (≥2 deltas); (ii) the deltas `d_i = seq[i+1]-seq[i]` are all the **same sign or zero** (monotone, no reversal); (iii) `|d_i|` is **non-increasing** (shrinking); (iv) the last relative delta `|d_last| / |seq[-1]| < rtol` with `seq[-1] != 0`. This is the §3a "monotone AND shrinking, not a single lucky pair" predicate. It does NOT itself check direction-toward-bound or t*-window — callers add those.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_simulation_helpers.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_helpers.py -k converged_monotone -v`
Expected: FAIL — `cannot import name '_converged_monotone'`.

- [ ] **Step 3: Implement the predicate**

In `valagents/sandbox/runner.py`, add (near `_rk4_integrate_capturing`):

```python
def _converged_monotone(seq, rtol):
    """§3a — convergence is monotone AND shrinking, not one lucky pair. seq = refining quantities
    [q0(base), q1(dt/2), q2(dt/4), ...]. True iff: >=3 samples (>=2 refinements); the consecutive deltas are
    all same-sign-or-zero (monotone, no reversal); |delta| is non-increasing (shrinking); and the last relative
    delta < rtol. A single coincidentally-close pair on an otherwise-receding sequence fails the shrinking test."""
    if len(seq) < 3:
        return False
    deltas = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    signs = [(d > 0) - (d < 0) for d in deltas]               # +1 / 0 / -1
    nonzero = [s for s in signs if s != 0]
    if len(set(nonzero)) > 1:                                 # a direction reversal -> not monotone
        return False
    mags = [abs(d) for d in deltas]
    for a, b in zip(mags, mags[1:]):
        if b > a:                                            # a delta grew -> not shrinking
            return False
    last = seq[-1]
    if last == 0:
        return False
    return mags[-1] / abs(last) < rtol
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_helpers.py -k converged_monotone -v`
Expected: all five PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_helpers.py
git commit -m "feat(sim): _converged_monotone predicate (monotone+shrinking, rejects single lucky pair)"
```

---

## Task 5: The honesty-check orchestrator `_bounded_observe` (§3)

**Files:**
- Modify: `valagents/sandbox/runner.py` — add `_trajectory_converges` (§3d) and `_bounded_observe` after `_converged_monotone`.
- Test: `tests/test_simulation_executor.py` (real-numpy, calling `_bounded_observe` directly through a small helper that parses RHS the way the executor does).

**Interfaces:**
- Consumes: `_rk4_integrate_capturing` (T3), `_extract_observable` with `max_abs` (T2), `_converged_monotone` (T4).
- Produces:
  `_bounded_observe(rhs_exprs, var_index, env_base, y0, n_steps, dt, observable, bound, t_end, max_halvings, conv_rtol, per_refine_max_steps, np, npfuncs) -> (verdict, info, steps_used)`
  - `verdict` ∈ `{"bounded", "unbounded", "uncertain"}`.
  - `info` is a dict for the detail row: `{"max_abs": <float | "diverged" | "refine_budget_exhausted" | "trajectory_unconverged" | "diverged_unconfirmed" | "morph_unconfirmed">, "t_star": <float | None>, "refinements": <int>}`. The sentinels: `"diverged"` = confirmed divergence (refute); a float = bounded value (pass) or confirmed breach (refute); `"refine_budget_exhausted"` = ran out without deciding; `"trajectory_unconverged"` = §3d failed (a deep-unstable stiff artifact whose `t_of` converged but path is `dt`-divergent); `"diverged_unconfirmed"`/`"morph_unconfirmed"` = a divergence that vanished/morphed under refinement. All non-`"diverged"`/non-pass cases are `verdict="uncertain"`.
  - `steps_used` is the total integration steps consumed (base + refinements) for the cumulative budget.
  - It does NOT catch its own exceptions: a raise (e.g. a `nan` domain error from the integrator, or a bad window) propagates to the caller's `try`, which maps it to uncertain (`_u`). The "uncertain" verdict value is returned only for the *decided-uncertain* cases (vanish / recede / type-morph / non-converge / budget-exhaust).

  Decision logic:
  1. **Base** at `(n_steps, dt)`: `traj, overflow = _rk4_integrate_capturing(...)`.
     - `overflow is not None` → refuting-DIVERGENCE; `q0 = overflow * dt`; `kind = "div"`.
     - else `m = _extract_observable(traj, ..., max_abs, ...)`; if `m <= bound` → return `("bounded", {"max_abs": m, "t_star": None, "refinements": 0}, n_steps)`; else refuting-BREACH; `q0 = m`; `kind = "breach"`.
  2. **Refine** `k = 1 .. max_halvings`, `n_k = n_steps * 2**k`, `dt_k = dt / 2**k`:
     - if `n_k > per_refine_max_steps` → return `("uncertain", {"max_abs": "refine_budget_exhausted", "t_star": None, "refinements": k-1}, steps_used)`.
     - integrate capturing at `(n_k, dt_k)`; classify same as base.
     - if BOUNDED → return `("uncertain", {"max_abs": <base q0 if breach else "diverged"-not-used>, ...})` — the refutation vanished. Record the base finite value when available: for a breach use `q0`, for a div there is no finite value so record `"refine_budget_exhausted"` is wrong; use a plain note. Concretely: `return ("uncertain", {"max_abs": (q0 if kind=="breach" else "diverged_unconfirmed"), "t_star": None, "refinements": k}, steps_used)`. (Sentinel string `"diverged_unconfirmed"` distinguishes a vanished divergence from a confirmed `"diverged"`.)
     - if the refinement's kind != base kind (div↔breach morph) → return `("uncertain", {... "max_abs": "morph_unconfirmed" ...}, steps_used)`.
     - else append `q_k`.
  3. **Decide** from `seq = [q0, q1, ...]` (all same kind, none vanished):
     - `kind == "div"`: refute (`"unbounded"`, info `"diverged"`) IFF **all three**: (i) `_converged_monotone(seq, conv_rtol)`; (ii) `seq[-1] < t_end * (1.0 - conv_rtol)` (§3b window guard); **(iii) `_trajectory_converges(div_levels, conv_rtol, np)` (§3d — the pre-overflow trajectory is `dt`-converged).** If (i)/(ii) fail → `"uncertain"` + `"refine_budget_exhausted"`; if (iii) fails → `"uncertain"` + the distinct sentinel `"trajectory_unconverged"` (a stiff numerical artifact: `t_of` converged but the path is `dt`-divergent garbage). **(i)+(ii) are necessary but NOT sufficient — a deep-unstable stiff instability (`ẋ=-1000x` at `dt=0.05`) has a converging `t_of`; (iii) is the load-bearing addition that separates a real singularity from numerical garbage. Still NO `seq[-1] <= seq[0]` clause — finding A stands; this case has *decreasing* `t_of` that would pass that clause anyway.**
     - `kind == "breach"`: if `_converged_monotone(seq, conv_rtol)` AND `seq[-1] > bound` → return `("unbounded", {"max_abs": seq[-1], "t_star": None, "refinements": len(seq)-1}, steps_used)`; else `("uncertain", {"max_abs": "refine_budget_exhausted", "t_star": None, "refinements": len(seq)-1}, steps_used)`. (The breach branch needs NO §3d check — its converged quantity is the physical `max_abs` itself, no proxy.)

  To support §3d, `classify` must ALSO return the partial trajectory + overflow step + step size for a div point, and the refine loop accumulates `div_levels = [(traj_k, overflow_step_k, dt_k), ...]` (one per refinement level, all overflowed). The trajectories are already in hand (the integrator returns them); §3d is pure post-processing — NO extra integration, so `steps_used` is unaffected. `steps_used` accumulates `n_steps + sum(n_k actually run)`.

  **`_trajectory_converges(div_levels, conv_rtol, np, n_samples=6) -> bool` (§3d):** `t_edge = min over levels of (overflow_step_k * dt_k)` (earliest overflow time; in the deep-stiff regime the finest refinement overflows earliest, so the min bounds the common finite window). For each of `n_samples` sample times `t_s` spread across `(0, t_edge]` (biased toward `t_edge`, `f * t_edge` for `f` in evenly-spaced fractions up to **0.95** — the unsafe "agree-then-diverge-in-the-last-bit" window is closed by construction, B-D8 meta-guard), compute the across-refinement state magnitudes `m_k = max(abs(traj_k[idx_k]))` where `idx_k = min(round(t_s/dt_k), overflow_step_k - 1)` (clamp into the finite prefix). If at ANY `t_s` the relative spread `(max(m) - min(m)) / max(abs(m)) >= conv_rtol` → return False (path is `dt`-divergent → artifact). If all `t_s` agree → True (real divergence). A `t_s` where all `m_k == 0` trivially agrees (skip). Errs toward False (uncertain) when the coarse levels disagree — the safe direction.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_simulation_executor.py` a small parser helper + tests. The helper parses RHS over state vars + params exactly as `_run_simulation` does, so `_bounded_observe` receives the same `rhs_exprs` shape:

```python
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

def test_bounded_observe_near_boundary_stiff_uncertain():
    # NEAR-BOUNDARY regime: x'=-200x, dt=0.02 -> dt*lam=4.0 (unstable); dt/2 -> 2.0 (<2.78, stable) so a finer
    # refinement is BOUNDED -> the "BOUNDED at a refinement" escape -> uncertain. Truly bounded decay.
    verdict, info, steps = _bounded({"x": "-200.0*x"}, ["x"], {}, [1.0], 200, 0.02, bound=2.0, t_end=4.0)
    assert verdict == "uncertain"

def test_bounded_observe_deep_unstable_stiff_uncertain():
    # DEEP-UNSTABLE regime (THE regression test for the t*-vs-trajectory hole, B-D8): x'=-1000x, dt=0.05 ->
    # dt*lam=50, and 50/2^3=6.25 > 2.78 -> unstable at EVERY refinement, so it overflows at every level and
    # t_of=[0.95,0.625,0.4375,0.40] CONVERGES (t*-convergence alone would FALSE-REFUTE). But the pre-overflow
    # trajectory is dt-DIVERGENT garbage -> §3d fails -> uncertain. x'=-1000x is a pure decay: TRULY bounded.
    verdict, info, steps = _bounded({"x": "-1000.0*x"}, ["x"], {}, [1.0], 100, 0.05, bound=2.0, t_end=5.0)
    assert verdict == "uncertain" and info["max_abs"] == "trajectory_unconverged"   # caught by §3d, not §3a/§3b

def test_bounded_observe_tstar_near_tend_uncertain():
    # x'=x^2 singularity at t*=1; t_end=1.05 so it DOES overflow inside the window (kind="div") but t*~=1.0 is
    # within conv_rtol of t_end (1.0 > 0.9*1.05=0.945) -> the §3b div gate fires -> uncertain (NOT the breach path).
    verdict, info, steps = _bounded({"x": "x**2"}, ["x"], {}, [1.0], 1050, 0.001, bound=10.0, t_end=1.05)
    assert verdict == "uncertain"
    # control: a wider window puts t* well inside -> refutes (proves the gate, not a blanket uncertain)
    v2, i2, _ = _bounded({"x": "x**2"}, ["x"], {}, [1.0], 2000, 0.001, bound=10.0, t_end=2.0)
    assert v2 == "unbounded"

def test_bounded_observe_budget_exhausted_uncertain():
    # a refuting (diverging) point but per_refine_max_steps too small to take >=2 refinements -> budget exhausted
    verdict, info, steps = _bounded({"x": "x**2"}, ["x"], {}, [1.0], 3000, 0.001, bound=10.0, t_end=2.0,
                                    per_refine_max_steps=4000)   # base 3000 ok, but 2x=6000 > 4000 -> stop
    assert verdict == "uncertain" and info["max_abs"] == "refine_budget_exhausted"
```

(Note on the two stiff tests — both are TRULY bounded decays (`x'=-λx`); a "refute" on either is the soundness bug. The near-boundary case (`λ=200, dt=0.02`) is caught by the "BOUNDED at a refinement" escape (a finer `dt` crosses below 2.78). The deep-unstable case (`λ=1000, dt=0.05`) is the regression test for B-D8: it stays unstable at every refinement and its `t_of` converges, so ONLY §3d (`trajectory_unconverged`) catches it — verified pre-implementation that `_bounded_observe` returns `uncertain` here. Do NOT weaken either assertion; if either fails, the §3d implementation or the `_trajectory_converges` sampling is wrong — fix it, don't relax the test. The confirmed-divergence test (`x'=x²`) must STILL return `unbounded` — its trajectory `dt`-converges, so it passes §3d; if §3d wrongly fails it, the sampling window is too aggressive (lower the top fraction toward `t_edge`).)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -k bounded_observe -v`
Expected: FAIL — `cannot import name '_bounded_observe'`.

- [ ] **Step 3: Implement `_bounded_observe`**

In `valagents/sandbox/runner.py`, after `_converged_monotone`, add:

```python
def _trajectory_converges(div_levels, conv_rtol, np, n_samples=6):
    """§3d — is the pre-overflow trajectory dt-converged? THE divergence discriminator (B-D8): t*-convergence
    alone is NOT sufficient (a deep-unstable stiff instability has a converging t_of too). div_levels: list of
    (traj, overflow_step, dt_k), one per refinement level (all overflowed). A real singularity's solution tracks
    the true solution up to the blow-up, so the across-refinement state magnitudes AGREE at any fixed pre-overflow
    time; a stiff artifact's path is dt-divergent garbage, orders of magnitude apart. Sample the state magnitude
    at several t_s spread across (0, t_edge] (t_edge = earliest overflow time; bias toward the edge — NEVER a
    single early point, where stiff refinements still sit near x0 and spuriously agree). Converged at EVERY t_s
    -> True (real); diverges at any -> False (artifact). Errs toward False (uncertain) when coarse levels disagree."""
    t_edge = min(ov * h for (_, ov, h) in div_levels)
    if t_edge <= 0:
        return False
    for i in range(1, n_samples + 1):
        t_s = (0.95 * i / n_samples) * t_edge                # fractions up to 0.95*t_edge, biased toward the edge
        mags = []
        for (traj, ov, h) in div_levels:
            idx = min(int(round(t_s / h)), ov - 1)           # clamp into the finite prefix [0, ov-1]
            mags.append(float(np.max(np.abs(traj[idx]))))
        mmax = max(abs(m) for m in mags)
        if mmax <= 0.0:
            continue                                         # all zero here -> trivially agree
        if (max(mags) - min(mags)) / mmax >= conv_rtol:      # disagree at this t_s -> dt-divergent path
            return False
    return True

def _bounded_observe(rhs_exprs, var_index, env_base, y0, n_steps, dt, observable, bound, t_end,
                     max_halvings, conv_rtol, per_refine_max_steps, np, npfuncs):
    """§3 honesty check for ONE grid point of a bounded claim (max_abs, op='le', threshold=bound).
    Returns (verdict, info, steps_used). verdict in {'bounded','unbounded','uncertain'}. A refuting point
    (overflow OR finite max_abs>bound) refutes ONLY if its refuting quantity converges monotone+shrinking; a
    DIVERGENCE additionally requires t* strictly inside the window (§3b) AND a dt-converged pre-overflow
    trajectory (§3d — t*-convergence alone is insufficient, B-D8). Anything else -> uncertain. Raises propagate
    -> uncertain upstream. Deterministic: fixed dt->dt/2 sequence; integer step budget."""
    def classify(ns, h):
        traj, overflow = _rk4_integrate_capturing(rhs_exprs, var_index, env_base, y0, ns, h, np, npfuncs)
        if overflow is not None:
            return "div", overflow * h, (traj, overflow, h)
        m = _extract_observable(traj, var_index, observable, np)
        return (("breach", m, None) if m > bound else ("bounded", m, None))

    steps_used = n_steps
    kind, q0, lvl0 = classify(n_steps, dt)
    if kind == "bounded":
        return "bounded", {"max_abs": q0, "t_star": None, "refinements": 0}, steps_used
    seq = [q0]
    div_levels = [lvl0] if kind == "div" else []
    for k in range(1, max_halvings + 1):
        n_k = n_steps * (2 ** k)
        if n_k > per_refine_max_steps:
            return "uncertain", {"max_abs": "refine_budget_exhausted", "t_star": None,
                                 "refinements": k - 1}, steps_used
        steps_used += n_k
        rk, rq, lvl = classify(n_k, dt / (2 ** k))
        if rk == "bounded":                                  # refutation vanished under refinement -> artifact
            note = q0 if kind == "breach" else "diverged_unconfirmed"
            return "uncertain", {"max_abs": note, "t_star": None, "refinements": k}, steps_used
        if rk != kind:                                       # divergence<->breach morph -> receding -> artifact
            return "uncertain", {"max_abs": "morph_unconfirmed", "t_star": None, "refinements": k}, steps_used
        seq.append(rq)
        if kind == "div":
            div_levels.append(lvl)
    refs = len(seq) - 1
    if kind == "div":
        # B-D8: t*-convergence (necessary) + §3b window + §3d pre-overflow trajectory convergence (sufficient).
        # NO "seq[-1] <= seq[0]" clause (finding A stands; the deep-unstable case has DECREASING t_of anyway).
        if not (_converged_monotone(seq, conv_rtol) and seq[-1] < t_end * (1.0 - conv_rtol)):
            return "uncertain", {"max_abs": "refine_budget_exhausted", "t_star": None, "refinements": refs}, steps_used
        if not _trajectory_converges(div_levels, conv_rtol, np):            # §3d: stiff artifact -> uncertain
            return "uncertain", {"max_abs": "trajectory_unconverged", "t_star": seq[-1], "refinements": refs}, steps_used
        return "unbounded", {"max_abs": "diverged", "t_star": seq[-1], "refinements": refs}, steps_used
    # kind == "breach" (no §3d needed: the converged quantity is the physical max_abs, no proxy)
    if _converged_monotone(seq, conv_rtol) and seq[-1] > bound:
        return "unbounded", {"max_abs": seq[-1], "t_star": None, "refinements": refs}, steps_used
    return "uncertain", {"max_abs": "refine_budget_exhausted", "t_star": None, "refinements": refs}, steps_used
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -k bounded_observe -v`
Expected: all five PASS (tune the stiff-test constants in Step 1 if needed so the bounded system stays bounded and the verdict is uncertain — never refute).
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_executor.py
git commit -m "feat(sim): _bounded_observe honesty check (converge-or-uncertain; t*-window; budget sentinel)"
```

---

## Task 6: Wire the bounded path into the `ode_integrate` sweep loop

**Files:**
- Modify: `valagents/sandbox/runner.py` — `_REQUIRED_CEILINGS` (runner.py:303-305) and the `ode_integrate` block (runner.py:360-415).
- Test: `tests/test_simulation_executor.py` (end-to-end `run_plan`).

**Interfaces:**
- Consumes: `_bounded_observe` (T5); `ceil["max_dt_halvings"]`, `ceil["conv_rtol"]` (T1).
- Produces: the executor honors a bounded claim. Selection: `is_bounded = (plan["observable"].get("name") == "max_abs" and plan["sim_criterion"].get("op") == "le")`. For a bounded point, `_bounded_observe` replaces `_rk4_integrate + _extract_observable + _eval_criterion`; `verdict == "bounded"` ⇒ point passes the criterion, `"unbounded"` ⇒ point fails (a confirmed refutation), `"uncertain"` ⇒ the WHOLE run returns `_u(...)`. Cumulative `steps_used` across the sweep is capped at `ceil["max_total_steps"]`. Discrimination (null_overrides) runs `_bounded_observe` per arm. `computed` uses `bounded:`/`unbounded:`/`bounded-discriminating:` prefixes.

- [ ] **Step 1: Write the failing end-to-end tests**

Add to `tests/test_simulation_executor.py` (these use `run_plan` + the `splan` fixture; note `splan`'s default observable is `final_value` — these override to `max_abs`):

```python
def test_run_bounded_pass_confirm():
    # decaying x, max_abs <= 2 across the sweep -> robustly bounded -> confirm
    v = run_plan(splan(observable={"name": "max_abs", "var": "x", "window_frac": "1.0"},
                       sim_criterion={"op": "le", "threshold": ["2.0"]}, robust_frac="1"), cfg())
    assert v.verdict == "pass" and v.result.matched == "confirm" and "bounded" in v.measured

def test_run_bounded_confirmed_divergence_refutes():
    # x' = x^2 diverges for every swept a-irrelevant point (rhs ignores a); singularity well inside t_span
    v = run_plan(splan(rhs={"x": "x**2 + 0*a"}, init={"x": "1.0"}, t_span=["0", "2"], dt="0.001",
                       max_steps=5000,
                       observable={"name": "max_abs", "var": "x", "window_frac": "1.0"},
                       sim_criterion={"op": "le", "threshold": ["10.0"]}, robust_frac="1"), cfg())
    assert v.verdict == "fail" and v.result.matched == "refute" and "unbounded" in v.measured

def test_run_bounded_stiff_artifact_uncertain():
    # numerically stiff but truly bounded decay -> uncertain, NOT refute (the soundness gate, end-to-end)
    v = run_plan(splan(rhs={"x": "-200.0*x + 0*a"}, init={"x": "1.0"}, t_span=["0", "4"], dt="0.02",
                       max_steps=5000,
                       observable={"name": "max_abs", "var": "x", "window_frac": "1.0"},
                       sim_criterion={"op": "le", "threshold": ["2.0"]}, robust_frac="1"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_run_max_abs_with_gt_uses_v1_rule_scoping():
    # max_abs + op != le -> NOT the bounded path; a blow-up uses the v1 rule -> uncertain (no inversion)
    v = run_plan(splan(rhs={"x": "x**2"}, init={"x": "10.0"}, t_span=["0", "50"],
                       observable={"name": "max_abs", "var": "x", "window_frac": "1.0"},
                       sim_criterion={"op": "gt", "threshold": ["1.0"]}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_run_bounded_determinism():
    p = dict(observable={"name": "max_abs", "var": "x", "window_frac": "1.0"},
             sim_criterion={"op": "le", "threshold": ["2.0"]}, robust_frac="1")
    a = run_plan(splan(**p), cfg()); b = run_plan(splan(**p), cfg())
    assert a.measured == b.measured and a.verdict == b.verdict

def test_run_bounded_domain_error_surfaces_label():
    # x' = log(x), x0=1 -> x decays through 0 -> log(neg) -> nan domain error at a finite, in-bounds state.
    # Whole run -> uncertain, and the diagnostic label distinguishes "model ill-posed" from "couldn't confirm".
    v = run_plan(splan(rhs={"x": "log(x) + 0*a"}, init={"x": "1.0"}, t_span=["0", "5"], dt="0.01",
                       observable={"name": "max_abs", "var": "x", "window_frac": "1.0"},
                       sim_criterion={"op": "le", "threshold": ["2.0"]}, robust_frac="1"), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
    assert "domain_error" in (v.result.error or "")

def test_run_bounded_discrimination_uncertain_propagates():
    # null arm a=0 leaves x' = x^2 -> diverges; mechanism arm a large keeps it bounded. If EITHER arm is
    # unconfirmed-uncertain the whole run is uncertain (per-arm honesty). Here construct a clean discriminating
    # pass: mechanism bounded (confirmed), null divergence (confirmed) -> discriminate.
    v = run_plan(splan(rhs={"x": "x**2 - a*x"}, init={"x": "1.0"}, params={},
                       param_sweep={"a": ["1000", "2000", "5"]}, null_overrides={"a": "0"},
                       t_span=["0", "2"], dt="0.001", max_steps=5000,
                       observable={"name": "max_abs", "var": "x", "window_frac": "1.0"},
                       sim_criterion={"op": "le", "threshold": ["10.0"]}, robust_frac="1"), cfg())
    # mechanism arm (large a) damps x below 10; null arm (a=0) -> x^2 diverges -> confirmed -> discriminate
    assert v.verdict == "pass" and "discriminating" in v.measured
```

(The implementer may tune the discrimination test's `a` range so the mechanism arm is genuinely bounded under `bound=10`; the assertion that matters is `pass` + `discriminating` in `measured`, proving the per-arm bounded path ran.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -k "run_bounded or run_max_abs" -v`
Expected: FAIL — the bounded plans currently route through `_extract_observable`/`_eval_criterion`; a divergence raises → uncertain (so the refute/pass-bounded/discrimination assertions fail).

- [ ] **Step 3a: Require the new ceilings**

In `_run_simulation`, extend `_REQUIRED_CEILINGS` (runner.py:303-305) to include the two new knobs:

```python
    _REQUIRED_CEILINGS = ("max_state_vars", "max_expr_nodes", "max_grid_points",
                          "max_steps", "max_total_steps", "min_grid_points",
                          "fixed_point_tol", "min_points_per_axis",
                          "max_dt_halvings", "conv_rtol")
```

- [ ] **Step 3b: Branch the `ode_integrate` per-point evaluation onto the bounded path**

In the `ode_integrate` block, just before the `passes = 0` line (runner.py:385), compute the bounded selection and pull the bounded knobs + per-refinement ceiling:

```python
            is_bounded = (plan["observable"].get("name") == "max_abs"
                          and plan["sim_criterion"].get("op") == "le")
            if is_bounded:
                bound = float(plan["sim_criterion"]["threshold"][0])
                t_end = t1
                max_halv = int(ceil["max_dt_halvings"])
                conv_rtol = float(ceil["conv_rtol"])
                per_refine_max = min(int(plan["max_steps"]), int(ceil["max_steps"]))
                total_budget = int(ceil["max_total_steps"])
                cum_steps = 0
```

Then replace the per-point loop body (runner.py:387-406) so the bounded path uses `_bounded_observe` per arm, propagates uncertain to the whole run, and accumulates the cumulative step budget. New loop:

```python
            passes = 0
            detail = []
            for pov, iov in grid:
                env_base = {**base_params, **pov}
                init_vals = {**base_init, **iov}
                y0 = np.array([init_vals[v] for v in state_vars], dtype=float)
                if is_bounded:
                    vm, im, sm = _bounded_observe(rhs_exprs, var_index, env_base, y0, n_steps, dt,
                                                  plan["observable"], bound, t_end, max_halv, conv_rtol,
                                                  per_refine_max, np, npfuncs)
                    cum_steps += sm
                    if cum_steps > total_budget:
                        return _u(f"bounded refinement work exceeds max_total_steps ({total_budget})")
                    if vm == "uncertain":
                        return _u(f"bounded check uncertain at params={pov} init={iov}: {im['max_abs']}")
                    crit_m = (vm == "bounded")
                    if null_overrides:
                        env_null = {**env_base, **null_parsed}
                        vn, ino, sn = _bounded_observe(rhs_exprs, var_index, env_null, y0, n_steps, dt,
                                                       plan["observable"], bound, t_end, max_halv, conv_rtol,
                                                       per_refine_max, np, npfuncs)
                        cum_steps += sn
                        if cum_steps > total_budget:
                            return _u(f"bounded refinement work exceeds max_total_steps ({total_budget})")
                        if vn == "uncertain":
                            return _u(f"bounded null-arm uncertain at params={pov}: {ino['max_abs']}")
                        crit_n = (vn == "bounded")
                        point_pass = bool(crit_m and not crit_n)
                        detail.append({"params": pov, "init": iov, "bounded_mech": crit_m, "info_mech": im,
                                       "bounded_null": crit_n, "info_null": ino, "discriminate": point_pass})
                    else:
                        point_pass = crit_m
                        detail.append({"params": pov, "init": iov, "bounded": crit_m, "info": im})
                    if point_pass:
                        passes += 1
                    continue
                traj_m = _rk4_integrate(rhs_exprs, var_index, env_base, y0, n_steps, dt, np, npfuncs)
                obs_m = _extract_observable(traj_m, var_index, plan["observable"], np)
                crit_m = _eval_criterion(obs_m, plan["sim_criterion"])
                if null_overrides:
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
```

- [ ] **Step 3c: `computed` string for the bounded path**

Replace the `computed = ...` block (runner.py:407-412) so the bounded path uses bounded-specific wording while the non-bounded path is unchanged:

```python
            frac = passes / gsize
            robust = frac >= rf
            if is_bounded and null_overrides:
                computed = f"bounded-discriminating: {passes}/{gsize} ({frac:.2f} >= {plan['robust_frac']})"
            elif is_bounded:
                kept = "bounded" if robust else "unbounded"
                computed = f"{kept}: {passes}/{gsize} within bound ({frac:.2f} >= {plan['robust_frac']})"
            elif null_overrides:
                computed = f"discriminating: {passes}/{gsize} ({frac:.2f} >= {plan['robust_frac']})"
            else:
                computed = f"robust: {passes}/{gsize} pass ({frac:.2f} >= {plan['robust_frac']})"
```

The `return {"ok": True, ...}` line below (runner.py:413-415) is unchanged.

- [ ] **Step 4: Run the new tests and the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -k "run_bounded or run_max_abs" -v`
Expected: all PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: ALL pass — every pre-existing executor/integration test (final_value, amplitude, linstab, discrimination, work caps, determinism) is untouched because they don't use `max_abs`+`le`.

- [ ] **Step 5: Commit**

```bash
git add valagents/sandbox/runner.py tests/test_simulation_executor.py
git commit -m "feat(sim): wire bounded honesty path into ode_integrate sweep (scoped, per-arm, budgeted, sentinels)"
```

---

## Task 7: Designer prompt + integration / gate tests

**Files:**
- Modify: `valagents/prompts.py:359-391` (`SIMULATION_DESIGNER`).
- Test: `tests/test_simulation_integration.py` (FakeLLM designer → real executor → gate).

**Interfaces:**
- Consumes: the whole bounded executor (Tasks 1–6); `design_simulation` (unchanged — `max_abs` is an existing observable name, `bound` is the existing `sim_criterion` threshold, so no new plan field and no `_FIELDS`/coercion change).
- Produces: the designer can emit a bounded claim; the gate maps a confirmed-unbounded `load_bearing novel_core` claim to `challenged` and a robust-bounded claim to a discounted `survived`.

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_simulation_integration.py`:

```python
BOUNDED_PLAN = {
    "primitive": "ode_integrate", "state_vars": ["x"], "rhs": {"x": "-x + 0*a"},
    "params": {}, "init": {"x": "1.0"}, "t_span": ["0", "5"], "dt": "0.01",
    "param_sweep": {"a": ["0.8", "1.2", "5"]},
    "observable": {"name": "max_abs", "var": "x", "window_frac": "1.0"},
    "sim_criterion": {"op": "le", "threshold": ["2.0"]}, "robust_frac": "1",
    "max_steps": 2000, "max_grid_points": 50, "max_state_vars": 4, "max_expr_nodes": 50,
}
BOUNDED_BODY = "```json\n" + json.dumps(BOUNDED_PLAN) + "\n```"
UNBOUNDED_PLAN = {**BOUNDED_PLAN, "rhs": {"x": "x**2 + 0*a"}, "t_span": ["0", "2"], "dt": "0.001",
                  "max_steps": 5000, "sim_criterion": {"op": "le", "threshold": ["10.0"]}}
UNBOUNDED_BODY = "```json\n" + json.dumps(UNBOUNDED_PLAN) + "\n```"

async def test_bounded_pass_is_discounted_survived():
    s = _store()
    await run_simulation_checks(s, router(BOUNDED_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived"
    assert s.current.claim_graph[0].checks == []          # discounted: no CheckRecord
    assert "max_abs" in sims[0].basis                      # the observable basis branch renders max_abs
    assert "simulation" in s.current.attack_surface.attempted

async def test_bounded_unbounded_challenges():
    s = _store(role="novel_core", load_bearing=True)
    await run_simulation_checks(s, router(UNBOUNDED_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "landed" and sims[0].severity == "fatal"
    assert s.current.verdict_class == "challenged"

def test_prompt_teaches_bounded_claim():
    from valagents.prompts import SIMULATION_DESIGNER
    assert "max_abs" in SIMULATION_DESIGNER and "window_frac" in SIMULATION_DESIGNER
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py -k "bounded or prompt_teaches" -v`
Expected: `test_prompt_teaches_bounded_claim` fails (prompt lacks the bounded guidance + `max_value`/`min_value` already mention `max`, so assert specifically on `max_abs`). The two run tests should already PASS if Task 6 is correct (they exercise the wired executor); if they fail, fix Task 6 — do not weaken them.

- [ ] **Step 3: Teach the bounded claim in `SIMULATION_DESIGNER`**

In `valagents/prompts.py`, in the observable list (prompts.py:369-370) add `max_abs` to the enumerated observables, and add a paragraph after the negative-control paragraph (after prompts.py:378) describing the bounded claim. Concretely, change the observable line to include `max_abs`:

```python
[lo, hi, n]. Give the observable (one of: final_value, mean_window, amplitude, settle_std, max_value, \
min_value, max_abs) with its var and window_frac in (0,1], a structured criterion {{"op": ge|le|gt|lt|in, \
```

and insert this paragraph after the `null_overrides` paragraph:

```python
For a BOUNDEDNESS claim (the dynamics must stay bounded), use observable "max_abs" (peak |var|) with criterion \
{{"op": "le", "threshold": ["<bound>"]}} and robust_frac 1 — bounded must hold across the WHOLE sweep, so any \
confirmed-unbounded point refutes. Use window_frac 1 to require boundedness for ALL t; a window_frac below 1 is \
the WEAKER "eventually bounded" claim (it ignores early transients) — only use it if you mean that. The executor \
treats a genuine divergence (or a breach above the bound) as a FAIL only when it survives dt-refinement, so a \
mere step-size blow-up stays uncertain; budget max_steps with headroom for refinement. \
```

- [ ] **Step 4: Run the new tests and the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py -k "bounded or prompt_teaches" -v`
Expected: all PASS.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: ALL pass (the teeth/gate-purity pins — `test_simulation_does_not_satisfy_magnitude_teeth`, `test_evaluate_ignores_simulation_fields` — stay green; `artifact.py`/`verdict_to_sim_attack` untouched).

- [ ] **Step 5: Commit**

```bash
git add valagents/prompts.py tests/test_simulation_integration.py
git commit -m "feat(sim): teach the bounded claim (max_abs + le bound + window_frac=1 + robust_frac=1)"
```

---

## Self-Review (run against the spec)

**Spec coverage:**
- §1/§2 `max_abs` + scoping (`op="le"` only) → Task 2 (observable), Task 6 (`is_bounded` selection + scoping test `test_run_max_abs_with_gt_uses_v1_rule_scoping`). ✅
- §3 honesty check (classify → refine → converge-or-uncertain) → Task 5 `_bounded_observe`. ✅
- §3a monotone+shrinking → Task 4 `_converged_monotone` + its lucky-pair test. ✅
- §3b `t*`-near-`t_end` → Task 5 (`seq[-1] < t_end*(1-conv_rtol)`) + `test_bounded_observe_tstar_near_tend_uncertain`. ✅
- §3c determinism → Task 6 `test_run_bounded_determinism`; fixed dt→dt/2 sequence + integer `cum_steps`. ✅
- §4 work bounds (`per_refine_max_steps`, cumulative `max_total_steps`) + config knobs → Task 1, Task 5 (budget sentinel), Task 6 (`cum_steps` cap). ✅
- §4 `conv_rtol` justification / BP-1 overflow trigger → documented in Global Constraints + `_DIVERGENCE_MAG` comment; routed to uncertain via the `nan`→raise path (Task 3 `test_capturing_integrator_nan_domain_error_raises`). ✅
- §5 gate reuse + sentinels (`"diverged"`, `"refine_budget_exhausted"`) → Task 6 (unchanged mapping), Task 5 (sentinel strings), Task 7 (`challenged`/`survived` gate tests). ✅
- §6 per-arm discrimination → Task 6 (per-arm `_bounded_observe`, uncertain-in-either-arm → `_u`). ✅
- §7 designer (`max_abs`+`le`+`window_frac=1`+`robust_frac=1`) → Task 7. ✅
- §8 test list → distributed across Tasks 2–7 (pass, confirmed-divergence, stiff-artifact, finite-breach via convergence, scoping, determinism, budget, discrimination, gate). ✅
  - *Gap noted:* §8's explicit "finite-breach survives → refute" and "finite-breach artifact → uncertain" end-to-end executor cases are exercised at the `_bounded_observe` unit level via the breach branch logic, but the `run_plan`-level breach tests are thinner than the divergence ones. **Added coverage instruction:** the Task 5 reviewer should confirm the breach branch (`kind=="breach"`) has at least one converging→unbounded and one receding→uncertain unit test; if absent, add them before marking Task 5 complete (a breach system: `x' = a` constant drift so `max_abs` grows linearly and converges above bound → refute; vs a coarse-dt overshoot that settles under bound at `dt/2` → uncertain).

**Placeholder scan:** no TBD/TODO; every code step shows complete code. The stiff-test and discrimination-test constants carry an explicit "tune to keep the system genuinely bounded / mechanism-bounded" instruction rather than a placeholder — the assertion (uncertain / pass+discriminating) is fixed. ✅

**Type consistency:** `_bounded_observe` returns `(verdict:str, info:dict, steps_used:int)` everywhere it's defined (Task 5) and called (Task 6). `_converged_monotone(seq, rtol) -> bool` consistent (Task 4 def, Task 5 calls). `_rk4_integrate_capturing(...) -> (traj, overflow_step)` consistent (Task 3 def, Task 5 `classify`). `_eval_expr(..., allow_nonfinite=False)` consistent (Task 2 def, Task 3 `deriv` call). `_DIVERGENCE_MAG` defined once (Task 3). ✅

**Decision-log addition:** BP-1 (overflow trigger: `inf`/`_DIVERGENCE_MAG` = divergence; `nan` = domain error → `_DomainError` → uncertain; `_eval_expr.allow_nonfinite` relaxes only the finiteness raise) is already in the spec's §9 decision log — keep code and spec in sync if either moves.

## Review Routing (for the controller running subagent-driven execution)

- **Tasks 4 and 5 carry the only logic with no external ground truth** (`_converged_monotone` and the `_bounded_observe` decision tree). Every prior Lens-3 slice's real bug lived exactly here (the params↔param_sweep circularity hole; persistence-vs-`t*`). Point the Task 4 and Task 5 reviewers specifically at the **convergence boundary** — the monotone/shrinking predicate and the classify→refine→decide branches — NOT at the integrator arithmetic (Task 3) which has clear ground truth.
- **Finding A** (the dropped `seq[-1] <= seq[0]` clause) is resolved in the plan; the Task 5 reviewer should confirm the div branch relies on `_converged_monotone` + the §3b window guard alone, and that a from-below-converging singularity is NOT false-uncertained.
- **Finding B** (the inf−inf→nan-in-one-stage edge) is documented in the Task 3 integrator docstring as a known safe-erring limitation — the Task 3 reviewer should confirm the docstring note is present, not treat the edge as a defect to fix.
- The Task 5 reviewer must also confirm the **finite-breach branch** has both a converging→unbounded and a receding→uncertain unit test (the self-review's noted gap).
