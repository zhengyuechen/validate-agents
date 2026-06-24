# Spec 2 Lens 3 slice-2 — Negative-Control / Discrimination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Lens 3 simulation executor with an opt-in **negative-control / discrimination** mode: a `null_overrides` parameter-off arm run alongside the mechanism arm, so a PASS means the claimed behavior is *attributable* to the proposed mechanism (present with it, absent without it), and "behavior present without the mechanism" now refutes.

**Architecture:** One new optional plan field (`null_overrides`) switches `_run_simulation` into a two-arm loop: per grid point, integrate the mechanism arm and the null arm (same RHS/init/sweep point, only the mechanism's coupling params overridden), and require `criterion(mechanism) AND NOT criterion(null)`. PASS iff ≥ `robust_frac` of points discriminate. The gate mapping (`verdict_to_sim_attack`, `run_simulation_checks`, `_evaluate`) is **unchanged** — only the executor's confirm/refute computation gets stricter. Empty `null_overrides` reproduces v1 single-arm robustness byte-for-byte.

**Tech Stack:** Python, Pydantic v2, the existing safe expression-tree evaluator + deterministic RK4 + sweep, numpy, subprocess sandbox, pytest (`conda run -n cosci-reproduce python -m pytest`).

## Global Constraints

Copied from `docs/2026-06-24-validate-agents-spec2-lens3-negcontrol-design.md` (NC-D1..D5). Every task implicitly includes these:

- **NC-D1 — opt-in parameter-override null.** `null_overrides: dict[str,str]` (param name → off-value); the null arm is the *same* dynamics with the mechanism's coupling at its off-value. Empty → v1 single-arm robustness (backward-compatible). NOT a free-form `rhs_null`.
- **NC-D2 — discrimination criterion.** Per grid point, `discriminate = criterion(mechanism) AND NOT criterion(null)`; PASS iff `≥ robust_frac` of points discriminate; else FAIL.
- **NC-D3 — gate mapping UNCHANGED.** confirm → discounted `survived` (no `CheckRecord`, no `independent_sources`, no route to `internally_validated`); refute → `landed` → `challenged` (`fatal` iff `load_bearing AND role=="novel_core"` else `major`); uncertain → no-op. `verdict_to_sim_attack`, `run_simulation_checks`, `IdeaArtifact._evaluate`, and the L2-D9 teeth (`"simulation"` decisive-only, does NOT satisfy mandatory `"magnitude"`) are NOT touched.
- **NC-D4 — fail-closed extended to two arms.** `null_overrides` keys must be declared params (`⊆ params ∪ param_sweep`, never a state var / undeclared name) → else uncertain; values restricted-parsed numbers; total-work cap `× n_arms`; a non-finite/complex/bad-window in EITHER arm → uncertain. All v1 guards retained.
- **NC-D5 — still `ode_integrate` only.** No new primitive/observable. F1 (no arbitrary code), F3 (code judges), finite-real, determinism — all as v1.

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `valagents/computation.py` | add `null_overrides` field to `ComputationPlan` | 1 |
| `valagents/sandbox/runner.py` | `_run_simulation` two-arm discrimination logic + `null_overrides` validation + total-work `× n_arms` | 1 |
| `valagents/agents/simulation_designer.py` | `_FIELDS` whitelist gains `null_overrides` | 2 |
| `valagents/prompts.py` | `SIMULATION_DESIGNER` prompt explains `null_overrides` | 2 |
| `tests/test_simulation_executor.py` | discrimination executor tests | 1 |
| `tests/test_simulation_integration.py` | designer + gate-mapping tests | 2 |

**Interfaces that already exist (consume verbatim):**
- `_run_simulation(plan: dict)` in `valagents/sandbox/runner.py` (current single-arm sweep loop at lines ~322–342); helpers `_rk4_integrate`, `_extract_observable`, `_eval_criterion`, `_build_grid`, `_parse_number`, `_npfuncs`, `_u`; `_SIM_REQUIRED`/`_SIM_CAPS`. `run_plan(plan, cfg)` injects `_sim_ceilings` for `kind=="simulation"`.
- `verdict_to_sim_attack(v, target_claim_id, fatal_eligible, tick=0)` and `run_simulation_checks(store, llm, cfg, tick=0)` — UNCHANGED; do not edit.
- `tests/test_simulation_executor.py`: `cfg()`, `splan(**kw)` (builds a single-arm simulation `ComputationPlan`), `run_plan`. `tests/test_simulation_integration.py`: `cfg()`, `_store(role=, load_bearing=, mechanistic=)`, `router(body)`, `PLAN` dict, `design_simulation`, `run_simulation_checks`.

**Test command (all tasks):** `conda run -n cosci-reproduce python -m pytest tests/ -q`

---

### Task 1: `null_overrides` model field + two-arm discrimination executor

**Files:**
- Modify: `valagents/computation.py` (add `null_overrides` to the simulation block of `ComputationPlan`)
- Modify: `valagents/sandbox/runner.py` (`_run_simulation`: validation + `n_arms` + total-work + two-arm loop)
- Test: `tests/test_simulation_executor.py`

**Interfaces:**
- Produces: `ComputationPlan.null_overrides: dict[str, str] = {}`; `_run_simulation` discrimination mode (verdict `confirm` iff `≥ robust_frac` of points satisfy `criterion(mechanism) AND NOT criterion(null)`; `computed` says `"discriminating: P/G …"`); single-arm path byte-identical when `null_overrides` is empty (`computed` stays `"robust: P/G pass …"`).

- [ ] **Step 1: Write the failing executor tests**

Add to `tests/test_simulation_executor.py`:

```python
def test_single_arm_backward_compat():
    # no null_overrides -> v1 single-arm; computed keeps the v1 "robust: ... pass" format
    v = run_plan(splan(), cfg())
    assert v.verdict == "pass" and "robust" in v.measured and "pass" in v.measured
    assert "discriminating" not in v.measured

def test_discriminating_pass():
    # mechanism (a in [0.8,1.2]) decays x below 0.2; null (a=0) leaves x at 1.0 (criterion fails) -> discriminate
    v = run_plan(splan(null_overrides={"a": "0"}), cfg())
    assert v.verdict == "pass" and v.result.matched == "confirm"
    assert "discriminating" in v.measured

def test_behavior_without_mechanism_refutes():
    # loose criterion (<=2.0) is met in BOTH arms (x stays <2 even with a=0) -> not attributable -> refute
    v = run_plan(splan(null_overrides={"a": "0"}, sim_criterion={"op": "le", "threshold": ["2.0"]}), cfg())
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_behavior_absent_with_mechanism_refutes():
    # tight criterion (<=1e-9) is NOT met even in the mechanism arm -> not discriminating -> refute
    v = run_plan(splan(null_overrides={"a": "0"}, sim_criterion={"op": "le", "threshold": ["1e-9"]}), cfg())
    assert v.verdict == "fail" and v.result.matched == "refute"

def test_null_override_undeclared_param_uncertain():
    v = run_plan(splan(null_overrides={"zzz": "0"}), cfg())   # zzz is not a declared param
    assert v.verdict == "uncertain" and not v.result.ok

def test_null_override_state_var_uncertain():
    v = run_plan(splan(null_overrides={"x": "0"}), cfg())     # x is a state var, not a param
    assert v.verdict == "uncertain" and not v.result.ok

def test_null_arm_blowup_uncertain():
    # null arm a=-1e6 -> dx/dt = 1e6*x -> blows up -> non-finite in the null arm -> uncertain
    v = run_plan(splan(null_overrides={"a": "-1e6"}), cfg())
    assert v.verdict == "uncertain" and not v.result.ok

def test_total_work_counts_both_arms():
    # grid 120 * n_steps 10_000 = 1.2M (< 2M for 1 arm) but 2.4M at x2 -> total-work cap fires in discrimination
    v = run_plan(splan(null_overrides={"a": "0"}, param_sweep={"a": ["0.8", "1.2", "120"]},
                       t_span=["0", "10"], dt="0.001", max_steps=200_000, max_grid_points=400), cfg())
    assert v.verdict == "uncertain" and not v.result.ok
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -q`
Expected: the new tests FAIL (`null_overrides` not a field → `ComputationPlan(**...)` raises in `splan`, OR it is ignored and the executor runs single-arm so the discrimination assertions fail).

- [ ] **Step 3: Add the `null_overrides` model field**

In `valagents/computation.py`, in the `ComputationPlan` simulation block, add the field immediately after `init_sweep: dict[str, list[str]] = {}`:

```python
    null_overrides: dict[str, str] = {}   # negative-control: param -> off-value; non-empty -> discrimination mode
```

- [ ] **Step 4: Validate `null_overrides` + compute `n_arms` in `_run_simulation`**

In `valagents/sandbox/runner.py`, inside `_run_simulation`, immediately AFTER the reserved-name shadow check (the `if reserved: return _u(...)` block) and BEFORE `local = {n: sympy.Symbol(n) for n in allowed}`, insert:

```python
        null_overrides = plan.get("null_overrides", {})
        declared_params = set(plan.get("params", {})) | set(plan.get("param_sweep", {}))
        bad_null = set(null_overrides) - declared_params
        if bad_null:                                  # NC-D4: a null override may only touch a declared coupling param
            return _u(f"null_overrides reference undeclared/non-param names: {sorted(bad_null)}")
        n_arms = 2 if null_overrides else 1
```

- [ ] **Step 5: Make the total-work cap count both arms**

In `valagents/sandbox/runner.py`, replace the existing total-work check:

```python
        if ceil and gsize * n_steps > int(ceil.get("max_total_steps", 0)):
            return _u(f"total work {gsize}*{n_steps} exceeds max_total_steps")
```

with:

```python
        if ceil and gsize * n_steps * n_arms > int(ceil.get("max_total_steps", 0)):
            return _u(f"total work {gsize}*{n_steps}*{n_arms} exceeds max_total_steps")
```

- [ ] **Step 6: Replace the single-arm sweep loop with the two-arm loop**

In `valagents/sandbox/runner.py`, replace the block from `# fixed params/init` through the `return {...}` (the current `base_params`/`base_init`/`passes`/`detail`/`for pov, iov in grid` loop/`frac`/`robust`/`return`) with:

```python
        # fixed params/init
        base_params = {k: parse_num(v) for k, v in plan.get("params", {}).items()}
        base_init = {k: parse_num(v) for k, v in plan["init"].items()}
        null_parsed = {k: parse_num(v) for k, v in null_overrides.items()}
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
        robust = frac >= parse_num(plan["robust_frac"])
        if null_overrides:
            computed = f"discriminating: {passes}/{gsize} ({frac:.2f} >= {plan['robust_frac']})"
        else:
            computed = f"robust: {passes}/{gsize} pass ({frac:.2f} >= {plan['robust_frac']})"
        return {"ok": True, "computed": computed,
                "matched": "confirm" if robust else "refute",
                "detail": detail}
```

- [ ] **Step 7: Run the executor suite + the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_executor.py -q`
Expected: PASS (the 8 new tests + all existing single-arm tests — the v1 `"robust: … pass"` string is preserved byte-for-byte).
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (no regression — the change is additive and the single-arm path is unchanged).

- [ ] **Step 8: Commit**

```bash
git add valagents/computation.py valagents/sandbox/runner.py tests/test_simulation_executor.py
git commit -m "feat(simulation): negative-control two-arm discrimination (null_overrides param-off arm; criterion(mech) AND NOT criterion(null); total-work x n_arms, fail-closed)"
```

(Plain commit message only — NO `Co-Authored-By:` / `Claude-Session:` / attribution trailer.)

---

### Task 2: Designer `null_overrides` support + gate-mapping integration tests

**Files:**
- Modify: `valagents/agents/simulation_designer.py` (`_FIELDS` adds `"null_overrides"`)
- Modify: `valagents/prompts.py` (`SIMULATION_DESIGNER` explains `null_overrides`)
- Test: `tests/test_simulation_integration.py`

**Interfaces:**
- Consumes: Task 1's `null_overrides` field + discrimination executor; `verdict_to_sim_attack` / `run_simulation_checks` (UNCHANGED).
- Produces: `design_simulation` accepts a `null_overrides` key from the LLM JSON and constructs it into the plan; end-to-end discrimination via the unchanged attack path.

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_simulation_integration.py` (reuses the existing `PLAN`, `_store`, `router`, `design_simulation`, `run_simulation_checks`, `cfg`):

```python
DISC_PLAN = {**PLAN, "null_overrides": {"a": "0"}}
DISC_BODY = "```json\n" + json.dumps(DISC_PLAN) + "\n```"
NOTNEC_PLAN = {**PLAN, "null_overrides": {"a": "0"}, "sim_criterion": {"op": "le", "threshold": ["2.0"]}}
NOTNEC_BODY = "```json\n" + json.dumps(NOTNEC_PLAN) + "\n```"

async def test_designer_emits_null_overrides():
    s = _store()
    p = await design_simulation(s.current.claim_graph[0], s.current, router(DISC_BODY), cfg())
    assert p is not None and p.null_overrides == {"a": "0"}

async def test_discriminating_pass_is_discounted_survived():
    s = _store()
    await run_simulation_checks(s, router(DISC_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived"            # discounted positive
    assert s.current.claim_graph[0].checks == []            # no CheckRecord injected
    assert "simulation" in s.current.attack_surface.attempted

async def test_behavior_without_mechanism_challenges():
    s = _store(role="novel_core", load_bearing=True)
    await run_simulation_checks(s, router(NOTNEC_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "landed" and sims[0].severity == "fatal"
    assert s.current.verdict_class == "challenged"
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py -q`
Expected: FAIL — `design_simulation` drops `null_overrides` (not in `_FIELDS`), so the plan runs single-arm: `test_designer_emits_null_overrides` fails (`null_overrides == {}`), and the discrimination-dependent assertions fail.

- [ ] **Step 3: Add `null_overrides` to the designer whitelist**

In `valagents/agents/simulation_designer.py`, add `"null_overrides"` to the `_FIELDS` tuple (append it after `"max_expr_nodes"`):

```python
_FIELDS = ("primitive", "state_vars", "rhs", "params", "init", "param_sweep", "init_sweep",
           "t_span", "dt", "observable", "sim_criterion", "robust_frac",
           "max_steps", "max_grid_points", "max_state_vars", "max_expr_nodes", "null_overrides")
```

- [ ] **Step 4: Teach the `SIMULATION_DESIGNER` prompt about `null_overrides`**

In `valagents/prompts.py`, in `SIMULATION_DESIGNER`, add a sentence about attribution after the criterion/robust_frac instructions (before the "Output the plan…" line), and add `null_overrides` to the key list. Insert this sentence:

```python
To test ATTRIBUTION (that the behavior is caused by the proposed mechanism, not incidental), parameterize the \
mechanism's coupling as a named parameter and give its OFF-value in null_overrides (e.g. a coupling "g" with \
null_overrides {{"g": "0"}}); the executor then requires the behavior to appear WITH the mechanism and vanish \
WITHOUT it. Omit null_overrides only if the mechanism cannot be turned off by a parameter.
```

And change the final key list line to include `null_overrides`:

```python
Output the plan as a SINGLE JSON object in a ```json fenced block, with these keys: primitive, state_vars, \
rhs, params, init, param_sweep, init_sweep, t_span, dt, observable, sim_criterion, robust_frac, max_steps, \
max_grid_points, max_state_vars, max_expr_nodes, and (optional) null_overrides."""
```

- [ ] **Step 5: Run the integration suite + the full suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_simulation_integration.py -q`
Expected: PASS — the 3 new tests plus all existing Task-5 integration tests (uncertain→no-mark, the magnitude-teeth pin, gate purity, designer malformed-JSON→None) stay green.
Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS (whole suite).

- [ ] **Step 6: Commit**

```bash
git add valagents/agents/simulation_designer.py valagents/prompts.py tests/test_simulation_integration.py
git commit -m "feat(simulation): designer null_overrides support + negative-control integration (discriminating pass -> discounted survived; behavior-without-mechanism -> challenged)"
```

(Plain commit message only — NO `Co-Authored-By:` / `Claude-Session:` / attribution trailer.)

---

## Notes for the executor

- **Gate untouched.** No task edits `valagents/artifact.py`, `verdict_to_sim_attack`, or `run_simulation_checks`. The existing `test_simulation_does_not_satisfy_magnitude_teeth` and `test_evaluate_ignores_simulation_fields` must stay green.
- **Backward compatibility is load-bearing.** Empty `null_overrides` (the model default) must run the v1 single-arm path with the v1 `"robust: P/G pass (…)"` computed string — do not change that string for the single-arm branch (existing tests assert `"robust"`/`"pass"` substrings).
- **Fail-closed both arms:** a non-finite/complex/bad-window in EITHER arm is caught by the existing `try → _u` wrapper; the `null_overrides`-key check and the `× n_arms` total-work cap are the only new fail-closed gates.
- **NC-D3 line not to cross:** a discriminating PASS is still only a `survived` attack — it must NOT inject a `CheckRecord` or reach `internally_validated`. The `claim.checks == []` assertion in Task 2 guards this.
