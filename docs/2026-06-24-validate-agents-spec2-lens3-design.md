# validate-agents — Spec 2 Lens 3 Design (Toy-Model / Simulation Executor)

- **Date:** 2026-06-24
- **Status:** Approved design, pending implementation plan
- **Builds on:** the symbolic lens (`docs/2026-06-23-validate-agents-spec2-design.md`) and the magnitude lens (`docs/2026-06-23-validate-agents-spec2-lens2-design.md`) — the same `ComputationPlan`-style frozen-plan discipline, the same subprocess sandbox + restricted parser, the same F1/F3/F4 rules, and the same anti-laundering teeth rule (L2-D9). And on Spec-1's Red-team attack machinery and the rev-4 mechanistic claims.
- **Status line:** *Lens 3 upgrades the reasoned mechanism check into an executed toy-model simulation — does the proposed dynamics actually produce the claimed behavior, robustly — while keeping the structured-plan-no-code rule and the pure gate.*
- **One-line goal:** Make "does the idea's own mechanism produce the claimed behavior, robustly across a preregistered sweep" a **code-computed, frozen-plan demonstration**, not an LLM judgment — and treat the result **conservatively**: a robust fail is a serious *challenge*, a robust pass is *discounted positive support*, never a refutation and never a standalone validation.

---

## 1. Scope

### In scope (this slice — `ode_integrate` + robustness sweep only)
- A **Simulation-Designer** (LLM) that, per **mechanistic** claim, emits a **structured** `SimulationPlan` with `kind="simulation"`, `primitive="ode_integrate"` (no runnable code — F1).
- A **trusted Executor** (code, not an LLM): a deterministic fixed-step **RK4** integrator over numpy, driven by a **safe expression-tree evaluator** of the restricted-parsed RHS; a **robustness sweep** over declared parameter *and* initial-condition ranges; a **code verdict** (F3).
- **Gate integration via the ATTACK path** (the core decision — §4): a robust **fail** → a landed `type="simulation"` attack → `challenged`; a robust **pass** → a `survived` attack, **discounted** (no passing claim, no route to validation); `uncertain` → no attack (F2 fallback).
- **Robustness sweep is the v1 anti-overfitting discipline.** Negative-control / discrimination is **deferred to a slice-2** (mirrors `sensitivity_ratio` → `discriminating_margin`).

### Out of scope (named so the seams are clear)
- **Other primitives** — `iterated_map`, `monte_carlo`, `linear_stability` — are **named in the schema's `primitive` Literal upfront** but built later. (`monte_carlo` will need a seeded RNG for determinism — flagged.)
- **Negative-control / discrimination** simulation (the behavior must vanish in a declared null variant) — slice-2.
- **`growth_rate`** observable — deferred (most ambiguity-prone; needs a precise window/epsilon/fit spec). v1 ships the unambiguous extractors only.
- A `bounded` observable that would make a numerical blow-up a *deliberate fail* — deferred; in v1 a blow-up is `uncertain` (§3.7).
- **Arbitrary generated code + container isolation** (the original lens-3 seed framing) — explicitly NOT this design. Deferred until real container/VM infrastructure exists; this machine has none. A best-effort local arbitrary-code sandbox is rejected for validation-grade results.
- Verifying that the toy model's *structure/parameters are physically correct* — they are LLM-authored, frozen, and auditable, but **not grounded** (the loud caveat, §6). Grounding is a Spec-3 follow-on.

---

## 2. The non-negotiable rules (preregistration discipline)

1. The Simulation-Designer **preregisters, before any execution**: `primitive`, `state_vars`, `rhs`, `params`/`init`, `param_sweep`/`init_sweep`, `t_span`/`dt`, `observable`, `criterion`, `robust_frac`, and the typed caps. The plan is **frozen** before the Executor runs; the Executor runs **only** the frozen plan.
2. **No arbitrary code (F1).** Each `rhs[var]` is parsed with the symbolic lens's restricted `parse_expr` (declared-symbol `local_dict`, whitelisted `global_dict`, `__builtins__={}`, `"__"`-rejected). Evaluation is **`eval`-free and `lambdify`-free** — a trusted expression-tree walk (§3.3). numpy performs float arithmetic only, never executes plan strings.
3. **The verdict is computed in CODE (F3).** The observable comes from a **fixed code vocabulary**; the `criterion` is a **structured** comparison (`op` + `threshold`); the sweep aggregation (`pass-fraction ≥ robust_frac`) is code. **`confirm_if`/`refute_if` are display-only glosses, never evaluated.** No LLM ever sees the result and decides.
4. **Isolation = subprocess + rlimits + no-network + restricted parser + size caps (F4).** Sufficient *because* no arbitrary code runs. The Task-2 sandbox (`executor.py::run_plan`) is reused verbatim; `runner.py` gains a `_run_simulation` branch.
5. **Conservative gate semantics.** A robust fail → **challenged**, never refuted. A robust pass → a **discounted survived** attack, never a passing claim and never a standalone validation. `uncertain` → the reasoned mechanism check stands (F2).
6. **Determinism.** Fixed-step RK4, no adaptive stepping, no RNG → a frozen plan yields a bit-reproducible verdict (re-runnable, like lenses 1–2). Every artifact saved (plan, per-grid-point observable table, result JSON).

---

## 3. Execution model

### 3.1 `SimulationPlan` (extend the computation models; `kind="simulation"`)

`valagents/computation.py` — a new model (or `ComputationPlan` extension) with simulation fields optional/defaulted so the other kinds are unaffected:

```python
primitive:   Literal["ode_integrate", "iterated_map", "monte_carlo", "linear_stability"]  # v1 builds ode_integrate
state_vars:  list[str] = []                    # e.g. ["x", "y"]
rhs:         dict[str, str] = {}               # {"x": "-y + a*x*(1 - x**2 - y**2)", "y": "x + a*y*(1 - x**2 - y**2)"}
params:      dict[str, str] = {}               # fixed scalars not swept, e.g. {"a": "0.5"}
init:        dict[str, str] = {}               # {"x": "0.1", "y": "0.0"}
param_sweep: dict[str, list[str]] = {}         # {"a": ["0.3", "0.7", "5"]} -> [lo, hi, n]
init_sweep:  dict[str, list[str]] = {}         # {"x": ["0.05", "0.2", "3"]} -> [lo, hi, n]   (EXPLICIT, §1 tightening)
t_span:      list[str] = []                    # ["0", "200"]
dt:          str = ""                          # fixed RK4 step, e.g. "0.01"
observable:  dict = {}                         # {"name": <vocab>, "var": "x", "window_frac": "0.2"}
criterion:   dict = {}                         # {"op": "in", "threshold": ["0.9", "1.1"]}
robust_frac: str = ""                          # K in [0,1]: PASS iff criterion holds on >= K of the grid
max_steps:        int = 0                       # plan-declared caps (preregistered, audited)
max_grid_points:  int = 0
max_state_vars:   int = 0
max_expr_nodes:   int = 0
target_claim_id:  str | None = None            # the mechanistic claim this simulation tests
confirm_if / refute_if: str = ""               # DISPLAY GLOSSES ONLY — never evaluated
```

### 3.2 Sweep grid and precedence
- The grid is the **Cartesian product of `param_sweep` × `init_sweep`**; each entry `[lo, hi, n]` expands to `n` evenly spaced values in `[lo, hi]` inclusive. Grid size = ∏ of all `n` across both sweeps.
- **Precedence:** a swept name overrides its fixed counterpart — `param_sweep[k]` overrides `params[k]`; `init_sweep[v]` overrides `init[v]`. Fixed values are the defaults for everything not swept.
- **A genuine sweep is required.** A grid with fewer than `min_grid_points` points (config ceiling, §3.8) **cannot establish robustness → `uncertain`** (no attack, no mark). A single fixed point is not a sweep — this closes the degenerate path where a no-sweep plan trivially "passes" (`pass_fraction = 1.0 ≥ robust_frac`), marks `"simulation"` attempted, and pads the attack-surface category count with a non-robust result. Robustness is the v1 discipline (L3-D3); the executor enforces that a real sweep happened.

### 3.3 Safe expression-tree evaluator (`_eval_expr(node, env)`)
The **only** path that turns an RHS into a number. Operates on an **already restricted-parsed** SymPy `Expr` (built with the lens-1 `parse_expr`: whitelist + `__builtins__={}` + `"__"`-reject). Recursive walk:
- `Symbol(name)` → `env[name]` (a finite real float / numpy array); **unbound → raise → uncertain**.
- `Integer` / `Float` / `Rational` / `NumberSymbol` (`pi`, `E`) → float.
- `Add(*a)` → sum of evaluated args; `Mul(*a)` → product.
- `Pow(b, e)` → `base ** exp`, **narrowed** (§3.4).
- `Function` in the whitelist (`sin, cos, tan, exp, log, sqrt, Abs, sign, tanh`) → the corresponding **numpy ufunc** on the evaluated arg.
- **any other node type → raise → uncertain.**

Two layers of defense, like lens 1: the restricted *parse* bounds which symbols/functions can appear; the *evaluator* then executes only whitelisted node **types**. No `lambdify`, no generated source, no `eval`.

### 3.4 Finite-real-float requirement (hard)
**Every evaluator output and every trajectory value must be a finite real float.** After each evaluation/step the runner checks `np.isfinite` AND real-valued (reject complex results and non-float/object dtypes). Specifically:
- `Pow` is **narrow**: numeric powers are allowed, but an operation that leaves the real domain (e.g. a negative base to a fractional exponent → complex / NaN) is rejected. **No complex continuation in v1.** Such a result → `uncertain`.
- Any complex value, object-dtype array, `NaN`, or `Inf` produced **anywhere** (an RHS evaluation, an RK4 step, or an observable) → the **entire verdict is `uncertain`**, with the offending grid point and quantity recorded **loudly** in the basis. (A numerical blow-up is treated as solver/step trouble, not scientific failure, in v1 — §6 caveat; a future `bounded` observable would make blow-up a deliberate fail.)

### 3.5 Deterministic RK4 + observable vocabulary
- **Integrator:** classical fixed-step RK4 over numpy. `n_steps = ceil((t_end − t_start) / dt)`. Deterministic — no adaptive stepping, no RNG. Same frozen plan → identical trajectory.
- **Observable vocabulary (v1, trusted numpy extractors; each returns one finite real float):**
  - `final_value(var)` — value at `t_end`.
  - `mean_window(var, window_frac)` — mean of `var` over the last `window_frac` of the trajectory.
  - `amplitude(var, window_frac)` — `(max − min) / 2` of `var` over the last `window_frac` (oscillation amplitude / limit-cycle size).
  - `settle_std(var, window_frac)` — standard deviation of `var` over the last `window_frac` (small ⇒ converged to a fixed point).
  - `max_value(var, window_frac)` / `min_value(var, window_frac)` — extrema over the last `window_frac`.
  - (`growth_rate` — **deferred**, §1; `time_to_threshold` also deferred — it brings event-detection semantics.)
- **`window_frac` validity:** require `0 < window_frac ≤ 1`; the window is the last `ceil(window_frac × n_samples)` samples. A `window_frac` outside `(0, 1]`, or a window with **fewer than 2 samples** for `amplitude`/`settle_std` (which are meaningless on < 2 points; the others need ≥ 1), → **`uncertain`**.
- **Structured criterion:** `op ∈ {ge, le, gt, lt, in}` against `threshold`. `in` means `threshold = [lo, hi]` and passes iff `lo ≤ observable ≤ hi` (**inclusive**). Evaluated in code per grid point.

### 3.6 Verdict logic (code, no LLM)
1. Validate caps (§3.8); on any breach → `uncertain`.
2. Parse each `rhs[var]` (restricted); on parse error or `__` → `uncertain`.
3. Build the grid (§3.2). For each grid point: integrate (RK4) → extract the observable → finite-real check (§3.4) → evaluate the structured criterion → record per-point pass/fail.
4. **PASS** (`matched="confirm"`) iff `pass_fraction ≥ robust_frac` (robust). **FAIL** (`matched="refute"`) iff `pass_fraction < robust_frac` (knife-edge / inert). Any non-finite anywhere → **UNCERTAIN** (overrides, §3.4).
5. `result.computed` summarizes, e.g. `"robust: 21/25 grid points pass (0.84 >= 0.80)"`; the per-grid-point observable table is saved as an artifact.

### 3.7 Fail-closed map (executor → verdict)
`ok=False → uncertain`; `matched="confirm" → pass`; `matched="refute" → fail` (reusing the existing `_verdict` in `executor.py`). Causes of `ok=False`: missing required field, a cap breach, a parse/`__` rejection, an unbound symbol, a non-whitelisted node, or any non-finite/complex value.

### 3.8 Resource & expression caps — two layers, fail-closed
- **Plan-declared caps** (preregistered, audited): `max_steps`, `max_grid_points`, `max_state_vars`, `max_expr_nodes`.
- **Config ceilings** — a new `SimCfg` in `valagents/config.py`, absolute hard limits the plan cannot exceed. Defaults: `max_state_vars=8`, `max_expr_nodes=200`, `max_grid_points=400`, `max_steps=200_000`, **`max_total_steps=2_000_000`** (the total-work cap, below), **`min_grid_points=4`** (the floor below which robustness cannot be assessed, §3.2).
- **Plan-declared caps must be positive.** The schema defaults `max_steps`/`max_grid_points`/`max_state_vars`/`max_expr_nodes` to `0`; the runner treats a missing/zero (≤ 0) plan cap as **invalid → `uncertain`**, never as "unlimited" and never as accidentally stricter. A frozen plan must preregister positive caps.
- **Total-work cap (fail early and loud).** The runner computes `total_steps = grid_size × n_steps` and rejects → `uncertain` if `total_steps > config.sim.max_total_steps`. This fails *before* integrating rather than leaning on the wall timeout (worst case under the per-axis ceilings alone would be `400 × 200_000 = 80M` steps).
- The runner rejects → `uncertain` (loud error) if **any** of: a plan cap is ≤ 0; a plan-declared cap exceeds its config ceiling; `len(state_vars) > max_state_vars`; any parsed RHS node-count (`count_ops` / preorder length) `> max_expr_nodes`; grid size `> max_grid_points`; **grid size `< min_grid_points`** (not a real sweep); `n_steps > max_steps`; **`grid_size × n_steps > max_total_steps`**. Expression-size caps bound the evaluator's cost even though no code runs — defense-in-depth against a pathological parsed expression.

---

## 4. Gate integration — the conservative ATTACK-path mapping

A simulation verdict becomes an **`Attack(type="simulation")`** on the `target_claim_id` (a `type=="mechanistic"` claim). The wiring runs per mechanistic claim, capped (like `inject_limit_checks` caps limits at 3). **If the idea has no mechanistic claim, Lens 3 is a no-op** — it never invents a mechanism target just to run.

| Verdict | Meaning | Gate effect |
|---|---|---|
| **FAIL** (robust criterion not met) | the mechanism-as-modeled does **not** produce the claimed behavior across the swept ranges | **landed** `type="simulation"` attack → **`challenged`**. Severity **`fatal`** (→ `severe_objection`) iff the target claim is **`load_bearing` AND `role == "novel_core"`**; else **`major`** (→ `open_objection`). **Never `refuted`.** |
| **PASS** (robust) | the toy model produces the behavior robustly across the grid | **`survived`** attack, severity `minor`. **Discounted:** creates **no `CheckRecord`**, sets **no `independent_sources`**, injects **no claim**, and is **never** a route to `internally_validated` by itself (identical treatment to a survived magnitude attack). |
| **UNCERTAIN** (parse / cap / non-finite / no-plan) | couldn't decisively simulate | **no attack** — the reasoned mechanism check (completion / Prover) stands (**F2**). |

**Mapping function** (code, no LLM): `verdict_to_sim_attack(v, target_claim_id, fatal_eligible, tick) -> Attack`: `confirm → Attack(type="simulation", status="survived", severity="minor")`; `refute → Attack(type="simulation", status="landed", severity=("fatal" if fatal_eligible else "major"))`, where `fatal_eligible = claim.load_bearing and claim.role == "novel_core"`. The `basis` surfaces the robust fraction, the observable, the criterion, and (loudly) any non-finite note.

**Anti-laundering (L2-D9 carried forward).** A **decisive** verdict (survived/landed) adds `"simulation"` to `AttackSurface.attempted`; an `uncertain`/fail-closed run adds **nothing**. So a non-run can never satisfy the teeth check.

**`"simulation"` does NOT satisfy the mandatory `"magnitude"` requirement.** `_thin_attack_surface()` still requires `"magnitude" in attempted` *and* `len(set(attempted)) ≥ min_attack_categories`. `"simulation"` is an **additional** probe category (it helps the category count) but cannot replace the magnitude requirement. **The gate `_evaluate()` is NOT changed** — simulation flows entirely through `attacks` and `attack_surface.attempted`, which the gate already reads; a landed `fatal`/`major` simulation attack maps to `needs_experiment`/`severe_objection`|`open_objection` exactly as a magnitude attack does.

---

## 5. Sandbox / security (reuse Task-2; add `_run_simulation`)
- `executor.py::run_plan` (subprocess + `RLIMIT_CPU`/`RLIMIT_AS` + wall timeout + minimal env + artifact saving) is reused **unchanged**; it already dispatches on `plan["kind"]`. `runner.py` gains a `_run_simulation` branch beside `_run_symbolic`/`_run_magnitude`. The subprocess imports only `sympy` + `numpy`; no network, no filesystem writes beyond the artifact dir.
- Two-layer expression safety (restricted `parse_expr` + whitelist-node `_eval_expr`) + the finite-real-float gate + the size caps make this **sufficient because no arbitrary code runs (F4)** — the same justification as lenses 1–2, extended to the larger numeric surface.
- **Artifacts:** `plan.json`, `result.json`, `stdout`/`stderr` (existing), plus the **per-grid-point observable table** (which points passed) — the audit trail behind a robust pass/fail. A non-finite blow-up is recorded loudly there and in `result.computed`.

---

## 6. The loud caveat (ship in the basis; state prominently)

> **A Lens-3 PASS means the toy model coheres with the proposed mechanism under a frozen sweep; it does not validate the theory.** The model's structure and parameters are LLM-authored (frozen, auditable, but **not grounded**), the dynamics are a deliberate simplification, the sweep covers only the declared parameter and initial-condition ranges, and agreement is **necessary, not sufficient**. **A Lens-3 FAIL means the proposed mechanism failed its own preregistered toy demonstration — a serious challenge, but not a contradiction with established knowledge** (that is lens 1's bar). Grounding the model against literature is a Spec-3 follow-on.

---

## 7. Testing (deterministic, no network, real numpy; only the Designer is faked)

- **Evaluator:** `_eval_expr` computes a known RHS correctly; a non-whitelisted node (e.g. `Derivative`, an unknown function) → raises → `uncertain`; an unbound symbol → `uncertain`; a `"__"` RHS → `uncertain`.
- **Finite-real / Pow:** a negative base to a fractional exponent (`(-1)**0.5`) → complex → `uncertain`; an RHS that drives the state to `Inf`/`NaN` → `uncertain` with the loud non-finite record; an object/complex value anywhere → `uncertain`.
- **RK4 correctness + determinism:** `dx/dt = −x` integrates to `x0·e^{−t}` within tolerance; a harmonic oscillator's `amplitude` is stable; the **same frozen plan run twice yields an identical result** (bit-reproducible).
- **Robustness sweep:** a criterion holding across the whole `param_sweep × init_sweep` grid → **PASS**; a knife-edge plan (holds at one grid point, `pass_fraction < robust_frac`) → **FAIL**. Grid precedence: a swept name overrides its fixed counterpart.
- **Caps (each → `uncertain`):** missing required field; a **zero/missing plan cap** (≤ 0); grid size `> max_grid_points`; **grid size `< min_grid_points` (a no-sweep / single-point plan — not a trivial pass)**; `n_steps > max_steps`; **`grid_size × n_steps > max_total_steps` (total-work cap)**; RHS nodes `> max_expr_nodes`; `state_vars > max_state_vars`; a plan cap exceeding its config ceiling.
- **`window_frac` validity (each → `uncertain`):** `window_frac > 1` or `≤ 0`; a window with `< 2` samples for `amplitude`/`settle_std`.
- **Gate mapping (FakeLLM designer + real executor):**
  - robust **FAIL** on a `load_bearing` `novel_core` mechanistic claim → **landed `fatal`** `type="simulation"` attack → `verdict_class == "challenged"`;
  - robust **FAIL** on a non-`load_bearing` (or non-`novel_core`) claim → **`major`** → `challenged`;
  - robust **PASS** → **`survived`** attack AND **no `CheckRecord(pass)` created AND `independent_sources` unset** (the "pass discounted" guard) AND status not `internally_validated` by this alone;
  - **`uncertain` → no attack AND `"simulation"` NOT in `attempted`** (the anti-laundering guard);
  - a **decisive** verdict → `"simulation"` in `attempted`;
  - **no mechanistic claim → no plan, no attack** (no-op).
- **Teeth requirement pin:** an artifact with `"simulation"` in `attempted` but **not** `"magnitude"` → `_thin_attack_surface()` still returns `True` (simulation does not satisfy the mandatory magnitude requirement).
- **Attack type:** the attack is `type="simulation"`, never `"magnitude"`.
- **Gate purity:** `inspect.getsource(IdeaArtifact._evaluate)` contains neither `"simulation"` nor `"primitive"`.

---

## 8. Decision log
- **L3-D1** Structured `SimulationPlan`, **no arbitrary generated code** (F1). Arbitrary-code + container isolation (the seed's framing) is rejected for v1 — no container infra exists locally, and it would break F1/F4. The structured-simulation executor is the natural successor to lenses 1–2.
- **L3-D2** v1 builds **`ode_integrate`** only; `iterated_map`/`monte_carlo`/`linear_stability` are named in the schema but deferred. (`monte_carlo` will need a seeded RNG.)
- **L3-D3** v1 anti-overfitting discipline = **robustness sweep** over `param_sweep × init_sweep`; PASS iff the criterion holds across `≥ robust_frac` of the grid. A grid below `min_grid_points` is **not a sweep → `uncertain`** (closes the degenerate single-point "pass"). **Negative-control / discrimination → slice-2.**
- **L3-D4** Verdict computed in **code** (F3): fixed observable vocabulary + structured `criterion` + sweep aggregation. `confirm_if`/`refute_if` are display-only glosses. The Simulation-Designer emits the plan only and never sees the result.
- **L3-D5** Safe numeric path = a **trusted expression-tree evaluator** over the restricted-parsed SymPy `Expr` (whitelisted node types only); **no `lambdify`, no `eval`, no generated code.** Plus the **finite-real-float** requirement and **narrow `Pow`** (no complex continuation) — non-finite/complex anywhere → `uncertain`.
- **L3-D6** Isolation = subprocess + rlimits + no-network + restricted parser + **typed size/expression caps** (two layers: plan-declared caps under config ceilings) — sufficient because no arbitrary code runs (F4). Plan caps must be **positive** (zero/missing → `uncertain`, never "unlimited"); a **total-work cap** (`grid_size × n_steps ≤ max_total_steps`) fails a pathological sweep early, before the wall timeout.
- **L3-D7** **Conservative gate mapping (attack path):** robust **FAIL → `challenged`** (severity `fatal` iff target is `load_bearing` AND `novel_core`, else `major`), **never refuted**; robust **PASS → discounted `survived`** attack (no `CheckRecord`, no `independent_sources`, no standalone validation); `uncertain → no attack` (F2). `Attack(type="simulation")`.
- **L3-D8** **Anti-laundering (L2-D9 carried):** `"simulation"` added to `attempted` only on a decisive verdict; never on `uncertain`. `"simulation"` is an additional probe category and **does not** satisfy the mandatory `"magnitude"` teeth requirement. `_evaluate()` unchanged.
- **L3-D9** **No-op when no mechanistic claim** — Lens 3 never invents a mechanism target.
- **L3-D10 (loud caveat)** A pass = toy-model coherence under a frozen sweep, **not** truth of the theory (LLM-authored, frozen, not grounded). A fail = the mechanism failed its own preregistered toy demonstration — a serious challenge, not a contradiction with established knowledge. Grounding → Spec-3.
- **L3-D11** **Determinism:** fixed-step RK4, no RNG → bit-reproducible verdict; every artifact (plan, per-grid-point observable table, result) saved.
- **L3-D12 (deferred-slice taxonomy + agreed build order, 2026-06-24)** The deferred work is **three different kinds**, not one "Lens 3 follow-up", and should not be blended: **(a) attribution** — negative-control / discrimination (changes what a PASS *means*: from "robust, not a single-point artifact" to "attributable to the proposed mechanism"); **(b) breadth** — more primitives (`linear_stability`, `iterated_map`, `monte_carlo`); **(c) measurement vocabulary** — more observables (`bounded`, `growth_rate`, `time_to_threshold`). **Agreed priority order:** (1) **negative-control / discrimination** — highest leverage, the only deferred item that strengthens positive evidence (attribution > more primitives for a *validation* system); (2) **`linear_stability`** — cheapest useful breadth (reuses parsed RHS, no long RK4, clear eigenvalue semantics, covers many mechanistic claims); (3) **`bounded`** observable — elevated above the other observables because a blow-up is *sometimes the real scientific failure mode*, not solver trouble (it would invert the v1 blow-up→`uncertain` rule for that one observable); (4) **`iterated_map`** — useful breadth, lower centrality unless target ideas are discrete-dynamics; (5) **`monte_carlo`** — needs seeded determinism + sample-size discipline; (6) **`growth_rate` / `time_to_threshold`** — only after their semantics are pinned (fit rule; event-detection), since each adds a semantic knob that is itself an overfitting surface. Cross-cutting, separate from this list: **source grounding (Spec-3)** — ground the LLM-asserted model values against literature (§7 caveat).

---

## 9. Build slices (order)
1. **Data model + verdict mapping** — `SimulationPlan` (`kind="simulation"`, `primitive` Literal naming all four, all v1 fields + typed caps); `verdict_to_sim_attack`. Unit tests (model + the attack mapping incl. the `fatal_eligible` rule).
2. **Simulation runner + executor branch** — `runner.py` `_run_simulation`: restricted-parse → expression-tree evaluator → deterministic RK4 → observable vocabulary → finite-real gate → robustness sweep → code verdict; the two-layer caps; `SimCfg`. Real-numpy executor tests (RK4 correctness/determinism, sweep PASS/FAIL, every fail-closed/cap/non-finite case).
3. **Simulation-Designer + wiring** — `design_simulation` agent + prompt (structured plan, no verdict); `run_simulation_checks` in the scheduler (per mechanistic claim, capped; attack path; mark `"simulation"` decisive-only; F2 fallback; no-op when no mechanistic claim). Integration tests (FAIL→challenged with the severity rule, PASS→discounted survived, uncertain→no-attack-no-mark, the magnitude-teeth-requirement pin, gate purity).
4. **(later)** slice-2 — negative-control / discrimination; then `linear_stability`, `iterated_map`, `monte_carlo` (seeded), and the `bounded`/`growth_rate` observables.
