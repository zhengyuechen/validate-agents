# validate-agents — Spec 2 Lens 3 Design (`linear_stability` primitive)

- **Date:** 2026-06-24
- **Status:** Approved design, pending implementation plan
- **Builds on:** Lens 3 v1 (`docs/2026-06-24-validate-agents-spec2-lens3-design.md`) + the negative-control slice — the structured `SimulationPlan`, the safe expression-tree evaluator (`_eval_expr`), the restricted param-aware parse + reserved-name/free-symbol guards, the param sweep + grid + `robust_frac`/`min_grid` discipline, the structured `sim_criterion`, and the conservative attack-path gate mapping. This is the **breadth** primitive named next in the L3-D12 build order.
- **Status line:** *`linear_stability` upgrades Lens 3 from "integrate the dynamics" to "linearize the mechanism at its preregistered equilibrium and read the spectrum" — validating (or refuting) that a proposed homogeneous state is linearly stable, OR that it goes linearly unstable (a bifurcation/instability-onset claim), robustly across a parameter sweep, with no time integration.*
- **One-line goal:** Make "does the proposed mechanism's equilibrium have the claimed linear-stability character (stable, or unstable at onset) robustly across the swept parameters" a **code-computed spectral test** — the model preregisters the equilibrium *as a formula in the parameters*, and trusted code verifies it and reads the Jacobian spectrum.

---

## 1. Scope

### In scope
- A `primitive="linear_stability"` compute branch: parse the RHS (existing), parse a **parametric fixed point**, build the symbolic Jacobian once, then per grid point — evaluate the fixed point under that grid's params, **verify `rhs ≈ 0`**, evaluate the Jacobian, compute the **spectral abscissa α(J) = maxᵢ Re λᵢ(J)**, and apply the structured `sim_criterion` to α.
- A new plan field `fixed_point: dict[str,str]` (state var → restricted expression over the **parameters**).
- A config `SimCfg.fixed_point_tol` (the equilibrium-residual tolerance — system-set, not an LLM knob).
- The robustness sweep (`param_sweep`), `robust_frac`, `min_grid`, the size caps, and the **unchanged** gate **status** mapping — all reused.
- The **general** criterion on α: `lt` validates a stability claim, `gt` validates an instability-onset claim. The soundness of the margin sign is enforced by the **prompt**, not the runner (§5).

### Out of scope (deferred / unchanged)
- **Single-arm only.** No `null_overrides` (negative-control *for stability* — "the equilibrium is stable with the mechanism, unstable without it" — is a later composition).
- `iterated_map`, `monte_carlo`, and the `bounded`/`growth_rate`/`time_to_threshold` observables remain deferred (L3-D12).
- No time integration: `t_span`, `dt`, `init`, `init_sweep`, the trajectory observables, and the time caps (`max_steps`, `max_total_steps`) do not apply to this primitive.
- Numerical equilibrium re-solving (Newton root-finding) — explicitly rejected for v1 (LS-D1); parametric fixed points cover the moving-equilibrium case without a root-finder's non-determinism/fakeability surface.
- Source grounding: the model/parameters remain LLM-asserted, frozen, not grounded (the v1 §7 caveat still flies).

---

## 2. The compute (`_run_linear_stability`)

Reuses the shared setup (presence checks, complete-ceiling check, positive-cap + ceiling checks, the param-aware restricted parse of the RHS with the reserved-name and free-symbol guards). Then:

**Once, before the sweep:**
- **Parse the parametric fixed point.** For each `var` in `state_vars`, parse `fixed_point[var]` with the restricted parser over **param symbols only** (`allowed_fp = params ∪ param_sweep keys`, NOT state vars). Reject (→ uncertain) if any `fixed_point[var]` references a symbol outside `allowed_fp` (a fixed-point coordinate may depend on parameters and whitelisted constants/functions, never on a state variable — that would be circular).
- **Build the symbolic Jacobian.** `J_ij = sympy.diff(rhs_expr_i, Symbol(state_var_j))` (the RHS exprs were parsed with the state vars as symbols). Computed once. **Cap each `J_ij`**: `count_ops()+1 ≤ max_expr_nodes` → else uncertain (diff can grow node count; keep the bounded-work discipline post-differentiation).

**Per grid point** (`pov` = the param-sweep override; `param_sweep` is the only sweep axis — `init_sweep` does not apply):
1. `params_env = {**base_params, **pov}`.
2. **Evaluate the fixed point:** `x_star = { var: _eval_expr(fp_expr[var], params_env) for var in state_vars }` (finite-real guaranteed by `_eval_expr`; a non-whitelisted node or non-finite → raises → uncertain).
3. `env = {**params_env, **x_star}`.
4. **Verify the equilibrium:** `max_residual = max( |_eval_expr(rhs_expr_i, env)| for i )`. If `max_residual > fixed_point_tol` → **the whole run is uncertain** (the declared point is not an equilibrium for these params — you cannot read stability off a non-equilibrium; this also fails-closed the case where a swept parameter moves the equilibrium and the formula doesn't track it).
5. **Spectrum:** evaluate the numeric Jacobian `J = [[ _eval_expr(J_ij, env) ]]` (n×n finite-real), `eigs = numpy.linalg.eigvals(J)`, `α = max(Re λ)` (the spectral abscissa). Non-finite α → uncertain.
6. **Criterion:** `stable_pt = _eval_criterion(α, sim_criterion)` (the existing op/threshold check — `lt margin` for a stability claim, `gt margin` for an instability claim).
7. Record the audit row (§7).

**Verdict:** `passes = #{points satisfying the criterion}`; `robust = passes/grid_size >= robust_frac` (the `robust_frac ∈ (0,1]` guard, validated before the loop); **PASS** (`confirm`) iff robust, else **FAIL** (`refute`). Any uncertain-causing condition above (off-equilibrium, non-whitelisted diff node, non-finite) → **uncertain**.

`result.computed` (e.g.): `"linear_stability: 5/5 points satisfy criterion (frac >= 0.80); alpha in [-0.42, -0.18]"` — surfaces the pass fraction and the α range across the sweep.

**Robustness stance (state explicitly):** `min_grid_points` forbids a single-point stability claim — a `linear_stability` plan must sweep `≥ min_grid_points`, so a PASS asserts **"α(J(x\*(θ), θ)) satisfies the criterion robustly across the swept θ-range,"** not at one cherry-picked operating point.

---

## 3. Fail-closed discipline (all v1 guards retained, plus)

- **`set(fixed_point) == set(state_vars)`** → else uncertain (every state variable needs an equilibrium coordinate; mirrors the existing "missing rhs for state var" check). A missing or extra coordinate is a malformed plan.
- **Fixed-point free symbols ⊆ params** (no state-var dependence) → else uncertain.
- **Equilibrium residual** `> fixed_point_tol` at any grid point → uncertain (§2.4).
- **Post-diff node cap** on each `J_ij` (§2) → uncertain.
- **Non-whitelisted node** in a derivative (e.g. `d/dx sign(x)` → `DiracDelta`) → `_eval_expr` raises → uncertain.
- **Non-finite / complex** anywhere (fixed-point eval, Jacobian entry, α) → uncertain (the existing `try → _u` wrapper covers the whole compute).
- **Primitive-specific required fields:** `linear_stability` requires `state_vars`, `rhs`, `fixed_point`, `sim_criterion`, `robust_frac`, `param_sweep`, and the size caps (`max_state_vars`, `max_expr_nodes`, `max_grid_points`); it does NOT require `t_span`/`dt`/`init`/`observable`/`max_steps`. The `_SIM_REQUIRED`/`_SIM_CAPS` are selected per primitive.
- F1/F3 intact: `sympy.diff` operates only on already-restricted-parsed Exprs (no code exec); evaluation is the whitelist-node `_eval_expr`; `numpy.linalg.eigvals` does float linear algebra on a numeric matrix, never plan strings. The designer emits a structured plan only.

---

## 4. Gate integration — status mapping UNCHANGED; one basis branch

The verdict flows through `run_simulation_checks` and the gate **unchanged**: confirm → discounted `survived` (no `CheckRecord`, no `independent_sources`, no route to `internally_validated`); refute → `landed` → `challenged` (`fatal` iff `load_bearing AND role=="novel_core"` else `major`); uncertain → no-op. The L2-D9 teeth, `_evaluate`, and `run_simulation_checks` are untouched.

**The one place "gate unchanged" isn't literal — the attack basis.** `verdict_to_sim_attack` currently builds its `basis` from `v.plan.observable`, which is unused for `linear_stability` (it would render `observable = ?(?)`). So `verdict_to_sim_attack` gains a **primitive-aware basis branch** (mirroring how `verdict_to_check`/`verdict_to_attack` are already kind-aware): for `primitive=="linear_stability"`, the basis reports the **fixed point** and the **spectral abscissa α(J)** (via `v.measured`'s α summary) and the criterion/margin, e.g.:
`"linear_stability: <computed>; fixed_point = {x: sqrt(a/b), y: 0}; criterion = lt 0; robust_frac = 0.8"`.
**Only the basis string changes** — the `status`/`severity` mapping (`confirm→survived/minor`, `refute→landed/fatal|major`) and the gate are identical.

---

## 5. Soundness + the loud caveat (pushed to the prompt; not enforced by the runner)

The runner is **general**: it checks `α(J)` against any `op`/`threshold`. Soundness of the *claim* is the designer's responsibility, guided by the prompt:
- **Stability claim:** `op="lt"`, `margin ≤ 0` (a *positive* stability margin — "α < some positive number" — is meaningless). Prefer a **strictly negative** margin: `α < 0` (and bounded away from 0) gives **hyperbolicity** (Hartman–Grobman: the linearization determines local nonlinear stability).
- **Instability-onset claim:** `op="gt"`, `margin ≥ 0` (Hopf / Turing / spinodal — the homogeneous state going linearly unstable; often the more interesting condensed-matter claim). Prefer a strictly positive margin.

> **Loud caveat (ship in the basis / state prominently).** `α = 0` is the **bifurcation point**, and a **defective (non-diagonalizable) Jacobian** can make the eigenvalue spectrum alone inconclusive about nonlinear stability. The linear spectrum is decisive only for a **hyperbolic** fixed point (α bounded away from 0). So a `linear_stability` result is trustworthy when the preregistered margin keeps the verdict strictly off α = 0; a marginal (α ≈ 0) result reflects the linearization's limit, not the idea's truth. And — as in all of Lens 3 — the model, its parameters, and the declared equilibrium formula are LLM-asserted and frozen (verified for `rhs ≈ 0`, but not grounded against literature); a PASS is discounted toy-model evidence, a FAIL is a serious challenge to the mechanism as modeled, neither is independent validation.

---

## 6. Designer

`design_simulation` / `SIMULATION_DESIGNER` gain `fixed_point`:
- `_FIELDS` (the JSON-key whitelist) adds `"fixed_point"`. The numeric-coercion hardening (C4) already stringifies nested values, so a fixed-point formula like `{"x": "sqrt(a/b)"}` (string) and numeric coordinates (`{"y": 0}` → `"0"`) both arrive correctly.
- The prompt teaches the `linear_stability` primitive: model the mechanism's RHS; **derive and preregister the equilibrium as a formula in the parameters** (`fixed_point`, one coordinate per state var, expressions over the params only); give a `sim_criterion` on the spectral abscissa α — **`lt` a non-positive margin for a stability claim, `gt` a non-negative margin for an instability-onset claim**, preferring a strict margin (hyperbolicity); a `param_sweep` for robustness; and the size caps. No `t_span`/`dt`/`init`/`observable` for this primitive.

`run_simulation_checks` is **unchanged** (it designs → runs → maps a decisive verdict to the attack path; the primitive dispatch is entirely inside the executor).

---

## 7. Audit

`_run_linear_stability` emits a per-grid-point `detail` row mirroring the `ode_integrate` table: `{ "params": pov, "fixed_point": x_star, "max_residual": max_residual, "alpha": α, "pass": stable_pt }` — so a pass/fail is auditable (where the equilibrium was, how well `rhs ≈ 0` held, and the spectral abscissa at each swept point). Persisted via `stdout.txt` exactly as the ode_integrate detail table.

---

## 8. Testing (deterministic, real numpy/sympy; only the Designer faked)

- **Compute (real spectrum):**
  - *Stable equilibrium:* `dx/dt = -a*x` (with `a > 0` swept), `fixed_point {"x": "0"}`, criterion `lt 0` → α = −a < 0 for all grid points → `confirm`.
  - *Instability onset:* same RHS with `a < 0` swept (or `dx/dt = a*x`, `a > 0`), criterion `gt 0` → α > 0 → `confirm` (validates an instability claim via `gt`).
  - *Parametric fixed point:* a 2D system whose nonzero equilibrium is `x* = sqrt(a/b)` — `fixed_point {"x": "sqrt(a/b)", "y": "0"}`; verify `rhs ≈ 0` holds across the (a,b) sweep and the spectrum is computed there.
  - *Non-equilibrium declared point → uncertain:* a `fixed_point` that does not satisfy `rhs ≈ 0` (residual > tol) at some grid point.
  - *Moving equilibrium not tracked → uncertain:* a constant `fixed_point` while a swept param moves the true equilibrium → residual exceeds tol → uncertain.
- **Fail-closed (each → uncertain):** `fixed_point` keys ≠ `state_vars` (missing/extra coordinate); a `fixed_point` referencing a state var; a derivative producing a non-whitelisted node; a `J_ij` exceeding `max_expr_nodes`; a `"__"` in `fixed_point`; `robust_frac` outside (0,1]; grid `< min_grid_points` (single-point stability forbidden).
- **Basis branch:** `verdict_to_sim_attack` on a `linear_stability` verdict renders a basis containing `"linear_stability"`, the `fixed_point`, and the α summary — NOT `observable = ?(?)`. The `status`/`severity` mapping is byte-identical to the ode_integrate path (confirm→survived/minor, refute→landed/fatal-or-major).
- **Gate mapping (FakeLLM designer + real executor):** stable robust pass → discounted `survived` (`claim.checks == []`, not `internally_validated`); instability/failed-stability robust → `landed` (fatal on a `load_bearing novel_core` claim) → `challenged`; uncertain → no attack, `"simulation"` not marked.
- **Unchanged pins stay green:** `"simulation"` does not satisfy the `"magnitude"` teeth; `_evaluate` references neither `"simulation"` nor `"primitive"`; `run_simulation_checks` diff-free; the ode_integrate + negative-control tests all green (the new branch is additive).
- **Designer:** emits a `linear_stability` plan with a `fixed_point`; malformed JSON → None; numeric fixed-point coordinates coerced to strings (C4).

---

## 9. Decision log
- **LS-D1** Parametric fixed point, code-verified (not numerical re-solving). `fixed_point: dict[str,str]` values are restricted expressions over the **parameters**; code evaluates them per grid point and verifies `rhs ≈ 0` (`≤ fixed_point_tol`) before reading the spectrum. This covers the moving-equilibrium case without a Newton root-finder's non-determinism / fakeability surface. A coordinate referencing a state var → uncertain.
- **LS-D2** Verdict via the **spectral abscissa** α(J) = maxᵢ Re λᵢ(J) (`numpy.linalg.eigvals` on the `sympy.diff` Jacobian evaluated at the equilibrium through the safe `_eval_expr`). The runner is **general** (any `op`/`threshold` on α): `lt` → stability, `gt` → instability-onset. F1/F3 intact (diff on restricted Exprs; whitelist-node eval; numpy linear algebra, no plan-string exec).
- **LS-D3** **Soundness pushed to the prompt, not the runner:** stability needs `lt` margin ≤ 0, instability needs `gt` margin ≥ 0; prefer a strict margin (hyperbolicity). α = 0 (bifurcation) and a defective Jacobian are marginal cases where the linear spectrum alone is inconclusive — the **loud caveat** (§5).
- **LS-D4** **Gate status mapping UNCHANGED; one basis branch.** Only `verdict_to_sim_attack`'s `basis` gains a `linear_stability` branch (fixed point + α), mirroring the existing kind-aware basis logic. `_evaluate`, `run_simulation_checks`, and the status/severity mapping are untouched. PASS → discounted `survived`; FAIL → `challenged`.
- **LS-D5** Fail-closed extended: `set(fixed_point)==set(state_vars)`; fixed-point free symbols ⊆ params; equilibrium residual ≤ tol per grid point (else whole run uncertain); post-diff `J_ij` node cap; non-finite/non-whitelisted → uncertain. Primitive-specific required-fields/caps (no time-integration fields). `min_grid` forbids single-point stability.
- **LS-D6** Single-arm v1; negative-control-for-stability and the other primitives/observables deferred per L3-D12.

---

## 10. Build slices (order)
1. **Model + executor + basis branch** — add `fixed_point: dict[str,str] = {}` to `ComputationPlan` and `fixed_point_tol` to `SimCfg`; implement `_run_linear_stability` (parametric fixed point parse + free-symbol/`==state_vars` guards; symbolic Jacobian + post-diff node cap; per-grid-point fixed-point eval + `rhs≈0` verify + eigvals + α + criterion + sweep; primitive-specific required-fields/caps; the `detail` audit table); dispatch `primitive=="linear_stability"` in `_run`; add the `linear_stability` **basis branch** to `verdict_to_sim_attack`. Real-numpy/sympy executor + model tests (stable, instability-via-`gt`, parametric fixed point, every fail-closed case, the basis-branch test).
2. **Designer + integration** — `SIMULATION_DESIGNER` prompt + `_FIELDS` gain `fixed_point` and the soundness/margin guidance; `run_simulation_checks`/gate untouched. Integration tests (stable→discounted survived, failed/instability→challenged, uncertain→no-mark, the teeth + gate-purity pins green, designer emits `fixed_point` incl. numeric-coercion).
