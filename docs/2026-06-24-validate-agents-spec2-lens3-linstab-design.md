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
- Config `SimCfg.fixed_point_tol` (equilibrium-residual tolerance, default `1e-6` absolute — §2) and `SimCfg.min_points_per_axis` (per-swept-axis density floor, default `5` — §2) — both system-set, not LLM knobs.
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

**Architecture — branch on `primitive` inside `_run_simulation`, after the shared front half.** `_run` keys on `kind`; a `linear_stability` plan has `kind=="simulation"` so it reaches `_run_simulation`, where the shared setup already lives. So the primitive branch is **inside `_run_simulation`**, not in `_run`: at the top, select the per-primitive `_SIM_REQUIRED`/`_SIM_CAPS` (since `linear_stability` omits `t_span`/`dt`/`init`/`max_steps`); run the **shared front half** (presence, complete-ceiling, positive-cap + ceiling checks, the **name-uniqueness guard** below, the param-aware restricted RHS parse with reserved-name/free-symbol guards, and the grid); then dispatch the per-primitive **back half** (ode: time caps + RK4; linstab: the spectral compute below). Factor the shared front so the ode path stays byte-identical.

**Shared guards folded into this slice (both primitives ride on them — findings carried from the spec review):**
- **Intra-plan name-uniqueness.** Names must be unique across `state_vars ∪ params ∪ param_sweep` → else uncertain. A name in both `state_vars` and `params` would otherwise be silently shadowed (state var wins in the integrand; a swept value overrides a fixed param) — and it opens a circularity hole in the fixed-point guard below (a name in both sets sits in `allowed_fp`, so `fixed_point` could reference a state var and pass `⊆ params`). Uniqueness closes both, for `ode_integrate` and `linear_stability` alike.
- **Grid-size cap on the projected product, before materializing.** Check `∏ₖ nₖ` (the product of all sweep counts — cheap, all integers) against `max_grid_points` and the ceiling **before** `_build_grid` materializes the list, so a pathological `n` can't allocate a huge grid (the in-process cap, not just the subprocess rlimits, bounds the work). Shared with `ode_integrate`.
- **`init_sweep` actively excluded for `linear_stability`.** The grid is built from `param_sweep` **only** (`_build_grid(param_sweep, {}, …)`); a **non-empty `init_sweep` → uncertain**. (Otherwise a one-point `param_sweep` × a 4-point `init_sweep` clears `min_grid` while sweeping *no parameter* — satisfying the robustness gate with zero real variation, since `init_sweep` axes never touch α with no integration.)

Then, the linear-stability compute:

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

**Robustness stance — the sweep is a counterexample search, so the floor is PER-AXIS.** A `linear_stability` PASS asserts **"α(J(x\*(θ), θ)) satisfies the criterion robustly across the swept θ-range."** The sweep is a *counterexample search* for an intermediate-θ crossing of the margin (a Hopf/Turing onset where α rises across the threshold **between** samples); grid density sets the finest crossing you can catch, and **no finite grid proves the for-all-θ claim** (a real epistemic limit, like the toy-model caveat). Two consequences, both pinned here:
- **Per-axis minimum, not just a total.** Each swept `param_sweep` axis must have `≥ SimCfg.min_points_per_axis` points (default **5**) → else uncertain. A bare total floor degenerates under multi-parameter sweeps (`min_grid=4` with two swept params is just the 2×2 corners — barely a sweep per axis). The total `min_grid_points` remains a backstop; the per-axis floor is where the robustness claim actually lives. (Compute is no constraint — each grid point is `n²` evals + `O(n³)` eigvals.)
- **A stability claim runs `robust_frac = 1`** (pushed to the prompt, §6). With `lt margin ≤ 0`, the grid becomes a *pure* counterexample search — "stable at 80% of sampled points" is not a stability guarantee. (An instability-onset `gt` claim may legitimately use `robust_frac < 1`.)

**Equilibrium tolerance (v1): `fixed_point_tol = 1e-6`, absolute.** The residual is pure float64 roundoff (no integration/truncation): at a *correct exact* formula it is `~ε·|term|` (e.g. `x* = sqrt(a/b)` in `f = a − b·x²` gives `~2ε·|a| ≈ 4e-16·|a|`), while a *wrong* formula isn't a root at all → residual `O(1)`. `1e-6` sits comfortably in that gap, and it errs *loose* on purpose — the harmful failure here is a **false uncertain** (rejecting a correct equilibrium), not a false accept (a wrong formula is `O(1)`, nowhere near `1e-6`). Do NOT tighten to `1e-8` — it doesn't improve wrong-formula detection and only shrinks the headroom. The headroom is finite: false uncertains begin once RHS-term magnitudes at the equilibrium exceed `~tol/ε ≈ 5e9` (a parameter swept to `~1e12`). v1 sweeps are expected in natural/dimensionless `O(1)`–`O(10³)` ranges, far below that. **v1.x upgrade (only if extreme magnitudes are ever swept):** a relative tolerance `|fᵢ| ≤ atol + rtol·maxₖ|termᵢₖ|` (terms via `fᵢ.as_ordered_terms()`, cheap), which drops the scale assumption.

---

## 3. Fail-closed discipline (all v1 guards retained, plus)

- **Intra-plan name-uniqueness** across `state_vars ∪ params ∪ param_sweep` → else uncertain (§2 — shared with `ode_integrate`; closes the silent-shadowing bug and the fixed-point circularity hole).
- **`init_sweep` non-empty for `linear_stability`** → uncertain (§2 — the grid is `param_sweep`-only; prevents faking robustness with non-parameter axes).
- **Projected grid-size** `∏ₖ nₖ > max_grid_points` (or ceiling) → uncertain, checked **before** materializing the grid (§2 — shared with `ode_integrate`).
- **Per-axis density:** any swept `param_sweep` axis with `< min_points_per_axis` points → uncertain (§2 — the robustness floor lives per-axis, not just on the total).
- **`set(fixed_point) == set(state_vars)`** → else uncertain (every state variable needs an equilibrium coordinate; mirrors the existing "missing rhs for state var" check). A missing or extra coordinate is a malformed plan.
- **`"__"` in any `fixed_point` value** → uncertain (the dunder pre-parse guard, replicated from the RHS path).
- **Fixed-point free symbols ⊆ params** (no state-var dependence) → else uncertain. (With name-uniqueness enforced, `params` and `state_vars` are disjoint, so this guard genuinely excludes state-var dependence.)
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

> **Loud caveat (ship in the basis / state prominently).** The linear spectrum decides local nonlinear stability only for a **hyperbolic** fixed point (α bounded away from 0 — Hartman–Grobman). At `α ≈ 0` the linearization is inconclusive: this is the **bifurcation point**, and it is also exactly where a **defective (non-diagonalizable) Jacobian** matters — for a hyperbolic point the sign of `Re λ` decides regardless of diagonalizability (a `tᵏe^{λt}` transient still decays when `Re λ < 0`), so a **strict** preregistered margin (keeping the verdict off α = 0) already covers the defective case. A marginal (α ≈ 0) result reflects the linearization's limit, not the idea's truth. And — as in all of Lens 3 — the model, its parameters, and the declared equilibrium formula are LLM-asserted and frozen (verified for `rhs ≈ 0`, but not grounded against literature); a PASS is discounted toy-model evidence, a FAIL is a serious challenge to the mechanism as modeled, neither is independent validation.

---

## 6. Designer

`design_simulation` / `SIMULATION_DESIGNER` gain `fixed_point`:
- `_FIELDS` (the JSON-key whitelist) adds `"fixed_point"`. The numeric-coercion hardening (C4) already stringifies nested values, so a fixed-point formula like `{"x": "sqrt(a/b)"}` (string) and numeric coordinates (`{"y": 0}` → `"0"`) both arrive correctly.
- The prompt teaches the `linear_stability` primitive: model the mechanism's RHS; **derive and preregister the equilibrium as a formula in the parameters** (`fixed_point`, one coordinate per state var, expressions over the params only); give a `sim_criterion` on the spectral abscissa α — **`lt` a (strictly) non-positive margin for a stability claim, `gt` a non-negative margin for an instability-onset claim**, preferring a strict margin (hyperbolicity); and a `param_sweep` for robustness with **enough points per axis** (the executor floors it). **A stability claim must set `robust_frac = 1`** (the sweep is a counterexample search — any sampled point above the margin refutes); an instability-onset claim may use `robust_frac < 1`. No `t_span`/`dt`/`init`/`observable` for this primitive.

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
- **Fail-closed (each → uncertain):** `fixed_point` keys ≠ `state_vars` (missing/extra coordinate); a `fixed_point` referencing a state var; a name in both `state_vars` and `params` (name-collision); a non-empty `init_sweep`; a projected grid product `∏ₖ nₖ > max_grid_points`; a swept axis with `< min_points_per_axis` points; a derivative producing a non-whitelisted node; a `J_ij` exceeding `max_expr_nodes`; a `"__"` in `fixed_point`; `robust_frac` outside (0,1]; grid `< min_grid_points` (single-point stability forbidden).
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
- **LS-D8 (tuned knobs)** `fixed_point_tol = 1e-6` **absolute** for v1 — the residual gap between roundoff-on-a-correct-formula (`~ε·|term|`) and a wrong formula (`O(1)`) is wide; err loose (a false-uncertain is the harmful mode); relative-tolerance upgrade noted for extreme magnitudes (§2). Grid floor is **per-axis** (`min_points_per_axis`, default 5) not just total — the sweep is a counterexample search for an intermediate-θ margin crossing, and a total floor degenerates under multi-param sweeps; **a stability (`lt`) claim must use `robust_frac = 1`** (prompt-enforced soundness). No finite grid proves the for-all-θ claim — density sets the finest catchable crossing (loud caveat).
- **LS-D7 (shared fixes folded in — `linear_stability` rides on both)** (a) **intra-plan name-uniqueness** across `state_vars ∪ params ∪ param_sweep` (closes silent shadowing + the fixed-point circularity); (b) the **grid-size cap on the projected product `∏ₖ nₖ` before materializing** (the in-process cap, not just subprocess rlimits, bounds the work). Both apply to `ode_integrate` too — done here because the new primitive reuses the same front half; the `ode_integrate` regression suite must stay green through the refactor. The `_run_symbolic` node-cap gap (finding 5) is a separate, independent follow-up.

---

## 10. Build slices (order)
1. **Shared front refactor + model + executor + basis branch** — (a) factor `_run_simulation`'s shared front half (per-primitive `_SIM_REQUIRED`/`_SIM_CAPS` selection, the new **name-uniqueness** guard, the **projected grid-size cap before materializing**, the param-aware RHS parse) so the `ode_integrate` path stays byte-identical and `linear_stability` reuses it — the primitive branch is **inside `_run_simulation`** after the shared front, NOT in `_run`. (b) add `fixed_point: dict[str,str] = {}` to `ComputationPlan` and `fixed_point_tol` to `SimCfg`; implement `_run_linear_stability` (active `init_sweep` exclusion + `param_sweep`-only grid; parametric fixed-point parse + `"__"`/free-symbol/`==state_vars` guards; symbolic Jacobian + post-diff node cap; per-grid-point fixed-point eval + `rhs≈0` verify + eigvals + α + criterion + sweep; the `detail` audit table); add the `linear_stability` **basis branch** to `verdict_to_sim_attack` (status mapping unchanged). Real-numpy/sympy executor + model tests (stable, instability-via-`gt`, parametric fixed point, every fail-closed case incl. name-collision + non-empty-init_sweep + projected-grid-cap, the basis-branch test, and the `ode_integrate` regression suite staying green through the refactor).
2. **Designer + integration** — `SIMULATION_DESIGNER` prompt + `_FIELDS` gain `fixed_point` and the soundness/margin guidance; `run_simulation_checks`/gate untouched. Integration tests (stable→discounted survived, failed/instability→challenged, uncertain→no-mark, the teeth + gate-purity pins green, designer emits `fixed_point` incl. numeric-coercion).

**Separate follow-up (NOT this slice):** finding 5 — `_run_symbolic` (the Spec-2 symbolic known-limit lens) has no `count_ops`/`max_expr_nodes` cap, so `sympy.simplify` on adversarial input is bounded only by wall/CPU rlimits. Independent of `linear_stability` (which never touches `_run_symbolic`); apply the same node cap there in its own small hardening.
