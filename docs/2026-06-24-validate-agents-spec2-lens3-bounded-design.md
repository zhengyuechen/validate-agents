# validate-agents — Spec 2 Lens 3 Design (`bounded` observable / `max_abs`)

- **Date:** 2026-06-24
- **Status:** Approved design, pending implementation plan
- **Builds on:** Lens 3 v1 (`ode_integrate`) + negative-control + `linear_stability` — the structured `SimulationPlan`, the safe expression-tree evaluator, the deterministic fixed-step RK4, the robustness sweep + structured `sim_criterion`, and the conservative attack-path gate mapping.
- **Status line:** *`bounded` lets a mechanistic claim assert its dynamics stay bounded — and makes a genuine divergence a deliberate FAIL — while refusing to let a fixed-step RK4 numerical blow-up launder into a physical refutation. The honesty check is dt-refinement convergence of the refuting verdict, not mere persistence of a blow-up.*
- **One-line goal:** Add a `max_abs` observable and an inverted non-finite rule scoped to the **bounded claim** (`max_abs le bound`), so a confirmed-dt-independent divergence (or an over-`bound` breach that survives refinement) **refutes** boundedness, while a step-size artifact stays **uncertain**.

---

## 1. Scope

### In scope
- A new observable **`max_abs(var, window_frac)`** = peak `|state[var]|` over the window (finite case computes like the other extractors).
- The **bounded claim** = `max_abs` with criterion **`op="le"`** (`max_abs ≤ bound`). For this and only this combination, the v1 non-finite→uncertain rule is **inverted under an honesty check**: a divergence (or a finite breach `> bound`) **refutes** boundedness — but only when the refuting verdict **survives dt-refinement**.
- The **honesty check (the load-bearing piece — §3):** "a refutation must survive dt-refinement." The discriminator is the **convergence** of the refuting quantity across halved `dt` — the overflow *time* `t*` for a divergence, the `max_abs` value for a finite breach — NOT mere persistence of a blow-up.
- `SimCfg.max_dt_halvings` (refinement depth) and `SimCfg.conv_rtol` (convergence tolerance) — system-set.
- The integrator returns the overflow step index (so `t_of` can be compared across refinements).

### Out of scope (deferred / unchanged)
- `ode_integrate` only (no trajectory in `linear_stability`).
- The inverted rule is scoped to `max_abs` + `op="le"`. `max_abs` with any other `op` falls back to the **v1 rule** (finite → value; blow-up → uncertain) — the mirror "exceeds/instability" claim (`max_abs` + `gt`, where a blow-up would false-*confirm*) is a deferred extension (B-D6).
- Full singularity classification beyond `t*`-convergence (e.g. order-of-blow-up fitting) — not needed; `t*`-convergence is the v1 discriminator.
- Source grounding (Spec-3); the model/parameters remain LLM-asserted, frozen, not grounded.

---

## 2. The `max_abs` observable + the inverted rule (scoped)

- `max_abs(var, window_frac)` joins the observable vocabulary: peak `|state[var]|` over the last `window_frac` of the trajectory (a finite real float; `window_frac` validity as the other observables).
- **Selection of the bounded path:** the executor uses the honesty path (§3) iff `observable.name == "max_abs"` AND `sim_criterion.op == "le"`. Otherwise `max_abs` behaves as an ordinary observable (finite → value via `_extract_observable`; a non-finite trajectory → uncertain, the v1 rule unchanged).
- **Guarded direction = false-refute.** A `BOUNDED` verdict at the base `dt` (finite, `max_abs ≤ bound`) is **accepted** (no refinement). *(Residual, stated in the caveat §5: a coarse `dt` could miss a transient spike → a false-PASS; mitigated by the mandated `window_frac=1` and an adequate base `dt`, and a PASS is discounted regardless. The harmful direction for a validation system is the false-REFUTE, which §3 guards.)*

---

## 3. The honesty check — a refutation must survive dt-refinement (the crux)

Per grid point, for a bounded claim (`max_abs`, `op="le"`, threshold `bound`):

1. **Integrate at `dt`** with the overflow-capturing integrator → `(traj_or_partial, overflow_step)` (`overflow_step` = index of the first non-finite state, or `None` if the trajectory completes finite). **Classify** the point:
   - **BOUNDED:** finite AND `max_abs ≤ bound`.
   - **REFUTING — divergence:** `overflow_step is not None` (a blow-up). Record `t_of = overflow_step × dt`.
   - **REFUTING — finite breach:** finite AND `max_abs > bound`. Record the `max_abs` value.
2. **BOUNDED at `dt` → accept** (the point is bounded; no refinement).
3. **REFUTING → refine** (`dt/2, dt/4, …` up to `max_dt_halvings`). Each refinement doubles `n_steps`; if a needed refinement would have `n_steps × 2^k > max_steps` → **uncertain** (cannot confirm at the available resolution). At each refinement, re-classify, and decide:
   - **Any refinement is BOUNDED → uncertain** (the refutation vanished under refinement — a step-size artifact).
   - **Divergence branch (overflow at the base `dt`):** the point must overflow at every refinement AND `t_of` must **converge — monotone and shrinking** (§3a) — to a `t*` that is **strictly inside the window** (`t* < t_span_end` AND `t*` not within `conv_rtol` of `t_span_end` — §3b) → **REFUTE** (a confirmed finite-time singularity). If `t_of` **recedes** (increases as `dt` shrinks — the hallmark of a stiff numerical instability whose overflow step pushes later), or the point becomes a finite breach (the divergence morphed to finite — receding), or it never converges within `max_dt_halvings` → **uncertain**.
   - **Finite-breach branch (finite `> bound` at the base `dt`):** the point must stay a finite breach at every refinement AND `max_abs` must **converge — monotone and shrinking** (§3a) — to a stable value **`> bound`** → **REFUTE**. If `max_abs` **trends down** toward `bound` (the spike tames under refinement) or never converges → **uncertain**.

> **§3a — Convergence is monotone AND shrinking, not a single lucky pair.** "Converged" requires BOTH: (i) the refining quantity `q` (`t_of` for divergence, `max_abs` for breach) is **monotone** across the refinement sequence (`t_of` non-increasing; `max_abs` not reversing direction), AND (ii) the successive absolute deltas `|q(h/2) − q(h)|` are **shrinking** across the sequence and the latest *relative* delta `|q(h/2) − q(h)| / |q(h)|< conv_rtol`. A pure two-value rtol gate is trippable by one coincidentally-close pair on an otherwise-receding sequence; the monotone-and-shrinking requirement closes that single-lucky-sample hole (the `bounded` analog of the linstab grid's "monotone + shrinking" guard). **At least two refinements (three `q` samples) are required** before convergence may be declared; if `max_dt_halvings < 2` the feature cannot refute and every refuting point is uncertain.
>
> **§3b — `t*` near `t_span_end` is its own uncertain zone.** A finite-time singularity at `t* ≈ t_span_end` is indistinguishable from a bounded trajectory that simply has not blown up *within the integration window* — "diverges at `t*=0.98·t_span_end`" cannot be told from "bounded, would-blow-up at `1.5·t_span_end` if integrated longer." So a `t*` converging to within `conv_rtol` of `t_span_end` → **uncertain** (extend the window and re-decide), never refute. Otherwise the verdict depends on the arbitrary choice of `t_span_end` — exactly the free-parameter dependence the honesty check exists to remove.

> **Why convergence, not persistence (the soundness argument).** Both a true finite-time singularity and an under-resolved stiff-but-bounded system keep blowing up as `dt` shrinks, so "blew up at every refinement up to K" would **false-refute** any stiff-bounded system whose RK4 stability threshold sits below `dt/2^K` — the *harmful* direction, and exactly the fast-slow regime where bounded claims matter. The discriminator is the overflow **time**: `ẋ = x²` (singularity at `t*=1`) has `t_of(dt) → 1⁻` (converges); `ẋ = -λx` with `dt·λ` above the RK4 stability boundary (~2.78) is bounded but numerically unstable, and its `t_of` **recedes** to ∞ then vanishes once `dt·λ < 2.78`. Reading the receding trend off 2–3 refinements is cheaper and more sample-efficient than halving all the way to the stable `dt`, and it restores the err-toward-uncertain symmetry with `fixed_point_tol`.

The result is fixed-step and deterministic (no adaptive stepping — L3-D11 holds); the only new capability is the integrator **returning** the overflow step index.

> **§3c — Determinism/ordering invariant (the refinement branch must not break bit-reproducibility).** Refinement is new, data-dependent control flow (it fires only on a refuting point), so the Spec-2 bit-reproducible-verdict invariant is restated for it explicitly: (i) the refinement sequence is **always** `dt → dt/2 → dt/4 → …`, the same order every run; (ii) a grid point's verdict depends only on that point's `(params, init)` and is **independent of sweep traversal order** — no point's refinement reads state mutated by another point; (iii) the cumulative-refinement-step budget (§4) is checked against a **deterministic running total of integration steps**, never wall-clock or any timing signal. Same plan → identical verdict including the full refinement path.

---

## 4. Work bounds + config

- **Per-refinement cap:** a refinement at `dt/2^k` runs `n_steps × 2^k` steps; if that exceeds `max_steps` → uncertain (can't confirm).
- **Base-plan headroom (prompt-budgeted):** the base `n_steps` must satisfy `n_steps × 2^(max_dt_halvings) ≤ max_steps` — else even one refinement exceeds `max_steps` and the feature is inert (instant uncertain on any refuting point). The designer prompt instructs budgeting this headroom.
- **Cumulative refinement budget:** refinement only fires on refuting points, but a genuinely unbounded sweep refutes at many points. The executor tracks **cumulative refinement steps across the sweep** and caps them at `max_total_steps`; exceeding → uncertain. (The nominal `grid × n_steps × n_arms ≤ max_total_steps` cap is unchanged and separate.)
- **Config (`SimCfg`, system-set, injected via `_sim_ceilings`):** `max_dt_halvings: int = 3`; `conv_rtol: float = 0.1`. Added to `_REQUIRED_CEILINGS`.

> **`conv_rtol = 0.1` — justification (it sits on the load-bearing test, like `fixed_point_tol`).** The two failure directions are asymmetric. **Too tight:** a genuine singularity whose `t_of(dt)` converges *slowly* (algebraically, not geometrically) reads as non-converging → uncertain — you miss real refutations, the **safe** direction (consistent with err-toward-uncertain). **Too loose:** a slowly-*receding* `t_of` looks "converged" → false refute — the **harmful** direction. At `0.1` we declare convergence when successive overflow times agree to 10%: for `ẋ=x²` near `t*=1` that is reached in 2–3 halvings, but a marginal stiff system whose `t_of` drifts ~10% per halving could slip through a *pure* rtol gate. The chosen value therefore biases slightly toward "converged," and the **mitigation is structural, not a tighter number**: the §3a monotone-and-shrinking requirement — a recede produces non-monotone or non-shrinking deltas that a single-pair rtol gate can miss — plus the §3b `t*`-near-`t_span_end` uncertain zone. `conv_rtol` is a system knob, never LLM-set.

---

## 5. Gate integration + the loud caveat

- **Gate mapping reused, unchanged.** A bounded **REFUTE** (confirmed divergence or confirmed breach) → the point fails the boundedness criterion → across the sweep, `< robust_frac` pass → executor verdict `refute` → `landed` → **`challenged`**; a robust bounded **PASS** → discounted `survived`; **uncertain** → no-op. `verdict_to_sim_attack`, `_evaluate`, `run_simulation_checks` untouched; the basis already renders the observable (`max_abs`) + criterion (`le bound`).
- **`computed`** summarizes: e.g. `"bounded: 5/5 points within bound (frac >= 1.00)"` or `"unbounded: 2/5 points diverge/breach (confirmed dt-independent)"`.
- **Detail row** distinguishes the two non-PASS outcomes by **separate sentinels** (never `+∞` — keeps the JSON finite/valid like the other rows), so an inconclusive sweep is auditable — is the budget too tight, or is the physics genuinely divergent?
  - `"diverged"` → a **confirmed** dt-independent divergence (→ refute).
  - `"refine_budget_exhausted"` → a refuting point that ran out of `max_dt_halvings` / step budget **without deciding** (→ uncertain).
  - A finite-breach refute records the converged `max_abs` value; a vanished/receded artifact records its (finite) base-`dt` `max_abs` with `verdict="uncertain"`.
  - Row shape: `{params, init, max_abs | "diverged" | "refine_budget_exhausted", t_star?, verdict, refinements_used}`.

> **Loud caveat.** A `bounded` **FAIL** means the toy model genuinely diverges (or breaches `bound`) where the idea claims boundedness, **confirmed dt-independent** across refinements — a serious challenge, not a step-size artifact. A `bounded` **PASS** means the dynamics stayed within `bound` across the swept range **at the chosen resolution** (a coarse `dt` could miss a spike — the residual false-pass; `window_frac=1` and an adequate base `dt` mitigate it, and a pass is discounted regardless). As in all of Lens 3, the model/parameters are LLM-asserted, frozen, not grounded. v1.x rigor option: order-of-blow-up fitting on top of the `t*`-convergence test.

---

## 6. Discrimination mode (per-arm)

When `null_overrides` is set (negative-control), the honesty check + uncertain-propagation run **per-arm**. The null arm is *meant* to diverge, but its blow-up still needs dt-confirmation. An **unconfirmed blow-up in EITHER arm → whole-run uncertain**. A point discriminates iff the mechanism arm is `BOUNDED` and the null arm is a **confirmed** `unbounded` (the mechanism is what keeps it bounded).

---

## 7. Designer

`design_simulation` / `SIMULATION_DESIGNER` gain the bounded claim guidance (no new field — `max_abs` is an observable name, `bound` is the `sim_criterion` threshold):
- For a **boundedness** claim: observable `max_abs` on the relevant state var with **`window_frac = 1`** (bounded for ALL `t`; `< 1` is the weaker *eventually-bounded* claim, excluding transients — state this explicitly, do not let it be silent), `sim_criterion {"op": "le", "threshold": ["<bound>"]}`, and **`robust_frac = 1`** (bounded must hold across the WHOLE sweep — any confirmed-unbounded point refutes). Budget `n_steps` with headroom for refinement (`n_steps × 2^max_dt_halvings ≤ max_steps`).

`run_simulation_checks` and the gate are unchanged.

---

## 8. Testing (deterministic, real numpy)

- **Observable:** `max_abs` on a known trajectory = peak `|var|` over the window (finite case).
- **Bounded pass:** `ẋ = -a*x` (decays), `max_abs le 2`, `window_frac=1`, swept `a>0` → never exceeds → `confirm`.
- **Confirmed divergence → refute:** `ẋ = x²` (finite-time singularity at `t*=1/x0`), `t_span` past `t*`, `max_abs le 10` → overflow with `t_of` converging across refinements → `refute`.
- **Stiff artifact → uncertain (THE soundness test):** `ẋ = -λ*x` with a base `dt·λ` above the RK4 stability boundary (numerically unstable but truly bounded), so it overflows at `dt` (and maybe `dt/2`) but `t_of` **recedes** / a finer refinement is BOUNDED → `uncertain`, NOT refute.
- **Finite-breach survives → refute:** a system whose true peak `> bound`, finite, `max_abs` converging `> bound` across refinements → `refute`.
- **Finite-breach artifact → uncertain:** a coarse-`dt` spike `> bound` that `dt/2` tames under `bound` → `uncertain`.
- **Monotone-shrinking guard (§3a):** a synthetic `t_of` sequence that is overall receding but has one coincidentally-close consecutive pair → **uncertain** (a pure two-value rtol gate would false-refute; the monotone+shrinking predicate must not).
- **`t*`-near-`t_end` (§3b):** a singularity whose converged `t*` lands within `conv_rtol` of `t_span_end` → **uncertain**, not refute; the same system with a longer `t_span` (so `t*` is well inside) → `refute`.
- **Budget-exhausted sentinel (§5):** a refuting point that converges-undecided within `max_dt_halvings` writes `"refine_budget_exhausted"` with `verdict="uncertain"`, distinct from a confirmed `"diverged"`.
- **Determinism of the refinement branch (§3c):** the same plan run twice → identical verdict and identical detail rows; a permuted grid order → identical per-point verdicts.
- **Work bounds:** a base `n_steps` with no refinement headroom (`n_steps × 2^K > max_steps`) on a refuting point → uncertain; a sweep whose cumulative refinement exceeds `max_total_steps` → uncertain.
- **Scoping:** `max_abs` + `op != le` falls back to the v1 rule (blow-up → uncertain, no inversion).
- **Discrimination:** bounded-with-mechanism + confirmed-unbounded-without → discriminating pass; an unconfirmed blow-up in either arm → uncertain.
- **Detail/JSON:** a confirmed divergence writes the `"diverged"` sentinel, not `Infinity`; the result JSON round-trips.
- **Gate (FakeLLM designer + real executor):** confirmed-unbounded on a `load_bearing novel_core` claim → `landed fatal` → `challenged`; robust bounded → discounted `survived` (`checks == []`); uncertain → no-mark. Teeth/gate-purity pins green.
- **Determinism:** same plan → identical verdict (incl. the refinement path).

---

## 9. Decision log
- **B-D1** New observable `max_abs` (peak `|var|`); the inverted non-finite rule is scoped to `max_abs` + `op="le"` (the bounded claim). Any other `op`/observable keeps the v1 blow-up→uncertain rule.
- **B-D2 (the crux)** The honesty check is **dt-refinement convergence of the refuting verdict**, not persistence of a blow-up. Persistence false-refutes stiff-but-bounded systems (both kinds keep blowing up as `dt` shrinks); the discriminator is the overflow **time** `t*` (converges → genuine singularity; recedes → numerical artifact) and, for a finite breach, the `max_abs` value (converges `> bound` → genuine; trends to `bound` → artifact). Err toward **uncertain** — symmetry with `fixed_point_tol` restored.
- **B-D3** A refutation is triggered by ANY refuting verdict (overflow OR finite `max_abs > bound`) and must survive refinement; a `BOUNDED` verdict at the base `dt` is accepted (guarded direction is the false-refute).
- **B-D4** Bounded work: per-refinement ≤ `max_steps`; base `n_steps` budgeted for `2^max_dt_halvings` headroom; cumulative refinement steps capped at `max_total_steps`; the integrator **returns** the overflow step index. Fixed-step, deterministic.
- **B-D5** Gate mapping reused (refute→challenged, pass→discounted survived, uncertain→no-op); only `computed`/detail change (the `"diverged"` sentinel). Discrimination honesty + uncertain-propagation are **per-arm**; unconfirmed blow-up in either arm → whole-run uncertain. `window_frac=1` + `robust_frac=1` are prompt-mandated for the standard claim.
- **B-D6 (the `gt` mirror is NOT a sign-flip — do not "restore symmetry" later)** Deferred: the `gt` "exceeds/instability" mirror, and order-of-blow-up fitting (a v1.x rigor add on top of `t*`-convergence). **Why `gt` is a separate, harder design and not the symmetric mirror of `le`:** under `le` the artifact is a *spurious blow-up* and the honesty check demands a *refutation* survive refinement — it filters a divergence that isn't there. Under `gt` (confirm-on-blow-up) the artifact is the **opposite**: a true-bounded stiff system that spuriously blows up would spuriously **confirm** the instability claim — a numerical artifact manufacturing a **PASS**, the exact laundering Spec 2 exists to stop. The convergence test does **not** symmetrically rescue `gt`, for two reasons: (i) the **guarded direction flips** — you must err toward *refute* (don't confirm instability unless the blow-up is dt-independent), the mirror image of `le`'s err-toward-uncertain; (ii) "instability onset" is frequently **not a finite-time `t*` singularity at all** — a Hopf/Turing instability is a *bounded* growing oscillation with no overflow, so the `t*`-convergence discriminator does not even apply to the most common `gt` claim. A future `gt` slice needs its own discriminator (e.g. a growth-rate fit), not a reused `bounded` path. **Do not symmetrize `_bounded_observe` to cover `gt`; that reintroduces a false-confirm laundering path.**
- **B-D7 (convergence-boundary hardening — same bug class as every prior slice)** The honesty check is pinned at its boundaries so it cannot be under-specified into a false verdict: **(a)** convergence is **monotone AND shrinking**, never a single rtol-close pair, and needs ≥2 refinements (§3a); **(b)** a `t*` within `conv_rtol` of `t_span_end` is **uncertain**, not refute — the verdict must not depend on the arbitrary window length (§3b); **(c)** the refinement branch preserves bit-reproducibility — fixed sequence, traversal-order-independent, step-count (not wall-clock) budget (§3c); **(d)** `conv_rtol`'s bias (slightly toward "converged") is mitigated structurally by (a)+(b), not by a tighter number (§4); **(e)** budget-exhausted-without-deciding gets its own `"refine_budget_exhausted"` sentinel, kept distinct from confirmed `"diverged"` (§5).

---

## 10. Build slices (order)
1. **Model/config + overflow-capturing integrator + `max_abs` + `_bounded_observe`** — `SimCfg.max_dt_halvings`/`conv_rtol` (+ `_REQUIRED_CEILINGS`); the integrator variant returning `(traj, overflow_step)` (the raising form unchanged for ode/linstab); `max_abs` in the observable vocabulary; `_bounded_observe` (the §3 honesty check: classify → refine → `t*`/`max_abs` convergence → bounded/unbounded/uncertain) with the per-refinement + cumulative work bounds; wire the bounded path into the `ode_integrate` sweep loop (scoped to `max_abs`+`le`); the `"diverged"` sentinel in the detail row. Real-numpy executor tests (pass, confirmed-divergence, **stiff-artifact→uncertain**, finite-breach survive/artifact, work bounds, scoping, determinism).
2. **Designer prompt + integration** — `SIMULATION_DESIGNER` teaches the bounded claim (`max_abs` + `le bound` + `window_frac=1` + `robust_frac=1` + refinement headroom); integration tests (bounded→discounted survived, confirmed-unbounded→challenged, uncertain→no-mark, per-arm discrimination, teeth/gate-purity pins green).
