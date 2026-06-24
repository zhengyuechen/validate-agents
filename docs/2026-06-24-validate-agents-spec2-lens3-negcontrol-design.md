# validate-agents — Spec 2 Lens 3 slice-2 Design (Negative-Control / Discrimination)

- **Date:** 2026-06-24
- **Status:** Approved design, pending implementation plan
- **Builds on:** Lens 3 v1 (`docs/2026-06-24-validate-agents-spec2-lens3-design.md`, merged: structured `SimulationPlan`, `ode_integrate`, the safe expression-tree evaluator + deterministic RK4, the robustness sweep, the conservative attack-path mapping). This slice is the **attribution** upgrade named in L3-D12 — the highest-leverage deferred item, the only one that changes what a PASS *means*.
- **Status line:** *Negative-control turns Lens 3's "the behavior is robust" into "the behavior is robustly ATTRIBUTABLE to the proposed mechanism" — by running a null arm (mechanism off) alongside the mechanism arm and requiring the claimed behavior to appear with the mechanism and vanish without it.*
- **One-line goal:** Make a Lens-3 PASS mean the toy model's behavior is **caused by the proposed mechanism** (present with it, absent without it), robustly across the sweep — and make "the behavior happens anyway" (present without the mechanism), which v1 could pass as merely robust, a **falsification** (challenge).

---

## 1. Scope

### In scope
- A new optional plan field **`null_overrides: dict[str,str]`** (parameter name → off-value) that declares how to "turn off" the proposed mechanism (set its coupling parameter to its null value). Non-empty → **discrimination mode**; empty → v1 single-arm robustness.
- A **two-arm executor** inside the existing `_run_simulation`: per grid point, run the **mechanism arm** and the **null arm** (same RHS / init / sweep point; only the `null_overrides` params differ), and require **discrimination** (criterion met with the mechanism, NOT met without).
- The fail-closed and resource discipline extended to two arms (total-work counts both; a non-finite in *either* arm → uncertain; `null_overrides` keys must be declared params).

### Out of scope (unchanged from v1 / per L3-D12)
- Still **`ode_integrate` only**; the other primitives (`linear_stability`, `iterated_map`, `monte_carlo`) and observables (`bounded`, `growth_rate`, `time_to_threshold`) remain deferred in the L3-D12 order.
- The **gate mapping is NOT changed** — `verdict_to_sim_attack`, `run_simulation_checks`, `IdeaArtifact._evaluate`, the L2-D9 teeth rule are all untouched (see §4). Negative-control changes only the *executor's confirm/refute computation*.
- A free-form alternative-dynamics null (`rhs_null`) — rejected (NC-D1): more gameable and removes the discipline of explicit mechanism parameterization.
- Source grounding (Spec-3) — the model values remain LLM-asserted, frozen, sourced-but-not-grounded (the §7 caveat of the v1 design still flies).

---

## 2. The discrimination model

`null_overrides` is the opt-in switch. Let `discriminate_mode = bool(null_overrides)` and `n_arms = 2 if discriminate_mode else 1`.

Per grid point `(pov, iov)` (a parameter-sweep override + an initial-condition override):
- **Mechanism arm:** `env = {**base_params, **pov}`, integrate, extract the observable `obs_m`, evaluate `crit_m = criterion(obs_m)`.
- **Null arm** (only in discrimination mode): `env = {**base_params, **pov, **null_overrides}`, **same** `init`/`y0`/`n_steps`/`dt`, integrate, extract `obs_n`, evaluate `crit_n = criterion(obs_n)`.
- The point **discriminates** iff `crit_m AND (not crit_n)` — behavior present **with** the mechanism, absent **without** it. (In single-arm mode the point "passes" iff `crit_m`, exactly v1.)

**Verdict:** `passes = #{discriminating points}`; **PASS** (`matched="confirm"`) iff `passes / grid_size >= robust_frac`; else **FAIL** (`matched="refute"`). Any compute failure / non-finite / bad window in **either arm** → the whole run is **uncertain** (the existing `try → _u` wrapper, extended to the null arm).

**What FAIL now captures** (the added power): a point fails to discriminate when EITHER `crit_m` is false (the mechanism doesn't produce the behavior — v1's failure) OR `crit_n` is true (the behavior appears **without** the mechanism — **new**: not attributable). A run that robustly hits the second case — which v1 would have PASSED as "robust" — now **refutes**.

`result.computed` (discrimination mode): `"discriminating: P/G points (frac >= K)"`. The per-grid-point `detail` table records both arms: `{params, init, obs_mech, crit_mech, obs_null, crit_null, discriminate}` (single-arm keeps the v1 shape `{params, init, observable, pass}`).

---

## 3. Fail-closed additions (all v1 guards retained)

In discrimination mode, in addition to every v1 guard (restricted param-aware parse + free-symbol rejection + reserved-name shadow guard, the two-layer caps, finite-real, the complete-ceiling requirement):
- **`null_overrides` keys must be declared parameters.** `set(null_overrides) ⊄ (set(params) ∪ set(param_sweep))` → **uncertain** (`"null_overrides reference undeclared/non-param names"`). This also forbids overriding a state variable (state-var names are not in the param sets) — a null override may only touch a declared coupling parameter. Values are restricted-parsed numbers via the existing `_parse_number` (rejects `"__"`, non-finite).
- **Total-work counts both arms.** `grid_size × n_steps × n_arms > max_total_steps` → uncertain (so discrimination mode is correctly twice as expensive and gated accordingly).
- **Non-finite in EITHER arm → uncertain.** Both `_rk4_integrate` calls and both `_extract_observable` calls are inside the same `try → _u` wrapper; a blow-up / complex / NaN in the null arm fails the whole run closed exactly as in the mechanism arm.
- **Backward compatibility:** an empty `null_overrides` (the model default) runs the v1 single-arm path byte-for-byte — every existing simulation plan, executor, and test is unchanged.

---

## 4. Gate integration — mapping UNCHANGED (the conservative choice)

The verdict still flows through the **existing** `verdict_to_sim_attack` and `run_simulation_checks` with **no changes**:
- **PASS** (robustly discriminating) → `Attack(type="simulation", status="survived", severity="minor")` — **discounted**: no `CheckRecord`, no `independent_sources`, no injected claim, **no route to `internally_validated`**. A discriminating toy-model success is the model's *own* evidence, not an independent check — the cardinal rule holds (NC-D3).
- **FAIL** (robustly non-discriminating — mechanism doesn't produce the behavior, OR the behavior appears without it) → `landed` attack → **`challenged`** (`fatal` iff target claim `load_bearing AND role=="novel_core"`, else `major`); **never refuted**.
- **UNCERTAIN** → no attack (F2 fallback; the reasoned mechanism check stands).

The discrimination is conveyed to the reader through `result.computed` (the `"discriminating: P/G …"` summary surfaced in the attack `basis`) — so `verdict_to_sim_attack` needs no change. The L2-D9 anti-laundering rule (`"simulation"` marked decisive-only; does NOT satisfy the mandatory `"magnitude"` teeth), the no-op rule, and gate purity are all **untouched**.

**Why this is the right conservatism:** negative-control's value is added *falsification* (catching "the behavior doesn't actually need your mechanism") plus a *stronger, attributable* — but still discounted — positive signal. Letting a discriminating pass route to `internally_validated` (the rejected Option B) would let the idea's own simulation validate it, breaking the independence guarantee.

---

## 5. Designer

`design_simulation` and the `SIMULATION_DESIGNER` prompt gain `null_overrides`:
- `_FIELDS` (the known-key whitelist) adds `"null_overrides"` (still F1: structured plan only, no verdict/code; unknown keys ignored; malformed JSON → None).
- The prompt instructs: to test **attribution**, parameterize the mechanism's coupling as a named parameter and give its **off-value** in `null_overrides` (e.g. a coupling `g` with `null_overrides: {"g": "0"}`), so the executor can check the behavior vanishes when the mechanism is off. A plan that omits `null_overrides` falls back to the v1 robustness check.

`run_simulation_checks` is **unchanged** (it already designs → runs → maps a decisive verdict to the attack path; the discrimination happens entirely inside the executor).

---

## 6. Testing (deterministic, real numpy; only the Designer faked)

- **Executor — discrimination (real compute):**
  - *Discriminating pass:* a mechanism whose behavior is present with the coupling on and absent with `null_overrides` off, robustly across the sweep → `matched="confirm"`, `verdict="pass"`, `computed` says `"discriminating"`.
  - *Not necessary → refute:* a behavior that the criterion meets in BOTH arms (present even with the mechanism off) → robustly non-discriminating → `matched="refute"`. (The case v1 would have passed.)
  - *Not produced → refute:* the criterion is not met in the mechanism arm → non-discriminating → `refute` (same as v1's robust fail).
  - *Bad `null_overrides` key (not a declared param, or a state var) → uncertain.*
  - *Non-finite in the null arm → uncertain* (blow-up only when the mechanism is off).
  - *Total-work counts both arms:* a plan within the single-arm budget but over it at ×2 → uncertain in discrimination mode.
- **Backward compatibility:** an empty `null_overrides` plan reproduces the v1 single-arm verdict exactly (re-run an existing v1 executor test through the new code path).
- **Gate mapping (FakeLLM designer + real executor):** discriminating pass → discounted `survived` (claim `checks == []`, not `internally_validated`); "behavior without mechanism" → `landed` (fatal on a `load_bearing novel_core` claim) → `verdict_class == "challenged"`; uncertain → no attack, `"simulation"` not marked.
- **Anti-laundering + purity (unchanged, must stay green):** `"simulation"` does not satisfy the `"magnitude"` teeth; `_evaluate` references neither `"simulation"` nor `"primitive"`; `verdict_to_sim_attack`/`run_simulation_checks` diff-free.
- **Designer:** emits a plan with `null_overrides`; malformed JSON → None.

---

## 7. Decision log
- **NC-D1** Opt-in via **`null_overrides` (parameter-override null)**, not a free-form `rhs_null`: the null is the *same* dynamics with the mechanism's coupling at its off-value — less gameable, and it forces the mechanism to be explicitly parameterized (identifiable). Empty `null_overrides` → v1 single-arm robustness (backward-compatible).
- **NC-D2** Discrimination criterion = per grid point `crit(mechanism) AND NOT crit(null)`; PASS iff `≥ robust_frac` of points discriminate — attribution composed on top of v1 robustness in one verdict.
- **NC-D3** **Gate mapping UNCHANGED (Option A).** confirm → discounted `survived` (no validation route — a toy model is not an independent check); refute → `landed` → `challenged`; uncertain → no-op. The new falsification is "behavior present WITHOUT the mechanism → refute". `verdict_to_sim_attack`, `run_simulation_checks`, `_evaluate`, and the L2-D9 teeth are untouched.
- **NC-D4** Fail-closed extended to two arms: `null_overrides` keys must be declared params (else uncertain); total-work `× n_arms`; a non-finite/complex/bad-window in EITHER arm → uncertain. All v1 guards retained.
- **NC-D5** Still `ode_integrate` only; other primitives/observables deferred per L3-D12. Source grounding remains a Spec-3 follow-on.

---

## 8. Build slices (order)
1. **Model + two-arm executor** — add `null_overrides: dict[str, str] = {}` to `ComputationPlan`; in `_run_simulation`, validate `null_overrides` keys, compute `n_arms`/total-work, and run the two-arm discrimination loop (mechanism + null; `crit_m AND not crit_n`); the `detail` table records both arms; `computed` says `"discriminating: P/G …"`. Real-numpy executor tests (discriminating pass, not-necessary→refute, not-produced→refute, bad-key→uncertain, null-arm non-finite→uncertain, total-work×2, v1 single-arm backward-compat).
2. **Designer + integration** — `SIMULATION_DESIGNER` prompt + `_FIELDS` gain `null_overrides`; `run_simulation_checks`/`verdict_to_sim_attack`/gate untouched. Integration tests (discriminating pass→discounted survived, behavior-without-mechanism→challenged, uncertain→no-mark, the teeth + gate-purity pins still green, designer emits `null_overrides`).
