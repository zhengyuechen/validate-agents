# validate-agents — Spec 2 Lens 2 Design (Magnitude / Detectability Executor)

- **Date:** 2026-06-23
- **Status:** Approved design, pending implementation plan
- **Builds on:** Spec 2 symbolic lens (`docs/2026-06-23-validate-agents-spec2-design.md`) — same `ComputationPlan`/`ComputationResult`/`ComputationVerdict` shapes, the same subprocess sandbox + restricted parser, the same F1/F3 discipline. And on Spec-1's Red-team **magnitude** category (`AttackSurface.attempted` already names it) and the rev-4 `Prediction` (`effect_size`, `discriminates_from`, `detectable`).
- **Status line:** *Lens 2 upgrades the reasoned magnitude/detectability check into an executed numeric comparison — catching "real but numerically inert" — while keeping the pure gate and the structured-plan-no-code rule.*
- **One-line goal:** Make "is the distinguishing effect actually detectable / does it respect known bounds / does it separate from the best alternative" a **preregistered, sourced, code-computed comparison**, not an LLM judgment — the first executed *empirical* check.

---

## 1. Scope

### In scope
- A **Magnitude-Designer** (LLM) that, per measurable/discriminating `Prediction`, emits a **structured** `ComputationPlan` with `kind="magnitude"` and a `comparison_kind` (no runnable code — F1).
- A **numeric Executor** (code, not an LLM) that parses the sourced quantities with the existing restricted parser and computes the comparison; the verdict is code (F3).
- **Two gate paths chosen by `comparison_kind`** (the core of this design — §3):
  - `sensitivity_ratio`, `discriminating_margin` → an executed **Attack** (failure → `challenged`).
  - `bound_check` → an injected load-bearing **claim** (violation → `refuted`).
- **Build `sensitivity_ratio` first**, then `bound_check`, then `discriminating_margin` — but **all three `comparison_kind`s and both gate paths are named in the data model upfront** so nothing reshapes later.

### Out of scope (named seams)
- **Lens 3 — Toy-Model / Simulation Executor** — the only lens that runs *arbitrary generated code*; needs a container-grade threat model + its own design. Deliberately last.
- Verifying that `sensitivity`/`bound`/`closest_prior_effect` are the *literature-correct* values — they are LLM-asserted, frozen, and **sourced** (loud), but not yet grounded. Same honest residual as the symbolic lens's `expected_source`; a Spec-3 grounding follow-on (§7).

---

## 2. The non-negotiable rules (preregistration + loud sources)

1. The Magnitude-Designer preregisters, before any execution: `comparison_kind`, the required numeric quantities for that kind, **a source for every threshold-like quantity** (`sensitivity_source`, `bound_source`), and `confirm_if`/`refute_if`.
2. The plan is **frozen** before the Executor runs. The Executor runs only the frozen plan.
3. The verdict is a **code comparison** (numeric), never an LLM judgment after execution (F3).
4. **The danger is the model inventing a friendly sensitivity/bound.** So `sensitivity_source` (for `sensitivity_ratio`) and `bound_source` (for `bound_check`) are **mandatory** and surfaced **loud** in the verdict basis. **Fail-closed: a missing required quantity OR a missing source → `uncertain`; never `pass`, never a landed attack, never a refute.**
5. **No arbitrary code (F1).** Quantities are parsed with the symbolic lens's restricted `parse_expr` (declared-symbol `local_dict`, whitelisted `global_dict`, `__builtins__={}`, `"__"`-rejected) then reduced to a float via `evalf()`. numpy is used only for the (float) arithmetic of the comparison — never to execute plan strings.
6. Sandbox: subprocess + rlimits + wall timeout + minimal env (the Task-2 machinery, unchanged). Every artifact saved (plan, computed values, result JSON).

---

## 3. The two gate paths (by `comparison_kind`) — the core decision

A magnitude verdict is one of two *ontologies*, and the ontology decides the gate path:

| `comparison_kind` | Formula (code) | Ontology | Path | Gate consequence |
|---|---|---|---|---|
| **`sensitivity_ratio`** | `abs(predicted_effect − baseline_or_null) / sensitivity ≥ threshold` | adversarial: "is it above the noise?" | **Attack** | clears → magnitude attack `survived`; **below (inert)** → `landed` attack, `fatal` if the prediction is discriminating (load-bearing) else `major` → **`challenged`** |
| **`discriminating_margin`** | `abs(predicted_effect − closest_prior_effect) / uncertainty ≥ threshold` | adversarial: "does it separate from the best alternative?" | **Attack** | clears → `survived`; **below (indistinguishable)** → `landed` "non-discriminating" attack (discriminating ⇒ `fatal`) → **`challenged`** |
| **`bound_check`** | `predicted_effect ≤ bound` (compliance) | constraint the idea **must respect** (empirical twin of a symbolic known-limit) | **Claim** | complies → injected load-bearing claim `pass`; **violates** (`predicted_effect > bound`) → claim `fail` → **`refuted`** (explicit empirical contradiction — "this would already have been seen") |
| *all* | — | — | — | **missing quantity / missing source / can't-compute → `uncertain`: no attack lands, no claim passes; the reasoned Red-team magnitude (and Prover) stand (F2)** |

Why the split (the principle): a measured *exclusion bound* is a constraint the idea must respect, so violating it is a contradiction with established data → it refutes, exactly like violating a symbolic known-limit. But "too small to detect" and "indistinguishable from the alternative" are *untestability*, not falsehood → they are serious objections (`challenged`), not refutations. The verdict's nature, not a flag, picks the path.

**Severity rule (attack path):** `fatal` if the prediction is *discriminating* (`discriminates_from` set — the idea's distinguishing claim is what's dead) else `major`. Both render as `verdict_class == "challenged"`; the severity controls the blocker reason (`severe_objection` vs `open_objection`).

---

## 4. Data model (extend the existing shapes — no new model files)

`valagents/computation.py` — extend `ComputationPlan` (magnitude fields optional; symbolic fields unchanged):

```python
class ComputationPlan(BaseModel):
    kind: Literal["symbolic", "magnitude"] = "symbolic"
    # --- symbolic (unchanged) ---
    expression: str = ""            # (symbolic) — now defaulted so magnitude plans omit it
    variables: list[str] = []
    limit_variable: str = ""
    limit_point: str = ""
    expected: str = ""
    expected_source: str = ""
    # --- magnitude (new; required-by-kind validated in the designer/executor) ---
    comparison_kind: Literal["sensitivity_ratio", "bound_check", "discriminating_margin"] | None = None
    predicted_effect: str = ""
    baseline_or_null: str = ""
    sensitivity: str = ""
    sensitivity_source: str = ""
    bound: str = ""
    bound_source: str = ""
    closest_prior_effect: str = ""
    uncertainty: str = ""
    threshold: str = ""
    target_claim_id: str | None = None     # which claim the magnitude objection is about (else artifact-level)
    discriminating: bool = False           # severity input: fatal if True else major
    # --- criterion / glosses (shared) ---
    criterion: Literal["symbolic_equality", "magnitude"] = "symbolic_equality"
    confirm_if: str = ""
    refute_if: str = ""
```

`ComputationResult` reuses `matched: confirm|refute|neither` — for magnitude, `confirm` = clears-the-bar (detectable / complies / discriminates), `refute` = fails-the-bar (inert / violates / indistinguishable); `computed` carries the numeric ratio/margin/value. `ComputationVerdict` is unchanged.

**Verdict mapping (code, no LLM):**
- `bound_check` → `verdict_to_check(v, …)` (exists) → `CheckRecord(lens="executor")` on the injected bound-claim: `confirm → pass`, `refute → fail`.
- `sensitivity_ratio` / `discriminating_margin` → **new** `verdict_to_attack(v, target_claim_id, discriminating, tick) -> Attack`: `confirm → Attack(type="magnitude", status="survived")`; `refute → Attack(type="magnitude"|"non_discriminating", status="landed", severity="fatal" if discriminating else "major")`. `uncertain` → no attack.
- The attack `basis` and the check `basis` surface the computed value, the threshold, and the **source** (`sensitivity_source`/`bound_source`) — the loud caveat in the output.

---

## 5. Flow (where it hooks)

In `valagents/scheduler.py`, a new `run_magnitude_checks(store, llm, cfg, tick)` runs in `_whole_artifact_lenses` **after** `predict()` and after the reasoned `red_team()` (so it augments, not replaces — F2):

```
for each Prediction p where p.measurable (and, for discriminating_margin, p.discriminates_from set):
  plan = await design_magnitude(p, art, llm, cfg)        # structured plan + comparison_kind + sources (F1)
  if plan is None: continue                               # fall back to the reasoned Red-team magnitude
  verdict = run_plan(plan, cfg, artifacts_dir=…)          # numpy execution, code verdict (F3)
  store.record({"event": "magnitude_executed", ...})      # audit, unconditional
  if verdict.verdict == "uncertain": continue             # FAIL-CLOSED: no attack, no claim, and (critically)
                                                          #   does NOT mark "magnitude" attempted (see below). Reasoned magnitude stands (F2).
  if plan.comparison_kind == "bound_check":
      inject a claim type="mathematical", load_bearing=True, origin="bound_check"
        ("the idea's predicted effect respects the established bound <…>"), then
        store.add_check(verdict_to_check(verdict))   # mirrors inject_limit_checks; type=mathematical
        so the executor pass/fail is a strongest-pass/fail that dominates incidental uncertainties
  else:  # sensitivity_ratio | discriminating_margin (attack path)
      store.set("attacks", art.attacks + [verdict_to_attack(verdict, plan.target_claim_id, plan.discriminating, tick)])
      add "magnitude" to AttackSurface.attempted   # ONLY HERE — a real magnitude attack was EXECUTED (survived/landed)
```

> **Pinned (anti-laundering of the teeth check).** `"magnitude"` is added to `AttackSurface.attempted` **only on a decisive, executed attack-path verdict** (`survived`/`landed`). A `uncertain` / missing-source / can't-compute magnitude run **must not** mark `"magnitude"` attempted — otherwise a *fail-closed non-execution* would satisfy `_thin_attack_surface()` (which only checks `"magnitude" in attempted`), letting an idea clear the teeth check without a real magnitude probe. The reasoned Red-team's self-reported `"magnitude"` attempt is separate and unchanged (the §2-boundary note already discounts it as coverage-not-strength); lens 2's marking is the *executed* upgrade, and it earns the mark only by actually running.

Gate consequences via the **unchanged `_evaluate`**: a landed `fatal`/`major` magnitude attack → `needs_experiment` (`severe_objection`/`open_objection`) → `challenged`; a bound-violation claim `fail` → `refuted`; a detectable/compliant/discriminating result → no negative effect (and the executed magnitude category satisfies the thin-attack-surface check). `_evaluate()` is **not changed** — it already reads `claim.status` and `attacks`.

---

## 6. Sandbox (reuse Task-2; numpy added)

- The existing `valagents/sandbox/runner.py` dispatches on `plan["kind"]`: `symbolic` (existing) vs `magnitude` (new). The magnitude branch parses each required quantity with the **same restricted `parse_expr`** (declared symbols + whitelist + `__builtins__={}` + `"__"`-reject), reduces to a float via `evalf()`, and computes the comparison with numpy; on any parse/compute failure → `ok=False`. Required-quantity-and-source presence is checked *before* compute; absence → `ok=False` (→ `uncertain`).
- `valagents/sandbox/executor.py` `run_plan` is unchanged (kind-agnostic subprocess + limits + artifacts).
- **`numpy` is the only new pinned dep.** It executes float arithmetic, not plan strings — the structured-plan-no-code rule (F1) is absolute, and matters *more* here because numeric execution is broader than symbolic; the restricted parser + suppressed builtins + dunder guard are reused exactly.

---

## 7. The loud caveat (state prominently; ship in the basis)

> The Executor verifies that the **predicted effect**, as transcribed, clears/violates the **preregistered, sourced `sensitivity`/`bound`/`closest_prior_effect`**. It does NOT yet verify that those threshold values are the literature-correct ones — they are LLM-asserted, frozen, and **sourced**, but not grounded. So this lens catches "the model's own numbers say the effect is inert / violates a bound it cited"; it does not catch "the model cited a wrong sensitivity." Grounding the sources is a Spec-3 follow-on. (And — as in the symbolic lens — a transcription error in `predicted_effect` could mis-land an attack or mis-refute a bound; mitigations: the reasoned Red-team is retained, every value+source is saved as an artifact, and a refute/landed result is a Repairer target.)

---

## 8. Testing (deterministic, no network; real numpy)

The Executor runs real, pinned numpy + the restricted parser — so tests exercise actual computation; only the Magnitude-Designer (LLM) is faked.

- **Executor (real compute):**
  - `sensitivity_ratio` detectable → `matched="confirm"` (e.g. predicted 1e-9, baseline 0, sensitivity 1e-12, threshold 3 → ratio 1000 ≥ 3).
  - `sensitivity_ratio` **inert** → `matched="refute"` (predicted 1e-18, sensitivity 1e-12 → ratio 1e-6 < 3).
  - **missing `sensitivity_source` → `ok=False` → `uncertain`** (the anti-laundering test — a thresholds-without-source plan never passes/lands).
  - `bound_check` complies (`predicted ≤ bound`) → `confirm`; **violates** (`predicted > bound`) → `refute`.
  - `discriminating_margin` clears → `confirm`; below → `refute`.
  - dunder/garbled quantity → `uncertain` (restricted parser, never executes).
- **Magnitude-Designer (FakeLLM):** emits the structured tail → `ComputationPlan` (right `comparison_kind`, required fields); missing source field → returns a plan the executor will mark `uncertain`, or `None`.
- **Verdict mapping:** `verdict_to_attack` — confirm→`survived`, refute→`landed` with severity `fatal` iff `discriminating`; `bound_check` refute→`CheckRecord(fail)`.
- **Integration (FakeLLM designer + real executor):** inert sensitivity_ratio on a discriminating prediction → landed `fatal` magnitude attack → artifact `verdict_class == "challenged"`; bound violation → injected claim `fail` → artifact `refuted`; missing-source → `uncertain` → no attack/claim, reasoned magnitude stands.
- **Teeth not laundered (the pinned rule):** a `sensitivity_ratio` run with a missing `sensitivity_source` → `uncertain` → `"magnitude"` is **NOT** added to `attempted`; assert that if this was the only magnitude attempt, `_thin_attack_surface()` still returns `True` (a fail-closed non-execution cannot clear the teeth check). Conversely, a decisive (`survived`/`landed`) run DOES add `"magnitude"`.
- **Gate purity:** `inspect.getsource(IdeaArtifact._evaluate)` contains neither `"magnitude"` nor `"comparison_kind"`.

---

## 9. Build slices (order)

1. **Data model** — extend `ComputationPlan` (`kind="magnitude"` + all three `comparison_kind`s + the sourced fields); `verdict_to_attack`; default the symbolic `expression` field. Unit tests (model + the attack/check mapping).
2. **Magnitude runner + executor branch** — `runner.py` `kind=="magnitude"` → `sensitivity_ratio` only, restricted-parse→float→numpy comparison, required-quantity+source check, fail-closed. Real-numpy executor tests (incl. the missing-source → uncertain test). Pin `numpy`.
3. **Magnitude-Designer + wiring** — `design_magnitude` agent + prompt (emits the structured plan, no verdict); `run_magnitude_checks` in the scheduler for `sensitivity_ratio` (attack path), augment+fallback; mark `AttackSurface.attempted` magnitude. Integration tests (inert→challenged, detectable→survived, missing-source→uncertain).
4. **`bound_check` (claim/refute path)** — runner branch + the inject-bound-claim wiring (twin of `inject_limit_checks`); violates→`refuted`, complies→`pass`, missing-source→uncertain.
5. **`discriminating_margin` (attack path)** — runner branch + wiring; below→landed (fatal) attack; gate-purity test.
6. **(later)** lens 3 — separate design + threat model.

---

## 10. Decision log
- **L2-D1** Two gate paths by `comparison_kind`: `bound_check` → claim (`violation → refuted`, empirical twin of symbolic limit-violation); `sensitivity_ratio`/`discriminating_margin` → attack (`failure → challenged`). The verdict's ontology (must-respect constraint vs adversarial probe) picks the path.
- **L2-D2** Fail-closed on missing quantity OR missing source → `uncertain`; never pass / land / refute. `sensitivity_source`/`bound_source` mandatory and loud (anti threshold-laundering).
- **L2-D3** Code judges (numpy float comparison); the Magnitude-Designer emits only the structured plan (F1/F3) — never sees the result.
- **L2-D4** Severity on the attack path: `fatal` if the prediction is discriminating else `major`; both → `challenged`.
- **L2-D5** Reuse the Task-2 sandbox + restricted parser verbatim; numpy executes float arithmetic only, never plan strings.
- **L2-D6** `_evaluate()` unchanged — magnitude flows through `attacks` (attack path) and `claim.status` (bound path), both already read by the gate.
- **L2-D7 (loud caveat)** Verifies clearance/violation against the *preregistered, sourced* threshold, not that the threshold is literature-correct (Spec-3 grounding). All values + sources saved as artifacts.
- **L2-D8** F2 fallback: an `uncertain` magnitude result adds no attack and no claim; the reasoned Red-team magnitude (and Prover) stand.
- **L2-D9 (anti-laundering of teeth)** `"magnitude"` is added to `AttackSurface.attempted` **only on a decisive executed attack-path verdict** (`survived`/`landed`) — never on a fail-closed `uncertain`/missing-source run. Otherwise a non-execution would satisfy `_thin_attack_surface()`. Tested (§8).
