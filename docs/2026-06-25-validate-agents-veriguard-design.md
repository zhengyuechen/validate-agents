# validate-agents — VeriGuard Design (spec-paired plans + counterexample-driven re-emission)

- **Date:** 2026-06-25
- **Status:** Draft for review (no user Q&A — design calls made by the author, flagged for scrutiny in §11)
- **Builds on:** `valagents/computation.py` (`ComputationPlan`, `ComputationResult`, `ComputationVerdict`, `verdict_to_check`, `verdict_to_sim_attack`), `valagents/sandbox/runner.py` (the in-sandbox simulation evaluator), `valagents/sandbox/executor.py` (`run_plan`), `valagents/agents/simulation_designer.py`, `valagents/scheduler.py` (`run_simulation_checks`).
- **Source:** the VeriGuard card + "build first (three)" in `docs/2026-06-25_papers_for_validate_agents_report.md`. Second of the three cheap pure-code wins (CiteAudit ✓ → **VeriGuard** → NLI), ahead of Popper.
- **One-line goal:** Make the simulation designer commit a *checkable contract* — the plan **plus** machine-checkable post-conditions on the result — so code can reject a criterion "pass" that rests on an untrustworthy trajectory (NaN, wrong sign, divergence), and a **concrete failing assertion** (not "try again") drives **one** bounded plan re-emission.

---

## 1. Problem & the load-bearing finding

Today the sandbox evaluates exactly **one** quantity per plan: `observable(var)` vs one `sim_criterion {op, threshold}`, then `robust = frac >= robust_frac` (`runner.py:608`). Nothing checks that the *result itself* is trustworthy — a trajectory full of `NaN`/`inf`, or one whose observable has the wrong sign or diverges, can still produce `matched: "confirm"` and a grounder-independent `pass` (`computation.py:93`, `indep = 1 if v.verdict == "pass"`). The per-grid-point values exist (`runner.py` `detail` list) but are written to disk and **never inspected by code**.

VeriGuard's finding (from the paper): a deterministic post-condition check that returns a **concrete counterexample** — not a generic "regenerate" — is what drives the failure rate toward zero. The counterexample, fed back for exactly one bounded re-emission, lets the designer fix the *specific* defect.

---

## 2. Scope

**In scope:** simulation plans (`primitive in {ode_integrate, linear_stability}` — the only primitives `runner.py` implements, `_SIM_REQUIRED` at `runner.py:437`). The designer emits `result_postconditions`; the runner evaluates them against the per-grid observable series it already computes; a violation forces the verdict to `uncertain` and yields a counterexample that drives one re-emission.

**Out of scope (decisions):**
- **Magnitude / symbolic plans (VG-D3, deferred).** Their results are scalars/symbols (`computed` string), not a grid series; a `finite` check is the only cheap post-condition and it is low-value (sympy/`np` already raise on non-finite). Add later if needed.
- **The bounded-observable honesty path** (`observable.name=="max_abs"` + `op=="le"`, `runner.py:547`) already has its own dt-refinement trust machinery (`_bounded_observe`); VeriGuard post-conditions apply to the **standard** observe path, not that one.
- **Arbitrary LLM-authored predicates / code.** Post-conditions are a **whitelisted, code-owned predicate vocabulary** (§4); the LLM picks `{kind, var, params}` from it, code evaluates. No LLM code execution (the sandbox runs LLM-authored `rhs` already, but post-conditions are pure code over the result).
- **The gate.** VeriGuard never grants credit; it can only **demote** an untrustworthy pass to `uncertain` (VG-D1). `artifact.py` untouched.

---

## 3. The cardinal-rule frame (why this is safe)

Post-conditions are a **trust gate on the result, demote-only**:
- A `confirm` match whose result violates a declared property (NaN, sign flip, divergence) is **not** a real pass — force `uncertain` (VG-D1). Never the reverse: post-conditions can never turn a non-match into a pass.
- The LLM *chooses* which post-conditions to assert (the say-so seam). This is safe because the choice is **demote-only and code-evaluated**: a weak/empty post-condition set just means weaker trust-checking → falls back to today's behavior (**no post-conditions = no new failure mode, no regression**). A wrong post-condition can only wrongly demote (fail-closed), never wrongly promote.
- Code, not the model, evaluates each predicate against the sandbox's own computed values. Network/LLM proposes the contract; pure code adjudicates it — the same firewall as grounding/CiteAudit.

---

## 4. The post-condition vocabulary (whitelisted, code-evaluated)

`ComputationPlan` gains `result_postconditions: list[dict] = []`. Each entry is a whitelisted predicate over the **per-grid observable series** `obs = [d["observable"] for d in detail]` (the values `runner.py` already computes at `runner.py:~600`). Kinds:

| `kind` | params | holds iff | catches |
|---|---|---|---|
| `finite` | — | every value in `obs` is finite (no NaN/inf) | blown-up / undefined trajectories |
| `sign` | `want ∈ {nonneg, nonpos, positive, negative}` | every value satisfies the sign | a physical magnitude going negative |
| `in_range` | `lo, hi` (floats) | every value in `[lo, hi]` | unphysical excursions |
| `monotone` | `sweep_param` (a `param_sweep` key), `direction ∈ {increasing, decreasing}` | `obs` sorted by `sweep_param` is monotone within `conv_rtol` tolerance | non-monotone where physics demands monotonicity |

Each post-condition: `{name: str, kind: str, ...params}`. Unknown `kind`, missing/ill-typed params, or a `monotone` referencing a non-swept param → that post-condition is **skipped with a recorded warning** (fail-open *on the post-condition itself* so a malformed contract never blocks a valid result — the trust-check is best-effort additive; the criterion still gates). The set is small and fixed; extend the table, not the LLM's freedom.

**Evaluation lives in the sandbox** (`runner.py`, VG-D6): the runner already holds `detail` in memory and is the security boundary. A new `_check_postconditions(detail, postconditions, conv_rtol) -> list[dict]` returns `[{name, kind, holds: bool, counterexample: str}]`, where `counterexample` names the *specific* offending point, e.g. `"sign(nonneg) failed: observable=-3.2e1 at params={'T':'5'}, init={'x':'1'}"`. The runner adds `"postconditions": [...]` to its output JSON.

---

## 5. Wiring

**`ComputationResult`** (`computation.py:60`) gains `postconditions: list[dict] = []`, populated by `executor.run_plan` from `out.get("postconditions", [])` (`executor.py:63`).

**`_verdict`** (`executor.py:24`) — after the existing `matched`→verdict mapping, **demote on any failing post-condition**:
```
if result.ok and any(not pc["holds"] for pc in result.postconditions):
    v = "uncertain"   # a matched result with a violated property is untrustworthy (VG-D1)
```
(Applied only when `result.ok`; a sandbox error is already `uncertain`.) The failing post-conditions' counterexamples are surfaced in `ComputationVerdict.measured` / the `CheckRecord.basis` so the demotion is legible and feeds the re-emission.

**One bounded re-emission** (`scheduler.py` `run_simulation_checks`, VG-D4): when a plan's verdict is demoted by a post-condition failure (distinguish from an ordinary `uncertain`: `result.ok and any failing pc`), call `design_simulation` **once more** with the failing post-condition counterexample appended to the prompt context, re-run, and keep the re-emitted result **iff** its post-conditions all hold; otherwise the claim stays `uncertain` (fail-closed). Cap = **1** (`SimCfg.postcondition_reemit_cap: int = 1`). This is per-plan and independent of the whole-artifact repair loop (`cfg.gate.repair_cap`) — it fixes a *buggy plan*, not a *refuted claim*.

**Designer** (`simulation_designer.py`): add `result_postconditions` to the JSON `_FIELDS` whitelist (`simulation_designer.py:10`) and to the `SIMULATION_DESIGNER` prompt — instruct the model to declare the physical properties any *correct* result must have (sign, finiteness, range, monotonicity), in the whitelisted shape. The re-emission prompt appends: *"Your previous plan produced a result that violated this property: <counterexample>. Emit a corrected plan."*

---

## 6. Off / errors / determinism

- **No post-conditions declared → today's behavior exactly** (the demotion clause is a no-op on an empty list; the re-emission never triggers). This is the regression pin.
- **Malformed post-condition → skipped** (warned), never blocks (§4) — the trust-check is additive.
- **Determinism:** the predicate evaluation is pure code over the sandbox's computed values; the only non-determinism is the LLM re-design (agent layer), bounded to one extra call. The post-condition results are recorded in the `CheckRecord`/result for reproducibility/showability.

---

## 7. Testing
- **`_check_postconditions` (pure unit, in-sandbox):** `finite` catches a NaN/inf in `obs`; `sign` catches a negative in a `nonneg` series with the right counterexample string; `in_range` catches an out-of-range point; `monotone` passes a monotone series and fails a non-monotone one (within `conv_rtol`); unknown kind / missing param → skipped+warned, `holds` absent (not a block); empty list → `[]`.
- **`_verdict` demotion:** a `confirm` result with a failing post-condition → `uncertain` (VG-D1); all-holding post-conditions → verdict unchanged; a sandbox error stays `uncertain`.
- **Re-emission (`run_simulation_checks`, FakeLLM):** first plan violates a post-condition → second `design_simulation` called once with the counterexample → second result holds → claim uses the corrected result; second result still violates → claim `uncertain` (no third try); cap respected.
- **Regression:** a plan with `result_postconditions=[]` produces a byte-identical `CheckRecord` to today; the existing simulation tests stay green.
- **Say-so-seam pin:** a post-condition can only demote — a result that does NOT match the criterion plus a (vacuously) holding post-condition set never becomes `pass`.

---

## 8. Cardinal-rule fit
Pure-code trust-checking of the sandbox's own outputs; demote-only (never promotes); the LLM-chosen contract is code-evaluated from a whitelist and can only force `uncertain`; empty contract = no regression. The one concrete counterexample (not "retry") drives a single bounded re-emission. Network/LLM proposes the contract + the fix; the sandbox adjudicates. Does not touch the gate.

---

## 9. Files
- `valagents/computation.py` — `ComputationPlan.result_postconditions`, `ComputationResult.postconditions`, `_verdict` demotion (note: `_verdict` is in `executor.py`).
- `valagents/sandbox/runner.py` — `_check_postconditions`, emit `postconditions` in output JSON (standard observe path).
- `valagents/sandbox/executor.py` — parse `postconditions` into `ComputationResult`; demote in `_verdict`.
- `valagents/agents/simulation_designer.py` + `valagents/prompts.py` (`SIMULATION_DESIGNER`) — emit `result_postconditions`; re-emission prompt suffix.
- `valagents/scheduler.py` — one bounded re-emission in `run_simulation_checks`.
- `valagents/config.py` — `SimCfg.postcondition_reemit_cap: int = 1`.

---

## 10. Decision log
- **VG-D1 (demote-only)** A failing post-condition forces a matched verdict to `uncertain`, never the reverse. Post-conditions are a trust gate, not a credit source.
- **VG-D2 (whitelisted vocabulary)** `finite | sign | in_range | monotone`, code-evaluated in the sandbox over the per-grid observable series; the LLM picks `{kind, var, params}`, never authors code.
- **VG-D3 (scope = simulation)** `ode_integrate` + `linear_stability` only; magnitude/symbolic deferred (scalar results, low-value finiteness only).
- **VG-D4 (one bounded re-emission)** A post-condition failure (not an ordinary uncertain) drives exactly one re-design with the concrete counterexample; the corrected result is kept only if its post-conditions hold; else `uncertain`. Cap=1, separate from `repair_cap`.
- **VG-D5 (say-so seam is safe)** The LLM chooses the contract, but it is demote-only + code-evaluated + empty=no-op → no false-promote, no regression.
- **VG-D6 (evaluate in-sandbox)** Post-conditions are checked inside `runner.py` where `detail` lives and at the security boundary; results returned as structured data.
- **VG-D7 (malformed post-condition fails open on itself)** An unparseable/ill-referenced post-condition is skipped+warned, never blocks a valid result — the trust-check is additive; the criterion still gates.

---

## 11. Reviewer-scrutiny flags (author's uncertainties — attack these)
1. **VG-D4 distinguishing "post-condition demotion" from "ordinary uncertain":** the re-emission must fire on a *post-condition* failure, not on a sandbox error or a genuine `refute`. Is the condition (`result.ok and any failing pc`) the right trigger, and does re-emitting risk masking a genuine `refute` (no — `refute` has `matched=="refute"`, post-conditions are orthogonal)? Confirm a genuine refutation is never re-emitted away.
2. **`monotone` tolerance & ordering:** sorting the grid by `sweep_param` and checking monotonicity within `conv_rtol` — is `conv_rtol` (0.1) the right tolerance, and how are multi-axis sweeps (param × init) handled (the spec assumes a single `sweep_param`; a 2-D sweep needs a marginalization rule)?
3. **VG-D7 fail-open on malformed post-conditions vs the cardinal rule:** skipping a malformed post-condition is fail-open *on the trust-check* (not on the verdict). Is that the right call, or should a malformed contract from a designer be a demotion (stricter)? Author chose additive/no-regression; reviewer should weigh.
4. **Does demote-to-uncertain interact with `_math_uncertainty_is_nonblocking`?** For a `mathematical` claim, an executor `uncertain` can be non-blocking if a proof pass exists (`artifact.py:139`). A post-condition-demoted executor uncertain on a math claim could thus be swallowed — is that correct (the math claim has another proof) or a hole?
5. **Scope honesty:** is simulation-only enough to be worth the wiring, given how many runs actually use simulation plans vs magnitude? (If most real runs are magnitude, VG buys little until magnitude post-conditions land.)
