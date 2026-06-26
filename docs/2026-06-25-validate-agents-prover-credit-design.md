# validate-agents — Prover credit: strip the say-so, restore it on code-witness

**Status:** design, approved-in-principle (2026-06-25). Awaiting external review before plan/implementation.
**One line:** the prover currently grants `independent_sources=1` on the model's own `DERIVATION: complete` (pure say-so) — strip that (PC-1a, urgent, surgical), then restore a *legitimate* credit by routing mathematical derivations through the symbolic executor (PC-1b, the decompose-on-math capability).

---

## 1. Problem — the prover self-credits on say-so (confirmed live)

The cardinal rule: a "validated" verdict must come from an independent, **code-adjudicated** check — never the model's own say-so. The grounder was hardened to this (Tier-2: `independent_sources` only from a code-witnessed verbatim quote, `grounder.py:143`), and the magnitude executor too (G-D6/D10: the LLM `bound_source` auto-credit was stripped, `computation.py:79-84`). **The prover is the lone un-hardened channel:**

```python
# prover.py:50
indep = 1 if verdict == "pass" and claim.type in {"definitional", "mathematical"} else 0
```

`verdict == "pass"` comes straight from the model emitting `DERIVATION: complete | FATAL_GAP: no`. No code runs. The model grades its own derivation and the gate credits it.

**Confirmed live** (run 2026-06-25 17:39, completed; 18:21 same seed): definitional claims **C1 and C2** earned `prover verdict=pass, independent_sources=1, status=pass` — validated purely on `DERIVATION: complete`. It did not flip *that* run's terminal verdict (`refuted`, dominated by C5's empirical red-team failure), but:
- it is a **confirmed latent false-validation vector** — any run whose load-bearing claim is definitional/mathematical can be "validated" on say-so;
- substantively it **rubber-stamped oversimplified single-band Hall statements** that the multi-band physics (and the grounder's own literature) would flag.

This is exactly the pre-Tier-2 grounder, still live in the prover.

---

## 2. Cardinal-rule framing & precedent

Two established strips set the pattern this spec follows:
- **Grounder (Tier-2):** credit only from a code-witnessed verbatim on-property quote; the field name carries the honest caveat. Say-so removed; code-witness substituted.
- **Magnitude executor (G-D6/D10, `computation.py:79-84`):** the LLM `bound_source` say-so auto-credit was stripped; `independent_sources` now requires a grounding-supports witness.
- **Symbolic/limit executor (`computation.py:93`):** earns `indep=1` legitimately — **because the sandbox actually executed the computation and the result matched.** No model say-so.

The prover violates this. The fix has the same two-part shape every hardened channel took: **remove the fake credit (1a), substitute a code-witnessed one (1b).** `artifact.py` and the gate's `verdict=="pass" and independent_sources>=1` logic are untouched throughout — we only change what the prover *feeds* the gate.

---

## 3. PC-1a — strip the prover's say-so credit (urgent, surgical, fail-safe)

**Change (`prover.py:50`):** the prover never self-credits.

```python
indep = 0   # PC-1a: a model 'DERIVATION: complete' is say-so, not a code-witnessed check; never auto-credit.
            # The prover keeps full VERDICT power (refute via CONTRADICTION/COUNTEREXAMPLE, fail via FATAL_GAP,
            # downgrade via gapped) — it can still kill a claim; it just cannot self-VALIDATE one.
```

**Why fail-safe:** the prover's *refutation* power is unchanged — `fail` (refutes), `uncertain` (fatal/gapped) all still flow and still block. We remove only the ability to *credit*. A claim can no longer reach `status=pass` on a prover pass alone; it needs the grounder (literature) or the executor (computation/derivation). This is strictly safe: it can only *remove* validations, never add one.

**Confirmed effect:** C1/C2 (definitional) drop from `status=pass` to `uncertain` (no independent check). That is the correct cardinal-rule outcome — a definition is not "validated" by the model judging it coherent.

**Necessary coupled fast-follow — PC-D6 (definitional exemption), NOT optional.** Trace PC-1a through the gate: after the strip, a **definitional** claim has *no path to credit at all* — the grounder credits empirical/mechanistic claims (literature), the executor (PC-1b) credits mathematical claims (algebra), but a definition has neither a literature witness nor algebra to run. So by `AtomicClaim.status` (artifact.py:155, `verdict=="pass" and independent_sources>=1`) and the load-bearing all-pass evaluation (`root_ancestors` / `_evaluate`), **any artifact with a load-bearing definitional claim becomes permanently un-validatable.** This is not hypothetical: **C1 and C2 in the test run are `type=definitional` and `load_bearing=True`.** For the false claim it was harmless (refuted on C5), but a *true* artifact resting on a definitional premise would read `uncertain` forever — vacuous over-blocking, the same failure mode as a premature ≥2 bar.

So PC-1a (fail-safe, ships now) and PC-D6 must be decided **in the same cycle.** The resolution: **exempt definitional claims as premises** — do not require an independent code-adjudicated check of a *convention*. This is categorically different from letting a definition self-credit on say-so (the thing PC-1a removes): a premise is *accepted as scaffolding*, not *asserted as validated*.

**Implementation constraint (must be authorized explicitly):** the exemption is a **gate-logic change in `artifact.py`** — there is no gate-pure substitute (it lives in `AtomicClaim.status` / `_evaluate`, the load-bearing all-pass path). `artifact.py` is under the standing **never-modify** rule (gate purity; Popper is the only documented exception). PC-D6 therefore requires the gate owner's explicit go before the edit is made; PC-1a and PC-1b do not touch `artifact.py` and proceed independently. The decided resolution is recorded here; the minimal gate edit (definitional claims exempt from the independent-check requirement, while remaining refutable) is presented for authorization, not made silently.

**Cost:** small + surgical (one line + the comment). The behavior change touches only **definitional/mathematical** claims whose validating check was a prover pass (empirical/mechanistic claims already get prover `indep=0`, so the common grounder-validated flows are unaffected). Whichever tests encoded a definitional/mathematical prover pass *as the credit* must be updated — identified by **running the suite**, not assumed.

---

## 4. PC-1b — restore legitimate credit via the symbolic executor (the capability build)

**The idea (decompose-on-math):** a "complete derivation" should be a **code-witnessed `computations/` entry**, not model text — moving the reasoning channel from say-so (Bucket C) to code-witness (Bucket A), exactly as `quote ∈ bytes` replaced the grounder's say-so. The machinery already exists:

- `ComputationPlan` / `ComputationResult` / `ComputationVerdict` (`computation.py`),
- the sandbox executor that runs them,
- `verdict_to_check` (`computation.py:77-99`), which **already** credits a symbolic pass with `indep=1` *because it executed* (line 93), `lens="executor"`.

So PC-1b is: for a **mathematical** claim, the prover (or the `computation_designer`) emits a **symbolic `ComputationPlan`** — the derivation's checkable steps as sympy identities / algebraic equalities / limit checks — the existing sandbox runs it, and `verdict_to_check` credits a code-witnessed pass through the *existing* legitimate path. The prover stops grading its own prose; the sandbox witnesses the algebra.

**Data flow:**
```
mathematical claim
  └─ prover/computation_designer → symbolic ComputationPlan (steps as sympy equalities/limits)
       └─ sandbox executor (existing) → ComputationResult (ran, matched/diverged)
            └─ verdict_to_check (existing, computation.py:93) → CheckRecord(lens="executor", indep = 1 if pass)
       gate sees a CODE-WITNESSED independent check — not a model 'complete'
```

**Scope & boundaries (v1 of 1b):**
- **Symbolic only.** A derivation step that can be expressed as a sympy identity / equality / limit is executable and creditable. The full computation/sandbox/`verdict_to_check` path is reused; this is an extension of the existing symbolic check, not new infrastructure.
- **Definitional claims earn nothing here** — there is no algebra to execute on a definition. That is the honest boundary (§3); definitions are premises, not code-witnessed checks.
- **Steps that resist symbolic encoding** (informal/physical arguments) earn no credit — `uncertain`, not a say-so pass. Honest under-credit beats fake credit.
- **Test target: a genuinely derivational claim** — Ran's QM meta-symmetry (a math claim with checkable algebra), **not** this empirical superconductivity seed (which has nothing to derive). The empirical seed is exactly why PC-1b cannot be validated on the current run.

**Cardinal-rule integrity:** the model still *proposes* the derivation steps (as it proposes computation plans today); **code executes and adjudicates**. A symbolic pass is credited because sympy verified the equality in the sandbox, not because the model said "complete." Same firewall as every other executor check.

**Honest ceiling (what a PC-1b credit does NOT mean).** Like every spec in this sequence (citation-circle's disjoint-author collusion, grounding's abstract-only substrate), PC-1b has an inherent limit it must state, because "code executes and adjudicates" is true but incomplete. A symbolic pass certifies that the model's **encoding** of the derivation is internally valid — **not that the claim is true.** Two gaps live in that distinction, both shared with the existing magnitude/simulation executors:
- **Encoding faithfulness:** the model's NL→sympy translation can be *faithfully wrong* — sympy then verifies the encoding, not the claim. (The same residue magnitude/simulation already carry: code witnesses what was encoded.)
- **Validity ≠ truth:** a flawless derivation from false premises passes every step. The executor witnesses derivational *validity*, not the truth of the premises.

So a PC-1b credit means exactly "**this derivational step is code-witnessed valid**" — one leg of the tripod (math / independent-physics / literature), narrowable by redundant independent encodings and dimensional/limit invariants, **not closed.** The basis text must say this, so a reader never mistakes an executor pass for "the math proves the claim."

---

## 5. Build order & relation to the other work

1. **PC-1a now** — closes the confirmed C1/C2 false-validation vector; one-line, fail-safe, `artifact.py` untouched. Ship first.
2. **PC-1b next** — the capability; its own plan, tested on a derivational claim. Restores the legitimate credit PC-1a removed.
3. **Grounder require-ALL recall** — *separate, still-open question*, demoted to a test (run a *true, abstract-supportable* claim and confirm the grounder credits real supports). The 18:21 run could not show a false-negative (no true supportable sub-claims; the supports the LLM emitted were mislabeled contradictions, correctly uncredited). Not a fix-now item; revisit after the true-claim test.
4. **citation-circle** — last, still blocked: 0 credited supports → 0 inputs; only bites at bar≥2; contradictions force-downgrade regardless of count. Spec ready (`docs/2026-06-25-validate-agents-citation-circle-design.md`), waits.

---

## 6. Testing

**PC-1a (`test_agent_lenses` / prover tests):**
- definitional claim, `DERIVATION: complete | FATAL_GAP: no` → `verdict=pass`, **`independent_sources=0`** (was 1).
- mathematical claim, same → `verdict=pass`, `independent_sources=0`.
- refutation unchanged: `GAPS: CONTRADICTION: …` → `verdict=fail`; `FATAL_GAP: yes` → `uncertain` with `FATAL_DERIVATION_GAP:` basis.
- **gate integration:** a definitional/mathematical claim whose only passing check is a prover pass no longer reaches `status=pass` (needs grounder/executor credit). Update whichever tests encoded a definitional/mathematical prover pass as the validating check — identified by the failing suite, not assumed.

**PC-1b (when built):**
- a mathematical claim with a verifiable symbolic step → symbolic `ComputationPlan` executed → `CheckRecord(lens="executor", verdict="pass", independent_sources=1)`; the `computations/<run>/…` artifact exists.
- a step that diverges → `verdict="fail"/uncertain`, `independent_sources=0`.
- a claim with no symbolic-encodable step → no executor credit (uncertain), never a say-so pass.

---

## 7. Design decisions

- **PC-D1 — the prover never self-credits.** `DERIVATION: complete` is model say-so; credit requires a code-witness. Mirrors the grounder (Tier-2) and magnitude (G-D6/D10) strips. (PC-1a.)
- **PC-D2 — strip is fail-safe.** The prover keeps refute/fail/uncertain power; it loses only self-validation. The change can only remove validations, never add one. `artifact.py` untouched.
- **PC-D3 — legitimate credit is restored via the *existing* symbolic executor path** (`verdict_to_check`, `computation.py:93`), not a new credit mechanism. (PC-1b.)
- **PC-D4 — symbolic-only in 1b; definitional and non-symbolic claims earn no prover/executor credit.** Honest under-credit over fake credit. Definitions are premises, not checks.
- **PC-D5 — test 1b on a derivational claim (Ran's QM), not the empirical seed.** The empirical seed has nothing to derive, which is why it can't validate this capability.
- **PC-D6 — definitional-claim exemption is a NECESSARY coupled fast-follow, decided in this cycle.** Post-PC-1a a load-bearing definitional claim is permanently un-validatable (vacuous over-blocking; C1/C2 are exactly this). Resolution: **exempt definitional claims as premises** (don't require an independent check of a convention) — categorically distinct from say-so self-credit. Implementation is a gate-logic change in `artifact.py` (no gate-pure substitute), so it requires explicit owner authorization to override the never-modify rule; presented for go, not made silently. PC-1a/PC-1b proceed independently of it.

---

## 8. Open questions

- **PC-D6 gate authorization:** the resolution is decided (exempt definitional claims as premises) and necessary in-cycle, but it requires an `artifact.py` gate edit under the never-modify rule. Pending the owner's explicit go on that specific edit. PC-1a ships fail-safe meanwhile (it only ever removes validations); the window before PC-D6 over-blocks *true definitional-load-bearing* artifacts only.
- **1b derivation-step representation:** the concrete schema by which a mathematical claim's derivation becomes a symbolic `ComputationPlan` (which step kinds — identity, equality, limit, dimensional check — are in v1). Resolved in the PC-1b plan, against the existing `ComputationPlan` kinds.
