# validate-agents — Falsifiability gate: demote the hard entry-block; let the math reach the prover

**Status:** design, approved-in-principle (2026-06-25). Awaiting external review before plan/implementation.
**One line:** a single LLM say-so flag (`formal_claim.falsifiable=False`) hard-blocks the whole pipeline *before decomposition*, killing checkable theory papers at entry — demote it from an entry-gate to a code-witnessed, last-resort verdict (FG-1, surgical), make formalization preserve the derivation it currently abstracts away (FG-2, the faithful-encoding fix), and document concrete seeding as the stopgap (FG-3).

---

## 1. Problem — an a-priori say-so flag hard-blocks before any check runs (confirmed live)

**Confirmed** (run 2026-06-25 15:20, Ran's QM meta-symmetry, arXiv:2511.20907): `status=needs_experiment`, `maturity=0.0`, `blocker={claim_id: None, reason: not_falsifiable}`. Nothing downstream ran. The formalizer compressed a concrete multi-step derivation into an **existential** claim and **deleted the empirical anchor**:

> "**There exists** a meta-symmetry principle M … such that imposing M … uniquely determines an operator 𝒯̂, and a Hawking-type exponential map … yields **the functional form** of ρ_Λ **(with its magnitude fixed empirically rather than derived).**"

`falsifiable=False` is true of *that sentence* and false of the paper. The paper's derivation (does M *uniquely* fix 𝒯̂? does the Hawking map *yield* that ρ_Λ form?) is symbolically checkable; its components ("not reducible to Born reciprocity", "Hawking exponential map") are grounder-feedable. None of it ran, because the gate fires at **entry, on a single lossy abstraction**.

**The gate fires in two places:**
- **`scheduler.py:29-32`** — `if not fc.falsifiable: art.finalized=True; record entry_gate not_falsifiable; return False`. Short-circuits `run_entry_gates` **before** faithfulness (line 35), decompose (line 50), and every lens. This is the operational kill.
- **`artifact.py:236-237`** — `if self.formal_claim and not self.formal_claim.falsifiable: return NEEDS_EXPERIMENT, self._b("not_falsifiable")`. The verdict-side entry-gate; `artifact.py:323` maps `not_falsifiable → ill_posed`.

**The conceptual error:** the gate conflates two different senses of "testable":
- **Empirically falsifiable** — predicts a number that could come out wrong. The abstracted claim ("derives the *form*, magnitude fitted") *is* weak here. The gate's reading isn't insane.
- **Derivationally / literature checkable** — does the math hold? is the uniqueness real? is M novel vs. Born reciprocity? *Always* available for a theory paper, and exactly what the prover/executor/grounder exist to check.

A single LLM-produced `falsifiable` bool, judged on a single pre-decomposition abstraction, hard-blocks the run and **discards the derivational and novelty checks that are perfectly assessable.** This is the encoding-faithfulness gap (the formalizer is the model-mediated encoding step) used as a gate.

---

## 2. Cardinal-rule framing — replace say-so-at-entry with code-witnessed-at-exit

The project's pattern is to strip a model's say-so credit and substitute a code-witness (grounder Tier-2; magnitude G-D6/D10; prover PC-1a/1b). The falsifiability gate is the same anti-pattern *on the blocking side*: **an LLM's opinion about an abstraction decides, a-priori, that nothing is worth checking.**

The fix reframes falsifiability as a **demonstrated, a-posteriori property**: *"falsifiable in practice" = at least one sub-claim received a landed, code-witnessed check* (prover-executed derivation, grounder-witnessed quote, or executor-run computation). The LLM's `falsifiable` flag becomes informational; the terminal `not_falsifiable`/`ill_posed` verdict is reserved for runs that **demonstrably could not check anything** — not for claims an LLM abstracted into an existence statement.

**Why this is safe — the strict gate is the real backstop.** Validation already requires, per root claim, `verdict=="pass" and independent_sources>=1` (`artifact.py:_has_independent_external_check`, used at the STRICT block ~`artifact.py:267`). A genuinely unfalsifiable claim cannot earn an independent code-witnessed check, so it **cannot reach `validated`** whether or not the entry-block exists. The entry-block's *only* distinct effect is to short-circuit **early** — which is precisely what kills checkable derivational claims. Removing it **loses no safety** and gains the ability to assess theory papers. (Same fail-safe logic as PC-1a: the change can only *remove* the premature block, never *add* a validation.)

**Caveat — the PC-D6 interaction, and the guard that re-seals this argument.** This safety claim was airtight when written, but PC-D6 (shipped in the prover-credit cycle) added a definitional exemption to the very function it rests on: `_has_independent_external_check` (`artifact.py:230`) now returns true for a *definitional* root on a non-refuting pass with **no `independent_sources`**. So an **all-definitional** root set would satisfy the strict gate's witness requirement on **zero real checks** — and FG-1, by removing the entry-block, is exactly what lets such an unfalsifiable artifact reach the gate. FG-1 + PC-D6 compose into a hole neither opens alone (the definitional exemption keys on the decomposer's LLM `type` label — say-so re-imported through the type). **Required guard (shipped with the PC-D6 cycle, strict gate, owner-authorized):** `any(c.type != "definitional" and self._has_independent_external_check(c) for c in rs)` — the artifact must rest on ≥1 **non-definitional**, code-witnessed root. This restores §2 verbatim: an unfalsifiable artifact cannot reach `validated` without ≥1 real check, regardless of definitional exemptions. **FG-1 must not ship without this guard in place** (it now is).

---

## 3. FG-1 — demote the hard block to a last-resort, evidence-gated verdict (urgent, surgical)

**3a. `scheduler.py` — stop short-circuiting.** Delete the `if not fc.falsifiable: … return False` block (lines 29-32). Replace with a **non-blocking record** so the flag is still surfaced:

```python
store.set("formal_claim", fc)
store.record({"event": "formalizer_falsifiable", "value": fc.falsifiable})   # surfaced, NOT a gate
# (run continues to faithfulness → decompose → lenses regardless of fc.falsifiable)
```

A `falsifiable=False` claim now flows into faithfulness (which has the existing re-formalize-preserving retry, `scheduler.py:38` — see FG-2) and into decomposition, so the prover, grounder, and executor actually run on the sub-claims.

**3b. `artifact.py:_evaluate` — move `not_falsifiable` from entry-gate to fallback.** Remove the early-return at lines 236-237 from the `===== ENTRY GATES =====` block. Re-introduce `not_falsifiable` **after** the REFUTATION and NEEDS-EXPERIMENT sections, gated on "nothing was checkable":

```python
# ===== LAST-RESORT: genuinely unassessable =====
if (self.formal_claim and not self.formal_claim.falsifiable
        and not any(self._has_any_landed_check(c) for c in rs)):
    return NEEDS_EXPERIMENT, self._b("not_falsifiable")
```

where `_has_any_landed_check(c)` is true iff some check on `c` reached a non-pending, lens-produced verdict (`pass`/`fail`/`uncertain` from grounder/prover/executor). Effect:
- **A `falsifiable=False` claim that got refuted** (a check landed and failed) → `REFUTED` (refutation section fires first). Correct.
- **A `falsifiable=False` claim whose sub-claims got code-witnessed checks** → its real verdict (`needs_experiment`/`validated`), with the flag surfaced as a caveat. Correct — Ran's derivation is assessable.
- **A `falsifiable=False` claim where *nothing* landed** → `not_falsifiable` / `ill_posed`, now meaning *demonstrably unassessable*, not *an LLM said so*. Correct and honest.

**Maturity falls out for free.** `maturity` already reads the verdict set (claim_graph/attacks/predictions/coverage), never `self.status` (`artifact.py:~335`). Letting the pipeline run replaces the flat `0.0` with a real score reflecting what was checked — no maturity code changes needed.

**Surfacing.** Keep the `falsifiable=False` flag visible in the report (`cli.py:85` already prints `_falsifiable: …_`) and add a one-line caveat when a non-`ill_posed` verdict ships on a `falsifiable=False` claim: *"empirical falsifiability weak; verdict rests on derivational/literature checks."*

**Cost:** small. Two deletions + one relocated conditional + one helper (`_has_any_landed_check`). `_b`, the verdict-class map (`artifact.py:323`, `not_falsifiable → ill_posed`), and the strict gate are otherwise untouched. Tests that asserted "falsifiable=False ⇒ immediate not_falsifiable, maturity 0.0" must be updated to the new semantics — identified by **running the suite**, not assumed.

---

## 4. FG-2 — make formalization preserve the derivation it currently abstracts away

**Root cause:** the formalizer is constrained to **one sentence** ("CLAIM: \<one sentence\>", `prompts.py:23`) and told to "sharpen only the statement". For a derivational theory paper that forces collapse into "there exists M such that …", dropping the operator, the map, and the explicit form — the very structure the prover/executor would check.

**FG-2a — formalizer prompt: preserve, don't abstract.** Add to `FORMALIZER` (`prompts.py:12`):
- *"If the seed states a derivation (a chain A ⇒ B ⇒ C, an operator, an explicit functional form), preserve the concrete objects and the chain. Do NOT collapse a derivation into an existence claim ('there exists X such that …') — name the operator/form and the steps."*
- The existing rubric line already counts a **"computationally checkable refutation condition"** as falsifiable (`prompts.py:17`); reinforce that a derivable claim with checkable steps is falsifiable *even if its magnitude is fitted*.

**FG-2b — existential-abstraction detector (code, cheap, surfaced).** A formal_claim that is purely existential is a code-detectable smell: it opens with "there exists / there is a / for some" **and** contains no relational operator or equation (no `=`, `⇒`, `∝`, named operator). When the **seed** carried concrete derivation content (an equation, an operator symbol, "derive(s)", a named map) but the **formal_claim** is existential-and-bare, flag `formalization_lossy` and route it through the **existing faithfulness retry** as a `narrowed` outcome — the re-formalize-preserving path at `scheduler.py:38` already exists; this just gives it a second trigger. (This is why FG-1 and FG-2 compose: once FG-1 stops blocking before faithfulness, faithfulness can catch the abstraction and re-formalize it.)

**FG-2c — boundary (honest).** FG-2 narrows the encoding gap; it does not close it (the formalizer is still model-mediated). A symbolic check downstream credits *the model's encoding of the derivation*, not the claim's truth (the PC-1b ceiling). FG-2 raises the chance the prover sees real algebra; PC-1b is what then witnesses it. The two are paired: **FG-2 supplies the structure, PC-1b executes it.**

---

## 5. FG-3 — concrete re-seed (operational stopgap, no code)

Immediate mitigation with zero code: seed a derivational paper with the **concrete chain and explicit forms**, not a one-line gloss. For Ran's paper, instead of *"QM is missing a meta-symmetry that fixes an operator and derives dark energy,"* state the actual symmetry, the operator 𝒯̂'s defining relation, the Hawking-map step, and the ρ_Λ functional form. Less surface for the formalizer to abstract → a concrete formal_claim → real sub-claims to the prover/grounder.

**Seed-quality checklist (document, e.g. in `docs/` or the CLI help):** a derivational seed should name (1) the premise/symmetry, (2) the derived object(s) and their defining relations, (3) the explicit functional form claimed, (4) what is *derived* vs. *fitted*. FG-3 is a usage practice, not a fix — FG-1 is what makes the pipeline robust to imperfect seeds.

---

## 6. Build order & relation to the rest

1. **FG-1 now** — closes the confirmed Ran's-run block; surgical, fail-safe (strict gate is the backstop), `not_falsifiable` demoted to demonstrated-unassessable. Ship first; it unblocks every downstream channel for theory papers.
2. **FG-3 immediately, in parallel** — re-seed Ran's paper concretely; zero code, gives a real run to read while FG-2/PC-1b are built.
3. **FG-2 next** — formalization faithfulness; reuses the existing faithfulness retry, pairs with **PC-1b** (FG-2 supplies derivation structure, PC-1b executes it). Test on a derivational claim.
4. **PC-1a / PC-1b** (separate spec, `…prover-credit-design.md`) — PC-1a unconditionally; PC-1b is the executor-witness that makes a preserved derivation a `computations/` entry. FG-2 + PC-1b together are the decompose-on-math capability.

FG-1 is the prerequisite that makes a derivational test claim (Ran's QM) reach the prover at all — without it, PC-1b has nothing to execute because the run dies at entry.

---

## 7. Testing

**FG-1 (`test_artifact_gate` / scheduler tests):**
- `falsifiable=False` + a root claim with a **landed grounder contradiction** → `REFUTED` (not `not_falsifiable`); decomposition and lenses ran.
- `falsifiable=False` + all root claims pass with independent checks + faithful + complete coverage → reaches the strict verdict (not blocked at entry).
- `falsifiable=False` + **no** landed check on any root claim → `not_falsifiable` / `ill_posed` (the demonstrated-unassessable fallback).
- `scheduler.run_entry_gates` returns `True` (proceeds) for a `falsifiable=False` formal_claim; records `formalizer_falsifiable=False`.
- maturity is a real score (not `0.0`) once lenses run on a `falsifiable=False` claim.
- regression: a `falsifiable=True` claim is unchanged end-to-end.

**FG-2 (`test_formalizer` / `test_faithfulness`):**
- existential-bare formal_claim + derivation-bearing seed → `formalization_lossy` flag → faithfulness `narrowed` → re-formalize retry fires.
- a re-formalized claim that preserves the operator/form is **not** flagged.
- prompt regression: a genuinely vague seed still yields `falsifiable=no` (we did not make the formalizer over-claim).

---

## 8. Design decisions

- **FG-D1 — falsifiability is demonstrated, not asserted.** The terminal `not_falsifiable` verdict requires the run to have produced **no landed check**, not an LLM flag on an abstraction. Mirrors the say-so→code-witness pattern (grounder/magnitude/prover), applied to the *blocking* side. (FG-1.)
- **FG-D2 — demotion is fail-safe, GIVEN the non-definitional-root guard.** The strict validation gate is the backstop, but only once it requires ≥1 NON-definitional code-witnessed root (`any(c.type != "definitional" and _has_independent_external_check(c) for c in rs)`) — without it, the PC-D6 definitional exemption + FG-1 compose to let an all-definitional unfalsifiable artifact validate on zero real checks (see §2 caveat). With the guard (shipped, PC-D6 cycle), an unfalsifiable artifact still cannot reach `validated`, and removing the entry-block only *un-blocks premature kills*. The guard is a **required precondition of FG-1**, not optional. (FG-1.)
- **FG-D3 — `not_falsifiable` moves from entry-gate to last-resort**, below REFUTATION and NEEDS-EXPERIMENT, so a refuted/checked `falsifiable=False` claim gets its real verdict. (FG-1, `artifact.py:_evaluate`.)
- **FG-D4 — formalizer preserves derivations.** Do not collapse a derivation chain into an existence claim; name the operator/form/steps so the prover/executor sees real structure. (FG-2a.)
- **FG-D5 — existential-abstraction reuses the faithfulness retry**, not a new mechanism — a derivation-bearing seed yielding an existential-bare formal_claim is a `narrowed` faithfulness failure. (FG-2b.)
- **FG-D6 — FG-2 narrows, PC-1b witnesses.** FG-2 supplies derivation structure; the symbolic executor (PC-1b) is what code-witnesses it. A symbolic pass still credits the encoding, not the claim's truth (honest ceiling). (FG-2c.)
- **FG-D7 — FG-3 is operational, not a fix.** Concrete seeding mitigates immediately; FG-1 is what makes the system robust to imperfect seeds.

---

## 9. Open questions

- **`_has_any_landed_check` predicate:** exact definition of "landed" (which lenses, which verdicts count). Candidate: any non-pending `pass`/`fail`/`uncertain` from grounder/prover/executor on a root claim. Resolved in the FG-1 plan against the `CheckRecord`/status model.
- **Maturity penalty for `falsifiable=False`-but-checked claims:** should a weak-empirical-but-derivationally-checked claim carry an explicit maturity discount, or is the surfaced caveat enough? Non-blocking; decide with the `artifact.py` owner.
- **Existential detector precision (FG-2b):** the "existential + no operator/equation" heuristic will have edge cases; it only ever *triggers a re-formalization* (cheap, fail-soft), never blocks — so false positives cost one retry, not a verdict. Tune in the FG-2 plan.
