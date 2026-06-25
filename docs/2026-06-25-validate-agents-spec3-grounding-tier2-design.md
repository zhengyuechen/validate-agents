# validate-agents — Spec 3 Grounding Tier-2 Design (grounder `[A1]` support adjudication)

- **Date:** 2026-06-25
- **Status:** Approved design, pending implementation plan
- **Builds on:** Tier-1 grounding (`valagents/grounding.py` — `_norm`, `_content_tokens`, `_quote_valid`, `SCALE_TABLE`, the quote-machinery + the four-outcome discipline), the grounder lens (`valagents/agents/grounder.py` `ground_claim`), `map_support_to_verdict` (`agents/base.py`), and the gate (`artifact.py`, untouched).
- **Status line (honest):** *Today the grounder passes a claim on the model's word that a topically-retrieved paper supports it (`map_support_to_verdict("supported", ≥1) → pass`, where `independent_sources` is capped only on retrieval-existence). Tier-2 makes that credit **code-witnessed for what code can actually witness — a real, verbatim, sentence-bounded, on-PROPERTY passage from a real retrieved source — and no more.** It does NOT make "supports" code-adjudicated: the supports-vs-contradicts direction stays the model's loud label. The honest gain is anti-fabrication + a non-vacuous topicality floor + a load-bearing contradiction guard; entailment and independence remain un-witnessed and are surfaced loud.*
- **One-line goal:** Replace the grounder's retrieval-existence credit with a code-checked one (quote ∈ the retrieved abstract, sentence-bounded, overlapping the claim's distinctive asserted-property tokens), force a verdict downgrade when a real contradiction is cited, and dedup the count — closing the say-so co-signer that makes the deferred ≥2 bar unsound, while claiming only what it witnesses.

---

## 1. Scope

### In scope
- The **grounder `[A1]` support-relation** in `ground_claim` (the primary external-grounding lens for `definitional`/`mathematical`/`empirical`/`mechanistic` claims).
- Per cited `[A1]`: a **quote gate** (anti-fabrication + sentence-bound + substantial + on-property topicality) and a **direction label** (supports/contradicts) that is the model's, surfaced loud.
- A **contradiction guard** (a passing `contradicts` citation forces the claim's grounder verdict to `uncertain`).
- **Dedup** the credited count by normalized source identity; a code-side cap.
- An **honest rename** of what the credited count means (it is not "independence").

### Out of scope (deferred / never)
- **Symbolic `expected_source`** grounding (a known-result match, Tier-1-shaped) — a separate slice.
- **Raising the bar to `≥2`** — Tier-2 makes it *available* (the grounder co-signer is no longer pure say-so) but does not raise it; that is a gate change in `artifact.py` with cross-lens semantics, its own slice (T2-D8).
- **Entailment adjudication** — whether a passage logically *backs* the claim is semantic and not code-decidable here. It stays **loud** (the verbatim quote in `basis`), mitigated structurally by the contradiction guard + the deferred ≥2-corroboration. We do **not** add an NLI model (that relocates say-so to another model — rejected).
- **Independence adjudication** — the `Article` schema has no authors; code cannot witness it. `relation="independent"` is LLM-asserted/hardcoded; we stop claiming code witnesses it.

---

## 2. The honest boundary (the load-bearing correction — read first)

A live review of the prior framing established, against the code, that "quote-backed support" was an over-claim. The honest decomposition of what each piece witnesses:

- **Code witnesses (new in Tier-2):**
  1. **Anti-fabrication** — the quote is a literal, **sentence-bounded** substring of the cited article's retrieved abstract (`_norm`-normalized, both directions). The model cannot cite a passage that isn't there, and cannot invert meaning by spanning a sentence boundary.
  2. **On-property topicality (non-vacuous)** — the quote overlaps the claim's **distinctive asserted-property** tokens, not merely the entity/topic tokens that retrieval already saturated. This is what makes "topicality" mean something (retrieval queries on `claim.statement`, so plain claim-overlap is vacuous — §5).
  3. **Real retrieved, deduped source** — the cited `[A1]` maps to an actually-retrieved article with a URL, counted once per distinct work.
- **Code does NOT witness (stays loud / un-witnessed):**
  - **Entailment** — whether the on-property passage *supports* vs merely *mentions* the property. The `supports`/`contradicts` **direction is the model's label.** A polarity flip that shares the property axis word (e.g. claim "temperature-**independent**", quote "strong temperature **dependence**") passes the topicality floor; only the model's direction label distinguishes it. → mitigated by the **contradiction guard** (§6) and the deferred **≥2** (a single Tier-2 source is thin evidence).
  - **Independence** — not witnessable from the `Article` schema (§8).

Every section below is held to this boundary; the spec claims presence + on-property topicality + anti-fabrication, never entailment.

---

## 3. Pipeline (`ground_claim`, revised)

```
ground_claim(claim, …):
  formatted, articles = search_articles(backend, claim.statement)   # abstracts (top-10), already retrieved
  label_to_article = {f"A{i}": a for i, a in enumerate(articles, 1)}
  subject = _retrieval_saturated_tokens(articles, cfg)              # §5: entity/topic tokens (code, ungameable)

  LLM (one call) → SUPPORT, BASIS, asserted_property, and a citations JSON:
        [{label:"A1", direction:"supports"|"contradicts", quote:"…"}]

  passing, contradicted = [], False
  for c in citations:
     art = label_to_article.get(c.label)                           # must be a real retrieved article (kept)
     if art is None: continue
     if not _support_quote_valid(c.quote, art.summary, claim.statement, asserted_property, subject, cfg):
         continue                                                  # fail-closed: fabricated / cross-sentence / off-property
     if c.direction == "contradicts":
         contradicted = True                                       # §6: a code-witnessed contradiction
     elif c.direction == "supports":
         passing.append(art)                                       # a code-witnessed on-property supporting citation

  code_witnessed = len(_dedup(passing))                            # §7: distinct works
  code_witnessed = min(code_witnessed, len(articles))             # code cap (cannot exceed retrieval)
  independent_sources = min(as_int(SUPPORT.independent_sources), code_witnessed)   # D8 cap kept

  verdict = map_support_to_verdict(SUPPORT, independent_sources)   # unchanged mapping
  if contradicted and verdict == "pass":
      verdict = "uncertain"                                        # §6 contradiction guard (force-downgrade)

  basis = SUPPORT.basis + loud per-citation quotes (supports + CONTRADICTION: ...)   # §8 honest, loud
  return CheckRecord(lens="grounder", verdict, basis, sources=<deduped, quote in basis>, independent_sources, tick)
```

The retrieval, the `[A1]→article` mapping, `map_support_to_verdict`, and the `CheckRecord` shape are reused; the changes are the citations-JSON + the quote gate + the property floor + the contradiction guard + the dedup/cap + the honest basis.

---

## 4. The quote gate (`_support_quote_valid`, pure code)

`_support_quote_valid(quote, source_text, claim_statement, asserted_property, subject_tokens, cfg) -> bool`, in `valagents/grounding.py`. Reuses Tier-1's `_norm`/`_content_tokens`. Returns True iff **all**:

1. **Anti-fabrication:** `_norm(quote)` is a literal substring of `_norm(source_text)` (the cited article's abstract). *(Mirror `_norm` on BOTH this and the contradicts path — the prior nit.)*
2. **Sentence-bounded:** the quote does not span a sentence boundary — it lies within a single sentence of `source_text` (split on `.?!` with abbreviation tolerance). Kills the "`…no long-range order is observed. The lattice…`" → "`order is observed. The lattice`" inversion (anti-fabrication-by-splicing).
3. **Substantial:** `≥ quote_min_tokens` (the existing GroundCfg knob, 6) whitespace word-tokens.
4. **On-property topicality (§5):** the quote's content tokens overlap `prop_distinctive` (the claim's distinctive asserted-property tokens) — NOT merely the claim's entity/topic tokens.

Any failure → the citation does not count (fail-closed). This gate witnesses presence + on-property topicality + anti-fabrication; it does **not** evaluate direction (that's the model's label, §6).

---

## 5. The property floor (non-vacuous topicality — the twice-reviewed piece)

**Why a plain claim-overlap floor is vacuous (measured):** `search_articles` queries arXiv on `claim.statement` itself, so every retrieved abstract is on-topic by construction; the quote is a substring of the abstract, so its tokens are a subset of tokens retrieval already maximized against the claim. A plain "quote shares claim content-tokens" floor re-tests what retrieval guaranteed — and (verified) a *contradicting* quote and an off-property *synthesis-method* sentence both pass it. The floor must test the claim's **asserted property**, not its **subject/entity**.

**Design — claim-derived property, retrieval-saturated subject subtracted:**
- The grounder emits `asserted_property`: a short phrase naming what the claim asserts about its subject (e.g. claim "the PSD of YbZn₂GaO₅ is temperature-independent" → `asserted_property = "temperature-independent"`).
- **Guard 1 (claim-derived — pure code):** `_content_tokens(asserted_property) ⊆ _content_tokens(claim.statement)`. The property must come *from the claim*; the model cannot invent a property the claim never made. Fail → the floor cannot be evaluated → no citation can pass → `uncertain` (fail-closed).
- **`subject_tokens = _retrieval_saturated_tokens(articles, cfg)`** — content tokens appearing in `≥ subject_saturation_frac` (default **0.6**) of the retrieved abstracts. Because retrieval matched every hit on the entity/topic, these are exactly the vacuous (entity/subject) tokens. **This is code-derived from the abstracts, NOT the model — so it is ungameable** (the model cannot under-specify a subject it doesn't control; this closes the "property-as-subject" gap *by construction*, not by guarding against gaming).
- **`prop_distinctive = _content_tokens(asserted_property) − subject_tokens`** — the claim's asserted-property tokens that retrieval did **not** saturate (e.g. `{temperature, independent} − {ybzn2gao5, psd, …} = {temperature, independent}`; if "temperature" is itself saturated across abstracts, it too drops, leaving the distinctive `{independent}`).
- **Guard 2 (non-vacuous — pure code):** `prop_distinctive` must be non-empty. If the asserted property is entirely subject/entity tokens (a property-as-subject emission, or a property fully saturated by retrieval), the floor would be vacuous → `uncertain` (fail-closed).
- **The floor:** the quote's content tokens overlap `prop_distinctive` (≥1).

**What it catches / doesn't (honest):** catches the measured off-property false-positives (a synthesis sentence sharing only `ybzn2gao5` → no `prop_distinctive` overlap → fails; an on-material sentence that never mentions the property → fails). Does **not** catch a polarity flip that contains the distinctive property word (claim "temperature-independent", quote "not temperature-independent") — that passes the floor; **direction stays loud** and is carried by §6 + the deferred ≥2. The floor witnesses *on-property topicality*, never *entailment*.

*(Considered & rejected: an LLM-emitted `subject` phrase to subtract — gameable (the model under-specifies the subject to inflate `prop_distinctive`, reopening vacuity). Retrieval-saturation is ungameable and needs no anti-gaming guard. T2-D4.)*

New GroundCfg knob: `subject_saturation_frac: float = 0.6`.

---

## 6. Direction + the contradiction guard (Blocker 2 — load-bearing)

The `direction` (supports/contradicts) is the **model's loud label** — code does not adjudicate it. But a code-witnessed contradiction must be load-bearing, and today it is not:

**The hole (verified):** `status` fails only on `verdict=="fail"`; the `CONTRADICTION:` basis prefix is honored **only** inside `_math_uncertainty_is_nonblocking` (mathematical-claims-only, uncertain-only). So for an empirical/mechanistic claim, a `supports`+`contradicts` citation pair → `pass` with a real contradiction sitting **ignored** in the basis.

**The guard:** in `ground_claim`, if **any** `contradicts` citation passes its quote gate (anti-fabrication + sentence-bound + substantial; *the property floor is not required for a contradiction* — a contradicting passage need not share the asserted-property polarity word), set `contradicted = True`, and **force the grounder verdict to `uncertain`** if it would otherwise be `pass` (before building the `CheckRecord`). **Not `fail`** — this preserves the grounder's existing stance ("do not auto-refute a novel claim merely because the literature predates it"); it refuses to let a claim *validate* while a real, verbatim, retrieved contradiction stands. `artifact.py` is untouched; the guard lives in `ground_claim`. The contradicting quote is surfaced loud (`CONTRADICTION:` in `basis`).

---

## 7. Counting: dedup + caps

- **Dedup (T2-D5):** count **distinct works**, not per-citation/per-label. Two arXiv hits for the same paper (preprint + published, or v1/v2) must count once. Dedup key: `references.normalize_id(url)` (arXiv-id base / DOI), falling back to a normalized title (casefold, strip punctuation/whitespace). `_dedup(passing)` collapses by this key.
- **Code cap (T2-D6):** `code_witnessed = min(len(_dedup(passing)), len(articles))` — cannot exceed the retrieved count (the LLM controls the citations and the self-reported `independent_sources`, but not the retrieval count).
- **D8 cap kept:** `independent_sources = min(as_int(SUPPORT.independent_sources), code_witnessed)` — the model can still *downgrade* but never inflate beyond the code-witnessed count.
- `map_support_to_verdict(SUPPORT, independent_sources)` unchanged; then the §6 contradiction guard.

---

## 8. The honest rename (independence is not witnessed)

`relation="independent"` is hardcoded for every retrieved source (`grounder.py:63-71`) and the `Article` schema carries no authors — **code cannot witness independence here.** The gate field is `independent_sources` (in `CheckRecord`, read by `artifact.py` — **untouched**, so the field name stays), but Tier-2 stops *claiming* it means independence:
- The `basis` and the decision log name the credited quantity honestly: **"quote-verified retrieved source(s)"** — a real, deduped, retrieved article carrying a code-checked on-property supporting passage. Independence (distinct author groups/lineages) is **LLM-asserted, not code-witnessed**, and the basis says so.
- This is a *documentation/labeling* correction, not a field rename (renaming `independent_sources` would touch `artifact.py`). The number is now *more* honest (quote-verified, not retrieval-existence), and we describe it as what it is.

---

## 9. Fail-closed, gate purity, config

- **Fail-closed everywhere:** no citations / quote not in abstract / cross-sentence / non-substantial / off-property (no `prop_distinctive` overlap) / Guard 1 or 2 fails / `SUPPORT != "supported"` → `code_witnessed=0` or the existing `map_support_to_verdict` downgrade → **`uncertain`**. Abstract-substrate (as Tier-1): the supporting passage must be in the *abstract*; numbers/claims buried in the body → `uncertain` (grounder passes get rarer and earned).
- **Gate purity:** `artifact.py` (`_evaluate`, `status`, `_has_independent_external_check`, `verdict_class`, the `≥1` bar) and `map_support_to_verdict` are **untouched**. Tier-2 only changes (a) how `independent_sources` is computed (code-witnessed, deduped, capped) and (b) the contradiction-downgrade — both inside `ground_claim`. `ground_novelty` is untouched.
- **Config (`GroundCfg`):** add `subject_saturation_frac: float = 0.6`; reuse `quote_min_tokens` (6).
- **Determinism boundary:** retrieval + the grounder LLM call are non-deterministic (agent layer); the quote gate + floor + dedup + caps are pure code. The credited result (quotes, deduped sources, count) is recorded in the `CheckRecord` for reproducibility/showability, as Tier-1.

---

## 10. Testing

Pure-code helpers tested in isolation (FakeLLM/Fake backend for the agent path):
- **`_support_quote_valid`** — supports (quote ∈ abstract, single sentence, substantial, overlaps `prop_distinctive`); **fabricated** quote (not in abstract) → False; **cross-sentence** splice → False; **off-property** (overlaps only subject/entity tokens, the measured `ybzn2gao5`-only synthesis sentence) → False; non-substantial → False.
- **Property floor** — Guard 1: an `asserted_property` with a token not in the claim → fail-closed (uncertain). Guard 2: `asserted_property` entirely subject/entity (property-as-subject) → `prop_distinctive` empty → uncertain. Off-property synthesis sentence (overlaps only retrieval-saturated subject tokens, the measured `ybzn2gao5`-only case) → fails the floor. **Honest-boundary pin (no over-claim):** a contradicting quote that is *mislabeled* `supports` and carries the distinctive property word (a negation, e.g. "not temperature-independent") *passes* the floor — this is the irreducible polarity residual; the test asserts it is NOT counted only when the model labels it `contradicts` (§6 guard), and is otherwise carried by the deferred ≥2, never claimed "caught" by the floor.
- **Contradiction guard (§6)** — a `supports`+`contradicts` pair where both quotes pass → verdict forced to `uncertain` (NOT pass, NOT fail); the `CONTRADICTION:` quote in basis. A lone `contradicts` → uncertain. (This is the regression test for the measured Blocker-2 hole.)
- **Dedup/cap (§7)** — two citations to the same work (preprint+published URLs) → `code_witnessed==1`; `code_witnessed ≤ len(articles)`; `min(llm_independent, code_witnessed)` (model can't inflate).
- **Verdict mapping** — `supported` + ≥1 quote-passing supporting citation → pass; `supported` + 0 passing (all fabricated/off-property) → uncertain (the say-so→code-witnessed upgrade); backend-off / no retrieval → uncertain (no regression).
- **Gate purity** — `artifact.py`/`map_support_to_verdict` untouched; a grounder pass still needs `independent_sources≥1` (unchanged ≥1 bar); existing grounder/agent-lens tests stay green (fixtures updated only where they relied on the old retrieval-existence credit — that change is the say-so→code-witnessed correction, not a weakened assertion).

---

## 11. Decision log
- **T2-D1 (scope)** Grounder `[A1]` support-relation only; symbolic `expected_source` and the `≥2` bar deferred.
- **T2-D2 (honest boundary — the core)** The credit witnesses **presence + on-property topicality + anti-fabrication**, NOT entailment and NOT independence. "Supports" direction is the model's loud label. The prior "quote-backed support" framing was an over-claim (a contradicting quote passed the plain floor; the credit was quote-backed *presence* + say-so *direction*). The spec, `basis`, and code comments claim only what is witnessed.
- **T2-D3 (mechanism)** F1/F3-for-reading: the LLM emits per-citation `{direction, verbatim_quote}` + the claim's `asserted_property`; pure code adjudicates the quote (in-bytes, sentence-bound, substantial, on-property). No NLI model (relocates say-so — rejected).
- **T2-D4 (property floor — claim-derived, retrieval-saturated subtraction)** The floor tests the claim's **distinctive asserted-property** tokens, because a plain claim-overlap floor is vacuous (retrieval already maximized claim-topicality, and the quote is a substring of the on-topic abstract). `asserted_property` is **claim-derived** (Guard 1: tokens ⊆ claim — no invention); the subtracted **subject** is the **retrieval-saturated** tokens (≥`subject_saturation_frac` of abstracts) — **code-derived and ungameable**, closing the property-as-subject gap by construction (an LLM-emitted subject was rejected as gameable). Guard 2: `prop_distinctive` non-empty else uncertain. The floor does not catch a polarity flip carrying the distinctive word — direction stays loud.
- **T2-D5 (dedup)** Count distinct works (`normalize_id` / normalized title), not per-citation — preprint+published collapse to one; protects the future `≥2` slice.
- **T2-D6 (code cap)** `code_witnessed ≤ len(retrieved)`, and `independent_sources = min(llm_self_reported, code_witnessed)` — the model can downgrade, never inflate; the only ceilings are code-controlled.
- **T2-D7 (contradiction guard — load-bearing)** A passing `contradicts` citation forces the grounder verdict to `uncertain` (not `fail`) before the `CheckRecord` — because the gate's `CONTRADICTION:` handling is mathematical-claims-only, so otherwise a real contradiction is ignored for empirical/mechanistic claims. `artifact.py` untouched.
- **T2-D8 (independence un-witnessed; rename)** Independence is not witnessable (no authors in `Article`); stop claiming it. The credited quantity is "quote-verified retrieved sources"; independence is LLM-asserted, said so in `basis`. Field name unchanged (`artifact.py` untouched).
- **T2-D9 (≥1 kept; ≥2 the one epistemic story)** A single Tier-2 source is thin (presence + say-so direction). `≥1` is still a strict upgrade over today's bare-label pass, so keep it (`artifact.py` untouched). The deferred **≥2-corroboration** is the real answer to the entailment residual — Tier-2's quote-verified, deduped count is what makes a future `≥2` sound.

---

## 12. Build slices (order)
1. **Pure-code quote gate + property floor (no network/LLM) — the honesty core.** In `valagents/grounding.py`: `_sentence_bounded` (single-sentence containment), `_retrieval_saturated_tokens(articles, frac)`, `_support_quote_valid(quote, source_text, claim_statement, asserted_property, subject_tokens, cfg)` (Guards 1/2 + the four checks); `GroundCfg.subject_saturation_frac`. Exhaustive unit tests incl. the measured off-property + contradicting-quote + cross-sentence + property-as-subject cases.
2. **Grounder prompt + agent wiring.** `GROUNDER_CLAIM`: emit `asserted_property` + a citations JSON (`[{label,direction,quote}]`); `ground_claim`: parse (JSON, `_extract_json` pattern — quotes contain `|`), run the gate per citation, dedup, caps, `min(llm,code)`, the §6 contradiction guard, the honest loud `basis`. Integration tests (FakeLLM citations JSON + fake backend abstracts): supports→pass, fabricated/off-property→uncertain, contradicts→uncertain, dedup, backend-off→uncertain, gate-purity pins green.
