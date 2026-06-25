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
- **Raising the bar to `≥2`** — Tier-2 makes it *available* (the grounder co-signer is no longer pure say-so) but does not raise it; that is a gate change in `artifact.py` with cross-lens semantics, its own slice (T2-D9). **This slice also carries the co-saturation recall fix** (§5): the fix needs a *non-saturation* subject signal to credit well-corroborated properties, and corroboration is exactly the ≥2 lever — so the two land together rather than risking a fail-open patch now.
- **Entailment adjudication** — whether a passage logically *backs* the claim is semantic and not code-decidable here. It stays **loud** (the verbatim quote in `basis`), mitigated structurally by the contradiction guard + the deferred ≥2-corroboration. We do **not** add an NLI model (that relocates say-so to another model — rejected).
- **Independence adjudication** — the `Article` schema has no authors; code cannot witness it. `relation="independent"` is LLM-asserted/hardcoded; we stop claiming code witnesses it.

---

## 2. The honest boundary (the load-bearing correction — read first)

A live review of the prior framing established, against the code, that "quote-backed support" was an over-claim. The honest decomposition of what each piece witnesses:

- **Code witnesses (new in Tier-2):**
  1. **Anti-fabrication** — the quote is a literal, **sentence-bounded** substring of the cited article's retrieved abstract (`_norm`-normalized, both directions). The model cannot cite a passage that isn't there, and cannot invert meaning by spanning a sentence boundary.
  2. **On-property topicality (non-vacuous, CODE-derived)** — the quote contains **ALL** the claim's **distinctive** tokens (require-ALL, T2-D11), where the distinctive set is `_content_tokens(claim.statement) − saturated_subject` computed **in code** (T2-D12 — the model supplies no property; it cannot under-declare to collapse the set). Not merely the entity/topic tokens retrieval already saturated, and not just one fragment of a compound property. This is what makes "topicality" mean something (retrieval queries on `claim.statement`, so plain claim-overlap is vacuous — §5). *(Two disclosed costs: (a) co-saturation — if the whole property is topic-defining it self-subtracts → `prop_distinctive` empties → `uncertain` (fail-closed); (b) entailment residual — require-ALL is set-containment not adjacency/sense, so a quote carrying all distinctive tokens in unrelated senses still passes (the single-token set is the sharpest case); entailment is never witnessed here. §5/§10.)*
  3. **Real retrieved, deduped source** — the cited `[A1]` maps to an actually-retrieved article with a URL, counted once per distinct work.
- **The model's only inputs to the floor are the citations (quote + direction) and the SUPPORT/count tail** — every one independently checked (admissibility + require-ALL against the code-derived set) or capped (`min(llm, code_witnessed)`). No model say-so chooses what counts as "on-property" (T2-D12).
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

  LLM (one call) → SUPPORT, BASIS, and a citations JSON ONLY (no property/subject — T2-D12):
        [{label:"A1", direction:"supports"|"contradicts", quote:"…"}]

  # §5 property floor (T2-D12): the distinctive set is CODE-derived from the CLAIM minus the
  # retrieval-saturated subject. The model supplies NEITHER the property NOR the subject — it has no
  # lever over the set, so it cannot under-declare/absorb to launder credit.
  subject_tokens = _retrieval_saturated_tokens(articles, cfg)       # code-derived, model-independent
  prop_distinctive = _content_tokens(claim.statement) - subject_tokens   # computed inside _support_quote_valid

  passing, contradicted = [], False
  for c in citations:
     art = label_to_article.get(c.label)                           # must be a real retrieved article (kept)
     if art is None: continue
     if not _quote_admissible(c.quote, art.summary, cfg):          # SHARED gate: anti-fab + sentence-bound + substantial
         continue
     if c.direction == "contradicts":
         contradicted = True                                       # §6: admissible only; NO property floor
     elif c.direction == "supports" and prop_distinctive and (prop_distinctive <= _content_tokens(c.quote)):
         passing.append(art)                                       # §5 on-property floor: quote contains ALL distinctive CLAIM tokens (T2-D12)

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

## 4. The quote gate (pure code) — factored into a shared admissibility check + a supports-only property floor

The gate splits in two, because `supports` and `contradicts` need different floors (§6): a contradiction carries the *opposite* polarity word, so requiring the distinctive-property overlap would reject genuine contradictions.

**Shared admissibility — `_quote_admissible(quote, source_text, cfg) -> bool`** (BOTH directions must pass; reuses Tier-1's `_norm`):
1. **Anti-fabrication:** `_norm(quote)` is a literal substring of `_norm(source_text)` (the cited article's abstract). *(Mirror `_norm` on BOTH directions — the prior nit.)*
2. **Sentence-bounded:** the quote lies within a single sentence of `source_text` (split on `.?!` with abbreviation tolerance) — kills the "`…no long-range order is observed. The lattice…`" → "`order is observed. The lattice`" splice inversion.
3. **Substantial:** `≥ quote_min_tokens` (existing GroundCfg knob, 6) whitespace word-tokens.

**On-property floor (supports ONLY, §5) — `_support_quote_valid(quote, source_text, claim_statement, subject_tokens, min_tokens)`:** `_quote_admissible(...)` **and** the quote contains **ALL** of `prop_distinctive`, where `prop_distinctive = _content_tokens(claim_statement) - subject_tokens` is **derived in code from the CLAIM** (T2-D12 — the model supplies no property). Guard: `prop_distinctive` non-empty else fail (claim fully saturated). **Require-ALL, not any-overlap (T2-D11):** a compound property like `temperature-independent` tokenizes to `{temperature, independent}`; crediting a quote that shares only one fragment lets an off-property quote earn credit (`temperature` variation of a *different* observable; `independent` in an unrelated sense). Requiring every distinctive token closes that. **Claim-derived, not model-declared (T2-D12):** an earlier design let the model emit `asserted_property`; the model could under-declare it (or absorb property words into a `subject_phrase`) to collapse the distinctive set to one generic token and launder an off-property quote — model say-so determining a pass. Deriving the set from the claim minus the code-saturated subject removes the model's lever entirely; `subject_tokens` is now saturation-only (model-independent), and subtracting it is a recall aid. Residual: a single-token distinctive set, where ALL = any (irreducible without semantics — §5).

So: a **supports** citation must pass `_quote_admissible` **and** the property floor; a **contradicts** citation must pass `_quote_admissible` **only** (§6) — a trivial or spliced "contradiction" therefore *cannot* force the downgrade (it must still be a substantial, single-sentence, real passage), but it need not carry the property polarity word. Any failure → the citation does not count (fail-closed). The gate witnesses presence + (for supports) on-property topicality + anti-fabrication; it does **not** evaluate direction (the model's label, §6) or entailment.

---

## 5. The property floor (non-vacuous topicality — the twice-reviewed piece)

**Why a plain claim-overlap floor is vacuous (measured):** `search_articles` queries arXiv on `claim.statement` itself, so every retrieved abstract is on-topic by construction; the quote is a substring of the abstract, so its tokens are a subset of tokens retrieval already maximized against the claim. A plain "quote shares claim content-tokens" floor re-tests what retrieval guaranteed — and (verified) a *contradicting* quote and an off-property *synthesis-method* sentence both pass it. The floor must test the claim's **asserted property**, not its **subject/entity**.

**Design — the distinctive set is CODE-derived from the claim (T2-D12):** `prop_distinctive = _content_tokens(claim.statement) − subject_tokens`, where `subject_tokens = _retrieval_saturated_tokens(articles, subject_saturation_frac=0.6)` — content tokens appearing in ≥0.6 of the retrieved abstracts (the entity/topic retrieval saturated on; **code-derived, model-independent**). **The model supplies NEITHER the property NOR the subject** — it emits only the SUPPORT/count tail and the citations (quote + direction).
- **Guard (non-vacuous — pure code):** `prop_distinctive` non-empty else `uncertain` (the claim's content fully co-saturated the corpus).
- **The floor (require-ALL — T2-D11):** `prop_distinctive <= _content_tokens(quote)` — the quote must contain **every** distinctive claim token, not just one. `_content_tokens` splits on non-alphanumerics, so a compound property `temperature-independent` → `{temperature, independent}`; an *any-overlap* floor credited an off-property quote that shared only the high-frequency fragment (`temperature` variation of a *different* observable; `independent` in an unrelated sense — a measured false-credit). Require-ALL closes it.

**Why code-derived, not model-declared (T2-D12 — the cardinal-rule fix):** an earlier design had the model emit `asserted_property` (+ `subject_phrase`), with `prop_distinctive = asserted_property − (saturated ∪ subject_phrase)` and a Guard-1 check `asserted_property ⊆ claim`. The final whole-branch review constructed (and we verified against the shipped code) that the model could **under-declare** the property — or absorb property words into `subject_phrase` — to shrink `prop_distinctive` to one generic token, then cite an off-property quote carrying that token (claim "specific heat shows **linear** temperature dependence" → declare property `"linear"` → a "linear background subtraction" quote passes). Guard 1 only blocked property *invention*, never *under-declaration*. That is a `pass` resting on the model's own property declaration — the say-so laundering the project exists to prevent. **Deriving `prop_distinctive` from the claim minus the code-saturated subject removes the model's lever entirely:** saturation is computed from the actual retrieved corpus (the model doesn't control the query or the hits) and the claim is fixed, so the model influences only *which real quote it cites* and *the count* — both independently checked and capped. `asserted_property`/`subject_phrase` are gone from the prompt and the floor.

**Subtraction is saturation-only — a recall aid, not soundness:** subtracting `subject_tokens` spares a genuine quote from restating the saturated subject. Disclosed recall cost: on a **thin/novel corpus** the subject formula doesn't saturate, so it stays in `prop_distinctive` and a passing quote must name it; and filler verbs in the claim (e.g. "shows") become required. Both are recall losses, **fail-closed** (never a false-credit). The earlier `subject_phrase` union (added for thin-corpus recall) is removed — it was a model lever, and T2-D12 closes the leak it created; the thin-corpus recall it bought is the accepted price of an ungameable floor.

**Co-saturation is fail-closed — a DECISION, not a residual to patch:** if retrieval keys on the *property* (all hits about superconductivity), the property co-saturates → it is subtracted → `prop_distinctive` empties → Guard → `uncertain`. Safe (recall drops for topic-defining/well-corroborated-property claims; never a false-support). A "refine-never-empty" variant (restore saturation-removed tokens when they'd empty the set) was **rejected** (T2-D10): it reopens a rich-corpus leak. The genuine co-saturation fix needs a **non-saturation subject signal** — deferred to the ≥2 slice (§1, T2-D9), where corroboration is the right lever anyway.

**What it catches / doesn't (honest):** catches the measured off-property false-credits — a synthesis sentence sharing only the subject/formula, AND (require-ALL, T2-D11) a quote sharing only one fragment of a compound property (`temperature` of a different observable; `independent` in an unrelated sense). Does **not** catch the **entailment residual**: require-ALL is set-*containment*, not *adjacency or sense* (`_content_tokens` returns a set), so any quote carrying **all** distinctive tokens still passes even if it (a) flips polarity ("not temperature-independent"), or (b) uses the tokens in unrelated senses scattered across the sentence ("temperature was logged while the independent referee reviewed…"). The single-token distinctive set (a one-word property, or a compound co-saturated to one fragment) is the *sharpest* case (then require-ALL = any), but the residual is general: the floor witnesses that the distinctive property tokens are all *present*, never that the passage *asserts the property*. This is irreducible without semantics; entailment is carried by the model's `direction` label (§6) + the deferred ≥2, never claimed by the floor. The floor witnesses *on-property topicality*, never *entailment*.

New GroundCfg knob: `subject_saturation_frac: float = 0.6`.

---

## 6. Direction + the contradiction guard (Blocker 2 — load-bearing)

The `direction` (supports/contradicts) is the **model's loud label** — code does not adjudicate it. But a code-witnessed contradiction must be load-bearing, and today it is not:

**The hole (verified):** `status` fails only on `verdict=="fail"`; the `CONTRADICTION:` basis prefix is honored **only** inside `_math_uncertainty_is_nonblocking` (mathematical-claims-only, uncertain-only). So for an empirical/mechanistic claim, a `supports`+`contradicts` citation pair → `pass` with a real contradiction sitting **ignored** in the basis.

**The guard:** in `ground_claim`, if **any** `contradicts` citation passes **`_quote_admissible`** (anti-fabrication + sentence-bound + substantial — the §4 shared check; the property floor is *not* required, because a contradiction carries the opposite polarity word), set `contradicted = True`, and **force the grounder verdict to `uncertain`** if it would otherwise be `pass` (before building the `CheckRecord`). **Not `fail`** — this preserves the grounder's existing stance ("do not auto-refute a novel claim merely because the literature predates it"); it refuses to let a claim *validate* while a real, verbatim, retrieved contradiction stands. Requiring `_quote_admissible` (substantial + single-sentence + in-bytes) means a trivial or spliced "contradiction" *cannot* force the downgrade. `artifact.py` is untouched; the guard lives in `ground_claim`. The contradicting quote is surfaced loud (`CONTRADICTION:` in `basis`).

---

## 7. Counting: dedup + caps

- **Dedup (T2-D5):** count **distinct works**, not per-citation/per-label. Two arXiv hits for the same paper (preprint + published, or v1/v2) must count once. Dedup key: `references.normalize_id(url)` (arXiv-id base / DOI), falling back to a normalized title (casefold, strip punctuation/whitespace). `_dedup(passing)` collapses by this key. *(Accepted edge: same-work-different-title — e.g. a preprint and journal version with a retitled abstract and distinct ids — slips dedup; acceptable for v1, and harmless at the ≥1 bar.)*
- **Code cap (T2-D6):** `code_witnessed = min(len(_dedup(passing)), len(articles))` — cannot exceed the retrieved count (the LLM controls the citations and the self-reported `independent_sources`, but not the retrieval count).
- **D8 cap kept:** `independent_sources = min(as_int(SUPPORT.independent_sources), code_witnessed)` — the model can still *downgrade* but never inflate beyond the code-witnessed count.
- `map_support_to_verdict(SUPPORT, independent_sources)` unchanged; then the §6 contradiction guard.

---

## 8. The honest rename (independence is not witnessed)

`relation="independent"` is hardcoded for every retrieved source (`grounder.py:63-71`) and the `Article` schema carries no authors — **code cannot witness independence here.** The gate field is `independent_sources` (in `CheckRecord`, read by `artifact.py` — **untouched**, so the field name stays), but Tier-2 stops *claiming* it means independence:
- The `basis` and the decision log name the credited quantity honestly: **"quote-verified retrieved source(s)"** — a real, deduped, retrieved article carrying a code-checked on-property supporting passage. Independence (distinct author groups/lineages) is **LLM-asserted, not code-witnessed**, and the basis says so.
- **The disclosure ships in the basis itself, not only in docs** (load-bearing — without it §8 is unmet in the artifact a reviewer reads). When `independent_sources >= 1`, `ground_claim` appends to the basis: *"[grounder credit: N retrieved source(s) carrying a code-witnessed verbatim on-property passage; entailment & independence are the model's label, not code-witnessed]"*. Gate-safe: the `CONTRADICTION:` prefix (math-claim handling) stays at the front; the suffix is free text nothing parses.
- This is a *documentation/labeling* correction, not a field rename (renaming `independent_sources` would touch `artifact.py`). The number is now *more* honest (quote-verified, not retrieval-existence), and we describe it as what it is.
- **Earmark:** the actual field rename (`independent_sources` → e.g. `corroborating_sources`) belongs to the deferred **≥2 slice (T2-D9)** — that slice already edits `artifact.py`'s gate, so the rename rides along there instead of forcing a gate-touching change now. Recorded so the rename isn't forgotten and isn't done prematurely.

---

## 9. Fail-closed, gate purity, config

- **Fail-closed everywhere:** no citations / quote not in abstract / cross-sentence / non-substantial / off-property (no `prop_distinctive` overlap) / Guard 1 or 2 fails / `SUPPORT != "supported"` → `code_witnessed=0` or the existing `map_support_to_verdict` downgrade → **`uncertain`**. Abstract-substrate (as Tier-1): the supporting passage must be in the *abstract*; numbers/claims buried in the body → `uncertain` (grounder passes get rarer and earned).
- **Gate purity:** `artifact.py` (`_evaluate`, `status`, `_has_independent_external_check`, `verdict_class`, the `≥1` bar) and `map_support_to_verdict` are **untouched**. Tier-2 only changes (a) how `independent_sources` is computed (code-witnessed, deduped, capped) and (b) the contradiction-downgrade — both inside `ground_claim`. `ground_novelty` is untouched.
- **Config (`GroundCfg`):** add `subject_saturation_frac: float = 0.6`; reuse `quote_min_tokens` (6).
- **Determinism boundary:** retrieval + the grounder LLM call are non-deterministic (agent layer); the quote gate + floor + dedup + caps are pure code. The credited result (quotes, deduped sources, count) is recorded in the `CheckRecord` for reproducibility/showability, as Tier-1.

---

## 10. Testing

Pure-code helpers tested in isolation (FakeLLM/Fake backend for the agent path):
- **`_support_quote_valid`** — supports (quote ∈ abstract, single sentence, substantial, contains **ALL** of `prop_distinctive`); **fabricated** quote (not in abstract) → False; **cross-sentence** splice → False; **off-property** (subject/entity tokens only) → False; non-substantial → False.
- **Compound-fragment regression (require-ALL — T2-D11, the measured false-credit).** Claim "noise PSD of YbZn2GaO5 is **temperature-independent**" (`prop_distinctive ⊇ {temperature, independent}`): a quote sharing only one fragment must **fail** — "magnetization shows strong **temperature** variation" (off-observable) → False; "results were **independent** of the growth batch" (unrelated sense) → False; the genuine "noise PSD is **temperature-independent**" (both tokens) → True. Under the old any-overlap floor all three were credited; require-ALL closes it.
- **Property floor (claim-derived — T2-D12)** — on-property: a genuine quote containing the distinctive `claim − subject` tokens → passes; off-property synthesis (subject-only) → fails. **Under-declaration closed (T2-D12 regression):** the distinctive set is computed from the claim in code, so there is *no* model property to under-declare — the final-review attack (claim "specific heat shows linear temperature dependence", a "linear background subtraction" quote) → `False`; the genuine quote → `True`. Guard (claim fully saturated → `prop_distinctive` empty) → uncertain. **Subject-subtraction-as-recall:** a genuine quote stating the property but omitting the subject formula passes once the subject is subtracted, and false-rejects without subtraction (subtraction is a recall aid, not soundness). **Honest-boundary pins (no over-claim):** (a) a contradicting quote *mislabeled* `supports` that carries the full distinctive set (a negation, "not temperature-independent") passes the floor — polarity residual, caught only when labeled `contradicts` (§6); (b) a quote carrying all distinctive tokens in unrelated senses passes (entailment residual; single-token set is the sharpest case) — irreducible without semantics. Both carried by §6 + the deferred ≥2, never claimed "caught" by the floor.
- **Co-saturation (fail-closed).** A corpus where the *property* word is topic-defining (saturates ≥0.6 of abstracts) → it is subtracted → `prop_distinctive` empty → Guard 2 → uncertain (recall-drop, asserted explicitly so the behavior is intentional, not accidental). The thin-corpus formula-leak (subject formula in only 1 of 3 abstracts, a formula-only quote labeled `supports`) now lands `uncertain` via **require-ALL** (the quote lacks the property tokens) regardless of subtraction — verified in the agent-path test.
- **Contradiction guard (§6)** — a `supports`+`contradicts` pair where both quotes pass → verdict forced to `uncertain` (NOT pass, NOT fail); the `CONTRADICTION:` quote in basis. A lone `contradicts` → uncertain. (This is the regression test for the measured Blocker-2 hole.)
- **Dedup/cap (§7)** — two citations to the same work (preprint+published URLs) → `code_witnessed==1`; `code_witnessed ≤ len(articles)`; `min(llm_independent, code_witnessed)` (model can't inflate).
- **Verdict mapping** — `supported` + ≥1 quote-passing supporting citation → pass; `supported` + 0 passing (all fabricated/off-property) → uncertain (the say-so→code-witnessed upgrade); backend-off / no retrieval → uncertain (no regression).
- **Gate purity** — `artifact.py`/`map_support_to_verdict` untouched; a grounder pass still needs `independent_sources≥1` (unchanged ≥1 bar); existing grounder/agent-lens tests stay green (fixtures updated only where they relied on the old retrieval-existence credit — that change is the say-so→code-witnessed correction, not a weakened assertion).

---

## 11. Decision log
- **T2-D1 (scope)** Grounder `[A1]` support-relation only; symbolic `expected_source` and the `≥2` bar deferred.
- **T2-D2 (honest boundary — the core)** The credit witnesses **presence + on-property topicality + anti-fabrication**, NOT entailment and NOT independence. "Supports" direction is the model's loud label. The prior "quote-backed support" framing was an over-claim (a contradicting quote passed the plain floor; the credit was quote-backed *presence* + say-so *direction*). The spec, `basis`, and code comments claim only what is witnessed.
- **T2-D3 (mechanism)** F1/F3-for-reading: the LLM emits per-citation `{direction, verbatim_quote}` + the claim's `asserted_property`; pure code adjudicates the quote (in-bytes, sentence-bound, substantial, on-property). No NLI model (relocates say-so — rejected).
- **T2-D4 (property floor — claim-derived property, UNION subtraction)** *(SUPERSEDED — by T2-D11 (require-ALL became the soundness anchor) and then by T2-D12 (the model-emitted `asserted_property`/`subject_phrase` and the union are REMOVED; the distinctive set is `claim − saturated`, fully code-derived; subtraction is saturation-only and a recall aid). The vacuousness argument below still motivates subtracting the saturated subject; ignore its UNION/model-property mechanics.)* The floor tests the claim's **distinctive asserted-property** tokens, because a plain claim-overlap floor is vacuous (retrieval already maximized claim-topicality, and the quote is a substring of the on-topic abstract). `asserted_property` is **claim-derived** (Guard 1: tokens ⊆ claim — no invention). The subtracted **subject** is a **UNION**: `subject_subtract = _retrieval_saturated_tokens(articles, frac) | _content_tokens(subject_phrase)` — the retrieval-saturated tokens (≥`subject_saturation_frac` of abstracts, **code-derived, ungameable**) *unioned with* the grounder-emitted `subject_phrase` tokens. **Why the union, not saturation alone (the thin-corpus fix):** saturation under-subtracts on a thin/novel corpus — the exact home turf of this tool — because the subject formula appears in too few of the retrieved abstracts to clear `frac`, leaking the formula into `prop_distinctive` and reopening a false-support for novel claims. The union closes that hole. **Why it cannot reopen a false-positive:** `prop_distinctive_union = property − (saturated ∪ subject_phrase) ⊆ property − saturated = prop_distinctive_saturation`; adding union members only *shrinks* `prop_distinctive`, so the floor only gets *stricter* — a gamed/empty `subject_phrase` degrades the floor to saturation-alone (the prior behavior), never weaker. Guard 2: `prop_distinctive` non-empty else uncertain. **Co-saturation fail-closed:** if the property word is *itself* topic-defining (saturates the corpus, e.g. a paper whose every abstract names the property), it is subtracted too → `prop_distinctive` empty → uncertain; recall drops for topic-defining-property claims, by design. The floor does not catch a polarity flip carrying the distinctive word — direction stays loud.
- **T2-D5 (dedup)** Count distinct works (`normalize_id` / normalized title), not per-citation — preprint+published collapse to one; protects the future `≥2` slice.
- **T2-D6 (code cap)** `code_witnessed ≤ len(retrieved)`, and `independent_sources = min(llm_self_reported, code_witnessed)` — the model can downgrade, never inflate; the only ceilings are code-controlled.
- **T2-D7 (contradiction guard — load-bearing)** A passing `contradicts` citation forces the grounder verdict to `uncertain` (not `fail`) before the `CheckRecord` — because the gate's `CONTRADICTION:` handling is mathematical-claims-only, so otherwise a real contradiction is ignored for empirical/mechanistic claims. `artifact.py` untouched.
- **T2-D8 (independence un-witnessed; rename)** Independence is not witnessable (no authors in `Article`); stop claiming it. The credited quantity is "quote-verified retrieved sources"; independence is LLM-asserted, said so in `basis`. Field name unchanged (`artifact.py` untouched).
- **T2-D9 (≥1 kept; ≥2 the one epistemic story)** A single Tier-2 source is thin (presence + say-so direction). `≥1` is still a strict upgrade over today's bare-label pass, so keep it (`artifact.py` untouched). The deferred **≥2-corroboration** is the real answer to the entailment residual — Tier-2's quote-verified, deduped count is what makes a future `≥2` sound.
- **T2-D10 (union subtraction kept; "refine-never-empty" REJECTED — fail-closed wins)** The union's co-saturation recall hit (a well-corroborated/topic-defining property self-subtracts → `uncertain`) is real and points the wrong way (recall inversely correlated with corroboration). A proposed fix — make `subject_phrase` the primary subtractor and *restore* saturation-removed tokens rather than empty the property — was **rejected by adversarial construction**: on a *rich* corpus, a model emitting `asserted_property="<formula>"` + `subject_phrase=""` has the saturated formula restored, and a real formula-only synthesis quote earns false credit. The union denies that exact case (saturation subtracts the formula **regardless of model output**). The error in the "no worse on the threat model" argument: the union's residual is *thin-corpus only* (a non-saturating formula); restoration *adds a rich-corpus residual*. Only **unconditional saturation subtraction is fail-closed**; the co-saturation recall hit is its *price*, disclosed in §2/§8/§10 + the basis, with the genuine fix (a non-saturation subject signal) deferred to the ≥2 slice. Cardinal rule decides: keep fail-closed.
- **T2-D11 (require-ALL distinctive tokens — the floor's soundness anchor; caught in Task-2 review)** The supports floor requires the quote to contain **every** distinctive property token (`prop_distinctive <= _content_tokens(quote)`), not just overlap one. An *any-overlap* floor was found (adversarial Task-2 review) to grant credit to genuinely off-property quotes: `_content_tokens` splits compounds, so `temperature-independent` → `{temperature, independent}`, and a quote sharing only the common fragment — `temperature` *variation* of a different observable, or `independent` in an unrelated sense — passed. This is the Blocker-1 vacuousness, not the accepted polarity residual. Require-ALL closes it and **reframes the whole floor:** soundness now rests on require-ALL; subject-subtraction (T2-D4) drops to a recall aid (don't force the quote to restate the subject). Residual: require-ALL is set-*containment*, not adjacency/sense, so a quote carrying all distinctive tokens in unrelated/scattered senses still passes (the entailment residual; the single-token distinctive set is its sharpest case, where ALL = any) — irreducible without semantics, disclosed (§2/§5/§10). Cost: a paraphrase that avoids the literal property tokens fail-closes (uncertain) — consistent with "on-property topicality, not entailment." Verified against the shipped functions before adoption. *(Note: T2-D11 kept the model-emitted `asserted_property` as the property source; T2-D12 then removes it — the distinctive set is derived from the claim in code.)*
- **T2-D12 (the distinctive set is CODE-derived from the claim; model emits no property/subject — final whole-branch review)** The whole-branch review constructed a cardinal-rule leak the per-task reviews structurally couldn't see: `asserted_property` was model-controlled and Guard 1 only checked it was a *subset* of the claim (blocks invention, not **under-declaration**). A model could declare a single generic token of a compound property (claim "specific heat shows **linear** temperature dependence" → `asserted_property="linear"`), or absorb property words into `subject_phrase`, collapsing `prop_distinctive` to that one token, then cite an off-property quote ("linear background subtraction") → a `pass` resting on the model's own declaration. The reviewer's first-cut fix (subtract subject from the *claim*) was **verified incomplete** — the model still controls `subject_phrase` and can absorb property words. **Resolution (chosen): `prop_distinctive = _content_tokens(claim.statement) − _retrieval_saturated_tokens(articles)` — fully code-derived.** The model emits NO property and NO subject (removed from the prompt); saturation is model-independent and the claim is fixed, so the model's only floor inputs are the cited quote and the count, both independently checked/capped. Verified against the shipped code: the under-declaration attack → `False` for every subject set; genuine support → `True`. Cost (disclosed, fail-closed): stricter recall — the quote must contain every non-saturated claim token, incl. filler verbs and (thin corpus) the subject formula; the `subject_phrase` union (T2-D4/D11) is removed. This is the cardinal rule applied literally — no model say-so decides what counts as on-property.

---

## 12. Build slices (order)
1. **Pure-code quote gate + property floor (no network/LLM) — the honesty core.** In `valagents/grounding.py`: `_sentence_bounded` (single-sentence containment), `_quote_admissible(quote, source_text, min_tokens)` (anti-fab `_norm` substring both directions + sentence-bound + substantial — the shared check used by **both** directions), `_retrieval_saturated_tokens(articles, frac)`, `_support_quote_valid(quote, source_text, claim_statement, subject_tokens, min_tokens)` = `_quote_admissible` **AND** require-ALL of `prop_distinctive = _content_tokens(claim_statement) − subject_tokens` (T2-D12 — derived from the claim, NOT a model property); `GroundCfg.subject_saturation_frac`. `subject_tokens` is the **code-derived saturated set** the caller passes. Exhaustive unit tests incl. off-property, compound-fragment (require-ALL), **under-declaration closed (T2-D12)**, cross-sentence, claim-fully-saturated, and subtraction-as-recall cases.
2. **Grounder prompt + agent wiring.** `GROUNDER_CLAIM`: emit a citations JSON only (`[{label,direction,quote}]`) — no property/subject (T2-D12); `ground_claim`: parse (JSON, `_extract_json` pattern — quotes contain `|`), `subject_tokens = _retrieval_saturated_tokens(articles,cfg)` (saturation-only), run `_support_quote_valid` per `supports` citation and `_quote_admissible` per `contradicts` citation, dedup, caps, `min(llm,code)`, the §6 contradiction guard, the honest loud `basis`. Integration tests (FakeLLM citations JSON + fake backend abstracts): supports→pass, fabricated/off-property→uncertain, **thin-corpus formula-leak→uncertain**, contradicts→uncertain, dedup, backend-off→uncertain, gate-purity pins green.
