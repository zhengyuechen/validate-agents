# validate-agents — Spec 3 Grounding Design (Tier 1: magnitude sourced values)

- **Date:** 2026-06-24
- **Status:** Approved design, pending implementation plan
- **Builds on:** Spec 2 (the magnitude lens — `ComputationPlan` `*_source` fields, `_run_magnitude`, `verdict_to_check`/`verdict_to_attack`), the existing retrieval layer (`web_search.py` backends, `references.py` resolvers), and the gate's `independent_sources` accounting.
- **Status line:** *Grounding closes the one axis still resting on LLM say-so — the asserted numeric inputs to the magnitude verdict. A value counts as grounded only when an independent, code-adjudicated check confirms its named source actually reports it: the model reads (a value + a verbatim quote), code judges (quote ∈ fetched bytes ∧ code-owned unit conversion ∧ numeric match ∧ conditions-compatible). The model never gets to say "yes, grounded"; the conversion that decides the match is never the model's.*
- **One-line goal:** Verify the magnitude lens's three asserted `*_source` numeric inputs (`sensitivity`/`bound`/`closest_prior`) against their named, resolvable sources; a confirmed value lifts the discount on that magnitude PASS (it becomes showable evidence), an unconfirmable one stays discounted (fail-closed), and a literature-contradicted one invalidates the verdict (uncertain, loud).

---

## 1. Scope

### In scope (Tier 1 — locked)
- Ground the **three magnitude `*_source` numeric inputs**, one per `comparison_kind`: `sensitivity` (`sensitivity_source`), `bound` (`bound_source`), `closest_prior_effect` (`closest_prior_source`). These are the asserted numbers that feed the magnitude *arithmetic*, so "the number IS the verdict input" holds and a clean three-way verdict exists.
- A resolvable-locator → fetch-full-text → LLM-extract → **code-adjudicate** pipeline producing `supports` / `unconfirmed` / `contradicts`.
- A **code-owned unit conversion** over a small closed scale-table (the conversion that decides the numeric match is never the LLM's — G-D3).
- A **conditions-compatibility** code predicate (the source's number must be for a quantity compatible with the claim's — both sides — G-D5).
- Gate modulation: `supports` → one real independent `Source` (discount lifts); `unconfirmed` → unchanged (discounted); `contradicts` → verdict downgraded to `uncertain`, loud (G-D6).

### Out of scope (deferred / never)
- **Symbolic `expected_source`** and the **grounder `[A1]` support-relation** (Tier 2 — a graded-correspondence object with different ontology; defer).
- **Full dimensional analysis** (a general units engine). v1 carries a *closed lookup table* of the handful of pairs condensed-matter magnitude claims use; an out-of-table unit is `unconfirmed`, and the missing pair is a one-line add (G-D3).
- **Replacing** an asserted value with a grounded one / re-running the magnitude arithmetic. Grounding **adjudicates, never replaces** (G-D6) — it must never silently manufacture a verdict.
- **Full semantic "this source endorses this exact use"** — code witnesses a *necessary* slice (number + unit + referent + dimensional + token-overlap); the remaining semantic mile is **surfaced loud** for human adjudication (§12), never handed to a second model.

---

## 2. The honesty mechanism (F1/F3 applied to *reading*)

The cardinal rule — *validated = survived an independent, code-adjudicated check, never the model's say-so* — applied one level up. An LLM that "reads the PDF and says it's grounded" just relocates the say-so. So:

- **The LLM only reads** (the F1 analogue): given the fetched source text and the asserted quantity, it returns a *structured extraction* — `{extracted_value, source_unit_token, referent, verbatim_quote}` or `not_found`. It reports the number **in the source's own units**, with the unit token quoted verbatim. It is never asked "does this support the claim?"
- **Code judges** (F3): every assertion the LLM makes is checked against the fetched **bytes** and a **code-owned** conversion — the quote must be a literal substring of the source text, the unit must be in the scale-table, the converted number must match within tolerance, and the referent must be conditions-compatible. The LLM cannot produce a passing result it didn't read out of the actual source.

Pipeline per sourced value:

```
ComputationPlan {value=1e-3, unit="µB", source="arXiv:2104.01234", source_quantity="Yb³⁺ effective moment, T<1K"}
        │
1. resolve locator (arXiv/DOI/URL)  ──references.py──>  fetch full text (PDF→text; abstract fallback)   [agent layer, network]
        │   (no locator / unreachable / wrong paper → fail-closed → unconfirmed)
2. extraction agent (LLM, reads only):  (text, source_quantity, asserted unit) → {extracted_value, source_unit_token, referent, verbatim_quote} | not_found
        │
3. adjudicator (PURE CODE, no LLM):
     quote ∈ normalize(text)?            (anti-fabrication; substantial, referent-binding — §6)
     source_unit_token in SCALE_TABLE?    (else unconfirmed — §7)
     convert extracted_value → asserted unit  (CODE owns this — §7)
     conditions-compatible(source_quantity, referent, dims)?   (§5)
     |converted − asserted| / |asserted| < ground_rtol ?
        → supports / contradicts / unconfirmed
        │
4. modulate verdict_to_check / verdict_to_attack   (§8)
```

---

## 3. Source identification & fetch

- The magnitude designer emits, per sourced value, a **resolvable locator** — an arXiv ID, a DOI, or a URL — in the `*_source` field (not a freeform "PDG2024"). A `source` string that does not parse as one of these → **ungrounded** → `unconfirmed` (the value is still *used* by the executor as today; it simply earns no grounding credit). Backward-compatible: existing freeform sources degrade to `unconfirmed`, no regression.
- Resolution reuses `references.py` (`ArxivResolver`, `DoiResolver`) extended with a **text fetch**: arXiv → fetch the PDF and extract a text layer (abstract guaranteed; full text when the PDF parses); DOI → publisher/crossref abstract; URL → fetch HTML→text. The fetched text is what the quote-check runs against.
- **Hallucinated / wrong locators fail closed automatically.** A hallucinated arXiv ID resolves to nothing (→ `unconfirmed`) or to a *different* paper whose bytes won't contain the asserted quote+number (→ `unconfirmed`). The quote-in-bytes check (§5) is the backstop; the locator never has to be trusted.
- Network lives in the **agent layer only**. The sandbox stays network-isolated (unchanged). Grounding runs in the scheduler/agent context where `web_search`/`references` already make HTTP calls.

---

## 4. The extraction agent

A new small agent (`valagents/agents/value_grounder.py`), prompt `VALUE_GROUNDER`. Structured output (parsed like the other designers), one call per sourced value:

- **Input it is shown:** the fetched source text (truncated to a budget), the **`source_quantity`** descriptor (what the claim's number physically measures + conditions), and the **asserted unit** (so it knows what quantity to look for — NOT the asserted value, so it can't anchor to the desired answer).
- **Output:** `{extracted_value: str, source_unit_token: str, referent: str, verbatim_quote: str}` or `not_found`.
  - `extracted_value` — the number the source reports for that quantity, **in the source's own units**.
  - `source_unit_token` — the unit as written in the source (e.g., `"Gauss"`, `"meV"`).
  - `referent` — the source's name/symbol for the quantity the number measures (e.g., `"ordered moment per Yb ion"`), as it appears in the quote.
  - `verbatim_quote` — a contiguous span of the source text that contains the numeral, the unit token, and the referent (§6).
- **The agent is NOT asked whether the source supports the claim.** It reports what the source says; code decides support. It is explicitly instructed: do not convert units, do not infer, quote verbatim, return `not_found` if the quantity is absent (`not_found` → `unconfirmed`, the honest default).
- **Anti-anchoring:** the asserted *value* is withheld from the agent (only the unit + quantity descriptor are shown), so a `contradicts` outcome is real (the agent reported the source's number without knowing the target), not a coerced agreement.

---

## 5. The adjudicator (pure code — the honesty core, F3)

`ground_value(asserted_value, asserted_unit, source_quantity, extraction, fetched_text, cfg) -> GroundingResult` where `GroundingResult.status ∈ {supports, contradicts, unconfirmed}` plus the loud fields (quote, converted value, referent). Deterministic; no LLM. Order (any failed gate before the numeric step → `unconfirmed`, never `supports`):

1. **Quote-in-bytes (anti-fabrication).** `normalize(quote)` must be a literal substring of `normalize(fetched_text)` (collapse whitespace, NFKC, casefold). Fail → `unconfirmed`.
2. **Quote substantiality / referent-binding (§6).** The quote must assert *this quantity has this value*, not merely contain a number. Fail → `unconfirmed`.
3. **Unit in table & conditions-compatible (§7, this section).** `source_unit_token` must resolve in `SCALE_TABLE` to the **same physical dimension** as `asserted_unit` (else `unconfirmed`). The **conditions-compatibility predicate** then requires, on **both sides**: (a) **dimensional** — source and asserted units are the same kind of quantity (code, robust, from the table); AND (b) **referent overlap** — the extracted `referent` shares ≥1 content token (after stop-word removal) with `source_quantity` (code, a *necessary* gate against a same-dimension-but-different-quantity match, e.g. grounding a magnetic moment to a magnetic *field*). Fail either → `unconfirmed`. *(The full semantic "same measurement under compatible conditions" is the surfaced-loud residual, §12.)*
4. **Code-owned conversion + numeric match.** Convert `extracted_value` from `source_unit_token` to `asserted_unit` using `SCALE_TABLE` (**code does this arithmetic, never the LLM** — G-D3). Then:
   - `|converted − asserted| / |asserted| < ground_rtol` → **supports**.
   - quote+unit+referent all valid but `|converted − asserted| / |asserted| ≥ ground_rtol` → **contradicts** (a real, code-adjudicated disagreement: the source reports a materially different number for a compatible quantity).
   - any earlier gate failed, or `extracted_value` unparseable → **unconfirmed**.

`ground_rtol` is a `GroundCfg` knob (default `0.5` — magnitude claims are order-of-magnitude; a factor-of-2 agreement is "supports", a >2× disagreement on a compatible quantity is "contradicts"; see G-D7 for the justification and the asymmetry).

---

## 6. Quote substantiality / referent-binding rule (the real anti-fabrication strength)

A bare-number quote defeats the substring predicate without technically failing it (the number appears *somewhere*, context-free, grounding nothing). So the quote must carry the **binding** between number and meaning. The quote is **valid** iff it is a contiguous source span containing **all three**:

1. the **asserted numeral** (in any of its surface forms — `1.2e-3`, `0.0012`, `1.2×10⁻³`);
2. the **unit token** (`source_unit_token`); AND
3. the **quantity referent** (`referent` — the name/symbol of what the number measures);

with **≥ M surrounding word-tokens** (M = 6) on the side of the span that carries the referent (so the referent is in a real sentence, not a table cell stripped of meaning). The test is not "is it long enough" — it is **"does this span, alone, assert *this quantity has this value*."** A quote that is the bare number, or number+unit without the referent, is **rejected → `unconfirmed`**. This same referent is what §5's conditions-compatibility checks compatibility *of* — the rule does double duty.

---

## 7. The unit scale-table (code-owned conversion — G-D3)

`SCALE_TABLE` maps a unit token → `(dimension, factor_to_canonical)`. v1 seeds the handful of pairs condensed-matter magnitude claims actually use:

| dimension | units (token → factor to canonical) |
|---|---|
| energy | `J`→1, `eV`→1.602e-19, `meV`→1.602e-22, `K`(k_B)→1.381e-23, `cm^-1`→1.986e-23 |
| magnetic field | `T`→1, `Gauss`/`G`→1e-4, `mT`→1e-3 |
| magnetic moment | `µB`/`mu_B`/`bohr magneton`→1, `J/T`→1.0785e23 (÷µB in SI) |
| (spectral noise) | `V/√Hz`, `T/√Hz`, `Φ0/√Hz` → identity within a dimension (no cross-conversion v1) |

- **Code owns the conversion**: `convert(value, from_token, to_token) = value × factor[from] / factor[to]`, only when `dim[from] == dim[to]`. The LLM never converts.
- **Out-of-table unit → `unconfirmed`** (fail-closed, the honest default). A missing pair is a one-line table add when it shows up — "how many pairs" is the deferred part, "code owns the conversion" is non-negotiable (G-D3).
- Same-dimension cross-conversions (e.g. `meV`↔`K`) are exact constants; the table is the single source of truth so the converted number is reproducible.

---

## 8. Gate integration (adjudicate, never replace)

Grounding runs **after** `_run_magnitude` produces its `ComputationVerdict`, **before** `verdict_to_check`/`verdict_to_attack`, and modulates them. The magnitude arithmetic and the asserted value are **never changed**.

- **`supports`** → attach **one** real `Source(locator=source, relation="independent", title/url/year from the resolver)` to the magnitude `CheckRecord`, and set `independent_sources = 1` (G-D6: one grounded value → exactly one source; no inflation). The PASS now counts toward the gate's `internally_validated` path — *the discount lifts, the value is showable evidence.* The `basis` carries the verbatim quote + converted value loudly.
- **`unconfirmed`** → **no change** from today: the magnitude verdict stands, `independent_sources = 0`, the PASS stays discounted. Fail-closed; grounding neither helps nor hurts.
- **`contradicts`** → the verdict's *input* is literature-contradicted, so the verdict is unreliable → **downgrade the `ComputationVerdict` to `uncertain`** (non-decisive — grounding never manufactures a refutation/attack from a wrong *input* number; that would conflate "your input is wrong" with "your idea is refuted" — G-D6), with a **loud** note: `"bound contradicted by arXiv:2104.01234: source reports 1.2e-2 µB (quote: '…'), asserted 1e-3 µB"`.

The gate (`artifact.py`), `_evaluate`, and `verdict_class` are **untouched** — grounding only changes which `CheckRecord` carries a real source and whether a verdict is `uncertain`. The gate already keys on `independent_sources >= 1`; grounding makes that count *mean* something.

---

## 9. Designer change & data model

- `ComputationPlan` gains one field: **`source_quantity: str`** — a short descriptor of what the (single, per-kind) sourced value physically measures + its conditions (e.g., `"Yb³⁺ effective magnetic moment, T < 1 K"`). Used by the extraction agent (what to look for) and the conditions-compatibility gate (§5b). Optional; absent → `unconfirmed` (can't check conditions). *(One sourced value per `comparison_kind`, so one descriptor suffices — G-D2.)*
- `MAGNITUDE_DESIGNER` prompt: each `*_source` must be a **resolvable locator** (arXiv ID / DOI / URL), and the designer must emit `source_quantity`. The existing "never invent a threshold/sensitivity/bound without naming its SOURCE" tightens to "…without naming a **resolvable** source and the **quantity** it reports."
- The asserted **unit** for each value: magnitude values are already strings (e.g. `"1e-3"`). **Decision:** add `source_unit: str` (the asserted unit, e.g. `"µB"`) — explicit, feeds the conversion target. (**Two** new plan fields total: `source_quantity`, `source_unit`; the locator reuses the existing `*_source` field.)

---

## 10. Fail-closed, config, determinism & reproducibility

- **Config (`GroundCfg`):** `backend` (existing) gates whether grounding runs — `backend == "none"` → grounding skipped → every value `unconfirmed` → **exactly today's behavior** (the safe default). Add `ground_rtol: float = 0.5` and `quote_min_tokens: int = 6`.
- **Fail-closed everywhere:** no locator, unresolvable, fetch timeout/error, empty text, `not_found`, out-of-table unit, non-substantial quote, unparseable number, missing `source_quantity` → **`unconfirmed`** (never `supports`). The only paths to `supports` are all-gates-pass; the only path to `contradicts` is all-gates-pass-but-number-differs.
- **Determinism boundary:** the adjudicator is pure/deterministic, but the **fetch + extraction are not** (network, LLM). So grounding runs in the agent layer and its **result is recorded into the artifact**, not re-derived at gate time — the fetched-text hash, the verbatim quote, the converted value, and the locator are **snapshotted into the `CheckRecord`/`Source`** so the verdict is reproducible *and* showable (the quote travels with the artifact; a reader can re-open the locator and find the quote). The gate reads the recorded result; it never re-fetches.
- **Network etiquette:** grounding is opt-in via backend; per-value fetch is capped (timeout, text-length budget); failures degrade to `unconfirmed`, never raise into the pipeline.

---

## 11. Testing

The fetch is injected (a `FakeResolver` returning fixture source text), the extraction is a `FakeLLM`, and the **adjudicator is pure code** — the honesty-critical path is fully deterministic and unit-tested in isolation.

- **supports:** asserted `1.2e-3 µB`, `source_quantity="ordered moment per Yb ion, low-T"`; fixture text contains `"…the ordered moment per Yb ion saturates at 1.2e-3 µB below 1 K…"`. The quote is substantial (numeral + `µB` + referent + ≥6 tokens), the unit is in the table, the referent overlaps `source_quantity`, and the numbers match within `ground_rtol` → `supports` + exactly one `Source`, `independent_sources == 1`, the quote appears in the basis.
- **Code-owned conversion:** source reports `12 meV`, asserted `139 K` (k_B); code converts `12 meV → 139.2 K`, within `ground_rtol` → `supports`. A FakeLLM that "pre-converted" is irrelevant — code ignores any LLM conversion and uses the table.
- **Fabricated quote → unconfirmed:** extraction returns a quote NOT in `fetched_text` → quote-in-bytes fails → `unconfirmed`.
- **Degenerate quote → unconfirmed:** quote is the bare number `"1.2e-3"` (no unit, no referent) → substantiality fails → `unconfirmed`.
- **Hallucinated locator → unconfirmed:** `FakeResolver` returns *different* paper text not containing the quote → `unconfirmed`.
- **Wrong-conditions probe (from day one):** number+unit match but the source's `referent` is a *different quantity* (e.g. `source_quantity="magnetic moment µB"`, source reports `1.2e-3 T` magnetic *field*) → dimensional mismatch (`µB` vs `T`) **or** referent non-overlap → `unconfirmed`, NOT `supports`. The core conditions-compatibility test.
- **contradicts → uncertain + loud:** source reports `1.2e-2 µB` for a compatible quantity, asserted `1e-3 µB`, ratio 12× > `ground_rtol` → `contradicts` → the `ComputationVerdict` downgrades to `uncertain`, the note names the source's number and the quote. Verify it is NOT an attack.
- **Out-of-table unit → unconfirmed:** source unit `"emu/mol"` not in `SCALE_TABLE` → `unconfirmed`.
- **backend=none → no regression:** every existing magnitude test passes unchanged; grounding is skipped; PASS stays discounted exactly as today.
- **Designer:** a freeform (non-locator) source → plan still builds, value grounds to `unconfirmed` (backward-compat); a resolvable locator + `source_quantity` → grounding runs.
- **Gate purity:** `artifact.py`/`_evaluate`/`verdict_class` untouched; the magnitude teeth/anti-laundering pins stay green; a `supports` sets exactly one `independent_sources`.

---

## 12. The loud residual (tightened — stated honestly per the cardinal rule)

What code now **witnesses** (no longer residual): the asserted number appears in a real, resolvable source, in a verbatim substantial referent-binding quote, in a dimensionally-compatible unit, after a **code-owned** conversion, with referent token-overlap to the claim's quantity. The unit conversion — the transform most likely to be silently wrong — is **code's, not the model's** (the §1 firewall held through the conversion, the whole reason option-1 was chosen over LLM-judges-support).

What stays **loud, not gated** (the honest remaining mile): "this source *endorses* using this number as the bound/sensitivity for *THIS* measurement under *fully* compatible conditions" is semantic and not safely code-adjudicable without a second model. So the **verbatim quote + the claim's `source_quantity`** are surfaced loudly side-by-side in the `CheckRecord.basis`; a human reader adjudicates the last mile. The gate stays model-free. v1.x can tighten referent-overlap toward a stronger quantity-name correspondence (the Tier-2 graded object).

---

## 13. Decision log
- **G-D1** Tier-1 scope = the three magnitude `*_source` numeric inputs only; symbolic `expected_source` and grounder `[A1]` support-relation deferred (different ontology — a graded-correspondence object).
- **G-D2** One sourced value per `comparison_kind` (`sensitivity`|`bound`|`closest_prior`), so one `source_quantity` descriptor and one grounded `Source` per check.
- **G-D3 (the firewall — code owns the unit conversion)** The LLM extracts the number **in the source's own units, unit token quoted verbatim**; **code** converts via a closed `SCALE_TABLE` and then checks the match. v1-light (LLM converts) was rejected: unit conversion is the one transform most likely to be silently wrong, and the loud quote can't catch it (quote is in source units, asserted value in claim units) — letting the LLM convert is `LLM-judges-support` smuggled into the verdict path. "Code owns the conversion" is non-negotiable; "how many unit pairs" is deferred (seed ~6, out-of-table → `unconfirmed`).
- **G-D4** The model reads, code judges (F1/F3 for reading). Every LLM assertion (quote, number, unit, referent) is code-checked against the fetched bytes; the asserted *value* is withheld from the extraction agent (anti-anchoring), so `contradicts` is a real disagreement.
- **G-D5 (conditions-compatibility, both sides)** A number+unit match is insufficient — the source's number must be for a *compatible quantity*. Code checks (a) **dimensional** compatibility (from the table) AND (b) **referent token-overlap** with the claim's `source_quantity` (a necessary gate against same-dimension-different-quantity). The referent comes from the substantial quote (§6), so the quote rule does double duty. Full semantic compatibility is the surfaced-loud residual. **Wrong-conditions probe is a day-one test.**
- **G-D6 (adjudicate, never replace; supports→one source; contradicts non-decisive)** Grounding modulates gate treatment, never re-runs the arithmetic or swaps the value. `supports` → exactly one independent `Source` (no inflation), discount lifts. `unconfirmed` → unchanged, discounted (fail-closed). `contradicts` → verdict downgraded to `uncertain` and surfaced **loud**, never an attack (a wrong *input* number ≠ a refuted *idea*).
- **G-D7 (`ground_rtol = 0.5`, asymmetric)** Magnitude claims are order-of-magnitude; a factor-of-2 agreement is `supports`, a >2× disagreement on a compatible quantity is `contradicts`. The harmful direction is a false `supports` (laundering a wrong value into "validated") — guarded by the quote+unit+referent+dimensional gates *before* the numeric step. A false `contradicts` (down-grading a fine value) is conservative (→ uncertain). Quote substantiality (`quote_min_tokens = 6`) is the actual strength of the anti-fabrication check, not a cosmetic min-length.
- **G-D8** Determinism boundary: fetch+extraction are non-deterministic (agent layer); the adjudicator is pure; the **result is snapshotted into the artifact** (quote, converted value, locator, text-hash) and read by the gate, never re-fetched — reproducible *and* showable.

---

## 14. Build slices (order)
1. **The adjudicator + scale-table + quote rule (pure code, no network/LLM).** `valagents/grounding.py`: `SCALE_TABLE` + `convert`; `_quote_valid` (substantiality/referent-binding, §6); `_conditions_compatible` (§5.3); `ground_value(...) -> GroundingResult` (§5). Exhaustive unit tests incl. every probe in §11 that needs no network (supports, conversion, fabricated/degenerate quote, wrong-conditions, contradicts, out-of-table). The honesty core, fully deterministic.
2. **Fetch + extraction agent.** Extend `references.py` resolvers with a text fetch (`fetch_source_text(locator) -> (text, meta) | None`); add `valagents/agents/value_grounder.py` + `VALUE_GROUNDER` prompt (structured extraction, anti-anchoring). Tests with a `FakeResolver` + `FakeLLM`.
3. **Data model + designer.** `ComputationPlan` gains `source_quantity`, `source_unit`; `*_source` documented as a resolvable locator; `MAGNITUDE_DESIGNER` updated; `GroundCfg` gains `ground_rtol`, `quote_min_tokens`. Backward-compat tests (freeform source → unconfirmed).
4. **Gate integration.** Wire grounding between `_run_magnitude`'s verdict and `verdict_to_check`/`verdict_to_attack`: `supports` → one Source + `independent_sources=1` + loud basis; `unconfirmed` → unchanged; `contradicts` → `uncertain` + loud note. Snapshot the result into the `CheckRecord` (G-D8). End-to-end gate tests (discount lifts on supports; no-regression on backend=none; gate purity pins green).
