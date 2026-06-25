# validate-agents — Spec 3 Grounding Design (Tier 1: magnitude sourced values)

- **Date:** 2026-06-24
- **Status:** Design rev 3 (adversarial-verification pass — field-asymmetry + emu-denominator false-supports closed, conditions ladders, factor verification), pending user re-review then implementation plan
- **Builds on:** Spec 2 (the magnitude lens — `ComputationPlan` `*_source` fields, `_run_magnitude`, `verdict_to_check`/`verdict_to_attack`), the existing retrieval layer (`web_search.py` backends, `references.py` resolvers), and the gate's `independent_sources` accounting.
- **Status line:** *Grounding closes the one axis still resting on LLM say-so — the asserted numeric inputs to the magnitude verdict. A value counts as grounded only when an independent, code-adjudicated check confirms its named source reports it for a compatible quantity under compatible conditions: the model reads (a value + its quantity + its conditions + a verbatim quote), code judges (quote ∈ fetched bytes ∧ code-owned unit conversion ∧ numeric match ∧ quantity-overlap ∧ conditions-compatible). The model never says "yes, grounded"; the conversion that decides the match, and the conditions that decide relevance, are never the model's word.*
- **One-line goal:** Verify the magnitude lens's three asserted `*_source` numeric inputs (`sensitivity`/`bound`/`closest_prior`) against their named, resolvable sources; a fully-confirmed value (number + quantity + **conditions** + verbatim quote) lifts the discount on that magnitude PASS, a showable-but-not-fully-confirmed value travels its quote in the basis without clearing the gate, an unconfirmable one stays discounted (fail-closed), and a same-regime literature contradiction invalidates the verdict (uncertain, loud).

---

## 1. Scope

### In scope (Tier 1 — locked)
- Ground the **three magnitude `*_source` numeric inputs**, one per `comparison_kind`: `sensitivity` (`sensitivity_source`), `bound` (`bound_source`), `closest_prior_effect` (`closest_prior_source`). These are the asserted numbers that feed the magnitude *arithmetic*, so "the number IS the verdict input" holds.
- A resolvable-locator → fetch-full-text → LLM-extract → **code-adjudicate** pipeline producing one of **four** outcomes: `supports` / `contradicts` / `inconclusive` (showable, non-gating) / `unconfirmed` (nothing to show).
- A **code-owned unit conversion** over a small closed scale-table (the conversion that decides the numeric match is never the LLM's — G-D3), with **every entry adversarially factor-verified** (an unverified entry is out-of-table — G-D9).
- A **two-axis relevance check** the slice exists for: the source's number must be for a compatible **quantity** (referent overlap — G-D5a) AND under compatible **conditions** (numeric conditions predicate — G-D5b). The wrong-conditions false-support is the primary threat and the day-one probe.
- Gate modulation: `supports` → one real independent `Source`, `independent_sources = 1` (discount lifts); `inconclusive` → quote in the basis (showable) but `independent_sources = 0`; `unconfirmed` → unchanged/discounted; `contradicts` → verdict → `uncertain`, loud (G-D6).

### Out of scope (deferred / never)
- **Symbolic `expected_source`** and the **grounder `[A1]` support-relation** (Tier 2 — graded-correspondence object, different ontology).
- **Full dimensional analysis** and **`emu·mol⁻¹` molar susceptibility** (the 4π / per-mole / Oe-vs-A·m⁻¹ tangle) — deferred to a factor-verified follow-up; out-of-table → `unconfirmed` for v1 (G-D9).
- **Semantic conditions matching** beyond the numeric subset, and **symbol↔prose quantity matching** (`µ_eff` ↔ "effective moment") — surfaced **loud**, not gated (§12); never handed to a second model.
- **Replacing** an asserted value / re-running the arithmetic with a grounded number. Grounding **adjudicates, never replaces** (G-D6).

---

## 2. The honesty mechanism (F1/F3 applied to *reading*)

The cardinal rule — *validated = survived an independent, code-adjudicated check, never the model's say-so* — applied one level up. An LLM that "reads the PDF and says it's grounded" relocates the say-so. So:

- **The LLM only reads** (F1 analogue): given the fetched source text and the **quantity to find** (and its asserted unit), it returns a *structured extraction* of what the source primarily reports for that quantity — `{extracted_value, source_unit_token, referent, source_conditions, verbatim_quote}` or `not_found`. It reports the number **in the source's own units**, and the source's own stated **conditions**, with a verbatim quote. It is **never** asked "does this support the claim?", and is shown **neither the asserted value nor the claim's conditions** (anti-anchoring — G-D4), so it cannot cherry-pick a value/regime to match.
- **Code judges** (F3): every assertion (quote, number, unit, referent, conditions) is checked against the fetched **bytes** and **code-owned** tables. The LLM cannot produce a passing result it didn't read out of the actual source, and cannot make the numbers match by converting units or by selecting a convenient regime.

Pipeline per sourced value:

```
ComputationPlan {value=1e-3, source_unit="µB", source="arXiv:2104.01234",
                 source_quantity="Yb³⁺ effective moment", claim_conditions="T < 1 K"}
        │
1. resolve locator (arXiv/DOI/URL) ──references.py──> fetch FULL text (PDF→text; abstract fallback)   [agent layer, network]
        │   (no locator / unreachable / wrong paper → fail-closed → unconfirmed)
2. extraction agent (LLM, reads only; NOT shown value or claim_conditions):
        (text, source_quantity, source_unit) → {extracted_value, source_unit_token, referent, source_conditions, verbatim_quote} | not_found
        │
3. adjudicator (PURE CODE, no LLM):
     quote ∈ normalize(text)?                         (anti-fabrication; substantial, referent-binding — §6)
     source_unit_token in SCALE_TABLE (factor-verified)?   (else unconfirmed — §7)
     referent overlaps source_quantity?               (quantity gate — §5)
     convert extracted_value → source_unit (CODE owns this — §7)
     conditions_compatible(claim_conditions, source_conditions)?   (numeric subset — §5)
     ratio = max(converted/asserted, asserted/converted)
        → supports | contradicts | inconclusive | unconfirmed   (§5 decision tree)
        │
4. modulate verdict_to_check / verdict_to_attack, snapshot into the CheckRecord   (§8, G-D8)
```

---

## 3. Source identification & fetch

- The magnitude designer emits, per sourced value, a **resolvable locator** — arXiv ID / DOI / URL — in the `*_source` field. A string that does not parse as one of these → **ungrounded** → `unconfirmed` (the value is still *used* by the executor as today; it earns no grounding credit). Backward-compatible: existing freeform sources degrade to `unconfirmed`, no regression.
- Resolution reuses `references.py` (`ArxivResolver`, `DoiResolver`) extended with a **text fetch**: arXiv → fetch the PDF and extract a text layer; DOI → publisher/crossref abstract; URL → fetch HTML→text. **Realism (the showability win is rarer than it reads):** magnitude numbers usually live in a *table or figure*, not the abstract, so **full-text** is what makes grounding work at all, and even then `inconclusive`/`unconfirmed` (number not in the text layer, conditions not stated near it) is the *common* outcome. That is acceptable (fail-closed), but the design does not promise that most real papers will reach `supports`.
- **Hallucinated / wrong locators fail closed automatically** — they resolve to nothing or to a different paper whose bytes won't contain the asserted quote+number. The quote-in-bytes check (§5) is the backstop; the locator is never trusted.
- Network lives in the **agent layer only**; the sandbox stays network-isolated.

---

## 4. The extraction agent (anti-anchoring on value AND conditions)

A new small agent (`valagents/agents/value_grounder.py`), prompt `VALUE_GROUNDER`. Structured output, one call per sourced value.

- **Shown:** the fetched source text (truncated to a budget), the **quantity to find** (`source_quantity`), and the **asserted unit** (`source_unit`, so it knows what kind of quantity — NOT for conversion). **Not shown: the asserted value, and the claim's conditions** (`claim_conditions`). Withholding both closes the anchoring channels — the model cannot tune the extracted number to the target, nor cherry-pick a regime that matches.
- **Output:** `{extracted_value, source_unit_token, referent, source_conditions, verbatim_quote}` or `not_found`.
  - `extracted_value` — the number the source **primarily** reports for that quantity, **in the source's own units** (report the paper's main/headline value, not "the one matching some target").
  - `source_unit_token` — the **full compound** unit as written in the source, verbatim, **denominator included** (e.g. `"Gauss"`, `"meV"`, `"emu/g"` — NOT a bare `"emu"` stripped from `"emu/g"`; stripping a per-mass/volume/mole denominator is the F2 false-support).
  - `referent` — the source's name/symbol for the quantity (e.g. `"ordered moment per Yb ion"`), as it appears in the quote.
  - `source_conditions` — the source's stated conditions for that value (e.g. `"T = 0.4 K, B = 0"`), as it appears in the quote.
  - `verbatim_quote` — a contiguous source span containing the numeral, the unit token, and the referent (§6); the conditions appear in the span when the source states them near the value.
- The agent is instructed: **do not convert units, do not infer, quote verbatim, report the source's primary value, return `not_found` if the quantity is absent.** `not_found` → `unconfirmed`.

---

## 5. The adjudicator (pure code — the honesty core, F3)

`ground_value(asserted_value, source_unit, source_quantity, claim_conditions, extraction, fetched_text, cfg) -> GroundingResult`, where `GroundingResult.status ∈ {supports, contradicts, inconclusive, unconfirmed}` plus the loud fields (quote, converted value, referent, source_conditions, reason). Deterministic; no LLM.

**Gates (any failure → `unconfirmed` — nothing showable):**
1. **Quote-in-bytes (anti-fabrication).** `normalize(quote)` is a literal substring of `normalize(fetched_text)` (collapse whitespace, NFKC, casefold).
2. **Quote substantiality / referent-binding (§6).** The quote asserts *this quantity has this value*, not merely contains a number.
3. **Unit in (factor-verified) table.** `source_unit_token` resolves in `SCALE_TABLE` to the **same physical dimension** as `source_unit`.
4. **Quantity gate (G-D5a).** `referent` shares ≥1 content token (after stop-word removal) with `source_quantity` — a *necessary* floor against same-dimension-different-quantity (a moment vs a field). *(Symbol↔prose misses err to `unconfirmed` — safe; §12.)*

**Once all gates pass, the value is SHOWABLE.** Now classify with the conditions axis and the numeric axis:

5. **Conditions predicate (G-D5b, code, numeric subset).** `conditions_compatible(claim_conditions, source_conditions)`: parse each into `{quantity: (op, value, unit)}` clauses for the quantities v1 handles (temperature, field). The predicate uses **its own quantity-aware unit ladders and MUST NOT call `SCALE_TABLE.convert`** (whose `K` is *energy*-via-k_B and whose `T` is *Tesla*) — temperature ladder `K:1, mK:1e-3, µK:1e-6`; field ladder `T:1, mT:1e-3, G`/`Gauss`:1e-4, Oe:1e-4` — disambiguating the *quantity symbol* from the *unit* (a clause `T < 1 K` is quantity=temperature, unit=K; a clause `B = 0.5 T` is quantity=field, unit=Tesla).
   - A claim clause `T < 1 K` is satisfied by a source point/range that lies within it (`0.4 K` ✓, `300 K` ✗). An `=` claim clause (e.g. `T = 0.3 K`) is satisfied **only** by a source point that matches it exactly after canonical conversion (**no relative tolerance** — a conditions tolerance would reopen the wrong-conditions false-support of G-D5b/G-D7); a source range satisfies an `=` claim only if the claim point lies within it.
   - **Symmetry (G-D5c — the field hole):** a parsed **source** clause on a v1-handled axis the **claim does not constrain**, with a **non-zero** value (unit-aware: `B=0`/`0 mT`/`0 Oe` count as baseline/absent), also **fails confirmation** — the source regime is more specific than the claim (claim silent on field, source `B = 5 T` → NOT confirmed; a moment in 5 T is a field-polarized, physically different regime). A zero or absent source value on that axis stays permissive. *(Field is where this bites — `B=0` is the genuine no-applied-field baseline a field-silent claim is compatible with; temperature has no zero-baseline, so any finite source T on a T-silent claim caps to `inconclusive`.)*
   - **Unparseable / absent / not-overlapping / claim-silent-axis-with-nonzero-source → conditions NOT confirmed.**
6. **Numeric zone.** `ratio = max(converted/asserted, asserted/converted)` after the **code-owned** conversion (§7).

**Decision:**
- `supports` ⟺ conditions **confirmed compatible** AND `ratio < supports_factor` (2). → `independent_sources = 1`, clears the bar.
- `contradicts` ⟺ conditions **confirmed compatible** AND `ratio ≥ contradict_factor` (10). → verdict `uncertain`, loud. *(Requires compatible conditions: a gross disagreement under the SAME regime. A far-off value at an unconfirmed/incompatible regime is irrelevant, not a contradiction.)*
- `inconclusive` ⟺ all gates passed but NOT (`supports` or `contradicts`): conditions unconfirmed, OR `ratio ∈ [2, 10)` (numeric can't-tell). → **showable** (quote + converted value + source_conditions in the basis), `independent_sources = 0`. The `reason` records `conditions_unconfirmed` vs `numeric_inconclusive`.
- `unconfirmed` ⟺ any gate (1–4) failed. → nothing shown.

---

## 6. Quote substantiality / referent-binding rule

A bare-number quote defeats the substring predicate without failing it. The quote is **valid** iff it is a contiguous source span containing **all three**:
1. the **asserted numeral** (any surface form — `1.2e-3`, `0.0012`, `1.2×10⁻³`);
2. the **full compound unit token** (`source_unit_token`, matched as a *whole token* — `emu/g`, not a bare `emu` stripped from it); AND
3. the **quantity referent** (`referent`);

with **≥ `quote_min_tokens` (6) surrounding word-tokens** on the side carrying the referent (so the referent sits in a real sentence, not a stripped table cell). The test is **"does this span, alone, assert *this quantity has this value*."** Bare number, or number+unit without the referent → **rejected → `unconfirmed`**. The same referent feeds §5's quantity gate; the rule does double duty. *(Conditions are not required for quote validity — a value can be showable/`inconclusive` without the source stating its conditions near the value; conditions only gate `supports`/`contradicts`.)*

---

## 7. The unit scale-table (code-owned, factor-verified — G-D3, G-D9)

`SCALE_TABLE` maps a unit token → `(dimension, factor_to_canonical)`. **Code** converts: `convert(v, from, to) = v × factor[from] / factor[to]`, only when `dim[from] == dim[to]`. The LLM never converts. **The in-table test is a WHOLE-TOKEN (exact key) match — never a substring/prefix.** Any token that is not a bare exact key (anything carrying an operator `/` or `·`, e.g. `emu/g`, `emu/cm³`, `emu·mol⁻¹`, `emu/Oe`) is **out-of-table → `unconfirmed`** — this is what stops a bare `emu` (moment) from matching inside `emu/g` (mass magnetization) and laundering a per-gram value as an absolute moment (the F2 false-support).

| dimension | tokens → factor (to canonical) |
|---|---|
| energy | `J`→1, `eV`→1.602177e-19, `meV`→1.602177e-22, `K`(k_B)→1.380649e-23, `cm^-1`→1.986446e-23 |
| magnetic field | `T`→1, `mT`→1e-3, `Gauss`/`G`→1e-4, `Oe`→1e-4 *(B-equivalence in vacuum; loud caveat — §12)* |
| magnetic moment | `µB`/`mu_B`→1, `emu`→1.078e20 *(bare emu only = absolute moment)*, `J/T`→1.078e23 |
| magnetic flux | `Φ0`/`Phi0`→1, `mΦ0`→1e-3, `µΦ0`→1e-6 *(identity within dimension only; NO cross to field — that needs device area)* |

- **Out-of-table unit → `unconfirmed`** (fail-closed). This includes **any `emu` with a denominator** (`emu/g` mass magnetization, `emu/cm³` volume magnetization/susceptibility, `emu·mol⁻¹` molar susceptibility, `emu/Oe`) — none is a moment; only **bare `emu`** = moment. A missing pair is a one-line add — but only with its verification (next bullet).
- **G-D9 (factor verification is mandatory, with a named tolerance):** a wrong factor is a **code-owned, reproducible, confidently-wrong** conversion — *strictly worse* than the LLM-converts case G-D3 rejects, because it launders systematically and silently. So **every entry carries a both-directions adversarial test against a known reference value** — the test passes iff `max(computed/ref, ref/computed) − 1 < reference_rel_tol` (default **`1e-3`**, §10), both ways. References: `1 meV ↔ 11.6045 K`, `1 cm⁻¹ ↔ 1.4388 K`, `1 Oe ↔ 1e-4 T`, `1 emu ↔ 1.078e20 µB`. An entry without a passing reference test is treated as **out-of-table** (`unconfirmed`). *(The energy row must carry full-precision factors above — at 4 sig figs `1 meV` computes to `11.600 K`, a `3.2e-4` miss that fails a tight tolerance and would silently render every energy grounding `unconfirmed`; `reference_rel_tol=1e-3` clears full-precision by ~3 orders yet still catches a real factor slip. Do NOT use `1e-4` — no headroom for future hand-added entries, the same fail-closed footgun.)* The CGS↔SI entries (`Oe`, `emu`) and especially **`emu·mol⁻¹` susceptibility (deferred)** are where a silent factor error does the most damage; molar susceptibility stays out-of-table until its factor is reference-verified.

---

## 8. Gate integration (adjudicate, never replace)

Grounding runs **after** `_run_magnitude`'s `ComputationVerdict`, **before** `verdict_to_check`/`verdict_to_attack`, and modulates them. The arithmetic and the asserted value are **never changed**.

- **`supports`** → attach **one** real `Source(locator=source, relation="independent", title/url/year from the resolver)` and set `independent_sources = 1` (G-D6: one grounded value → exactly one source; no inflation). The PASS now counts toward `internally_validated` — *the discount lifts.* The `basis` carries the verbatim quote + converted value + source_conditions loudly.
- **`inconclusive`** → the quote + converted value + source_conditions + `reason` go into the `basis` (**showable** — a reader can open the locator and check the regime), but `independent_sources = 0` (does **not** clear the bar). The verdict is unchanged.
- **`unconfirmed`** → unchanged from today: verdict stands, `independent_sources = 0`, PASS stays discounted, nothing shown.
- **`contradicts`** → the verdict's *input* is literature-contradicted under a compatible regime → **downgrade the `ComputationVerdict` to `uncertain`** (non-decisive — never an attack; a wrong *input* number ≠ a refuted *idea*), with a **loud** note naming the source's number, units, conditions, and the quote.

**`internally_validated` threshold (G-D10, explicit decision):** the gate's existing rule — a root claim needs **≥1** `pass` check with `independent_sources ≥ 1` — is **left UNCHANGED** (`artifact.py` untouched). A single *fully-confirmed* grounding clears the per-claim bar, because that bar now demands a **real, code-adjudicated, conditions-checked** source (a strict upgrade over the prior LLM-asserted `≥1`). **Raising to ≥2 is not merely deferred — it is *unsound now*.** This magnitude-grounding lens is the **only** lens that code-adjudicates that a retrieved source's value actually *supports* the claim under compatible conditions; the others write `independent_sources ≥ 1` from say-so or self-contained proof (the grounder `[A1]` path sets `relation="independent"` from an LLM citation capped only by URL existence — never code-checking support; the executor's `verdict_to_check` sets it from an LLM-designed plan locator; the prover from a self-contained proof). Since this lens sets exactly `independent_sources = 1` (G-D6, no inflation), a ≥2 bar would be clearable **only by adding a co-signer from a non-support-adjudicated producer** — re-admitting the exact say-so the project excludes (cardinal rule). **Revisit trigger:** raise to ≥2 only once a **second lens code-adjudicates external *support*** for the same check (not merely writes the count), and even then ≥2 alone doesn't stop shared-blind-spot laundering (two sources both clearing the token-overlap quantity gate via `µ_eff`↔"effective moment"), which additionally needs the symbol-aware quantity axis (the §12 Tier-2 residual). Until then, ≥1 of a real conditions-checked source is the honest bar. *(So flipping ≥2 on now is a regression, not a benign tightening.)*

The gate (`artifact.py`), `_evaluate`, and `verdict_class` are **untouched**.

---

## 9. Designer change & data model

- `ComputationPlan` gains **three** fields:
  - **`source_quantity: str`** — what the sourced value physically measures (the referent target), e.g. `"Yb³⁺ effective magnetic moment"`. Absent → `unconfirmed` (can't run the quantity gate).
  - **`claim_conditions: str`** — the claim's regime for that value, e.g. `"T < 1 K"`. Absent → conditions cannot be confirmed → at most `inconclusive`.
  - **`source_unit: str`** — the asserted unit (the conversion target), e.g. `"µB"`.
- `MAGNITUDE_DESIGNER` prompt: each `*_source` must be a **resolvable locator** (arXiv ID / DOI / URL); the designer must emit `source_quantity`, `claim_conditions`, `source_unit`. "Never invent a threshold/sensitivity/bound without naming its SOURCE" tightens to "…without naming a **resolvable** source, the **quantity** it reports, the **conditions**, and the **unit**."
- Backward-compat: a freeform (non-locator) source or a missing new field → the value grounds to `unconfirmed`/`inconclusive`; the plan still builds and the magnitude check still runs exactly as today.

---

## 10. Fail-closed, config, determinism & reproducibility

- **Config (`GroundCfg`):** `backend` (existing) gates whether grounding runs — `"none"` → skipped → every value `unconfirmed` → **exactly today's behavior**. Add `supports_factor: float = 2.0`, `contradict_factor: float = 10.0`, `quote_min_tokens: int = 6`, `reference_rel_tol: float = 1e-3` (the G-D9 scale-table both-directions test tolerance — §7).
- **Fail-closed everywhere:** no locator, unresolvable, fetch timeout/error, empty text, `not_found`, out-of-table or unverified unit, non-substantial quote, failed quantity gate, unparseable number → **`unconfirmed`**. Conditions unparseable/absent/incompatible, or `ratio ∈ [2,10)` → **`inconclusive`** (showable, non-gating). The only path to `supports` is all gates pass **and** conditions confirmed **and** `ratio < 2`.
- **Determinism boundary (G-D8):** fetch + extraction are non-deterministic (network, LLM); the adjudicator is pure. Grounding's **result is snapshotted into the artifact** — the fetched-text hash, the verbatim quote, the converted value, the source_conditions, the locator, the status+reason are recorded in the `CheckRecord`/`Source` and **read by the gate, never re-fetched**. The verdict is reproducible and the quote travels with the artifact for showability.
- **Network etiquette:** opt-in via backend; per-value fetch capped (timeout, text-length budget); failures degrade to `unconfirmed`, never raise into the pipeline.

---

## 11. Testing

The fetch is injected (a `FakeResolver` returning fixture source text), the extraction is a `FakeLLM`, and the **adjudicator is pure code** — the honesty-critical path is fully deterministic and unit-tested in isolation.

- **supports:** asserted `1.2e-3 µB`, `source_quantity="ordered moment per Yb ion"`, `claim_conditions="T < 1 K"`; fixture text `"…the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K…"`. Quote substantial, unit in table, referent overlaps, conditions compatible (0.4 K within T<1 K), ratio≈1 → `supports` + one `Source`, `independent_sources == 1`, quote in basis.
- **Wrong-conditions (THE day-one probe — the hard regime, same quantity/unit/number):** identical asserted value and quantity, but the source states `"…1.2e-3 µB at T = 300 K…"`. All gates pass, ratio≈1, but conditions **incompatible** (300 K ∉ T<1 K) → **`inconclusive`** (`reason=conditions_unconfirmed`), **NOT `supports`**; `independent_sources == 0`; quote still in basis (showable).
- **Claim-silent-axis, non-zero source field (G-D5c — the F1 field hole):** `claim_conditions="T < 1 K"` (silent on field), source `"…1.2e-3 µB at T = 0.4 K, B = 5 T…"`, same value/quantity/unit. T satisfied, but the source pins a **non-zero field the claim doesn't constrain** → conditions NOT confirmed → **`inconclusive`**, **never `supports`**. The same source with `B = 0` (or no field stated) → `supports` (baseline is permissive).
- **Sub-Kelvin ladder (F3):** `claim_conditions="T < 1 K"`, source `"…at 400 mK…"` → the conditions parser's **own** temperature ladder (`mK:1e-3`, NOT `SCALE_TABLE`'s energy-K) resolves `400 mK = 0.4 K` ∈ T<1 K → `supports`. A field-conditions probe: claim `B < 0.1 T`, source `5 mT` → compatible.
- **Conditions-absent → inconclusive:** source states the value with no nearby conditions → conditions unconfirmed → `inconclusive`, not `supports`.
- **Numeric can't-tell → inconclusive:** compatible conditions, source reports `4e-3 µB` vs asserted `1.2e-3` (ratio≈3.3 ∈ [2,10)) → `inconclusive` (`reason=numeric_inconclusive`), not `contradicts`.
- **contradicts (same regime, interior + boundary-pinned):** an INTERIOR fixture — compatible conditions, source `1.5e-1 µB` vs asserted `1.2e-3` (ratio≈125) → `contradicts` → `ComputationVerdict` → `uncertain` + loud note; verify it is NOT an attack. Plus exact-boundary pins constructed from clean inputs (not `1.2e-2/1.2e-3`, whose float ratio is `10.000…002`): `converted=10.0, asserted=1.0` (ratio `==10.0`) → `contradicts` (`≥10`); `converted=2.0, asserted=1.0` (ratio `==2.0`) → `inconclusive` (`<2` supports is false, `2.0 ∈ [2,10)`). These pin the inequalities so a future `≥`→`>` / `<`→`≤` edit is caught.
- **No false-contradict from wrong regime:** source reports `1.2e-2 µB at 300 K` (ratio≈10 but conditions incompatible) → `inconclusive`, **never `contradicts`** (doesn't kill the verdict).
- **Code-owned conversion:** source `12 meV`, asserted `139 K`; code converts `12 meV → 139.25 K` (full-precision factors), ratio≈1 → (conditions permitting) `supports`. A FakeLLM that pre-converted is ignored.
- **Scale-table factor verification (G-D9):** a standalone test asserts **each** table entry both directions against its reference within `reference_rel_tol` (`1 meV ↔ 11.6045 K`, `1 cm⁻¹ ↔ 1.4388 K`, `1 Oe ↔ 1e-4 T`, `1 emu ↔ 1.078e20 µB`); a deliberately-wrong factor (or a 4-sig-fig energy factor against a `1e-4` tolerance) must fail this test (the guard that catches a silent factor error).
- **Fabricated quote / degenerate quote / hallucinated locator / out-of-table unit / failed quantity gate** → `unconfirmed` (each a probe).
- **backend=none → no regression:** every existing magnitude test passes unchanged; grounding skipped; PASS stays discounted exactly as today.
- **Gate purity:** `artifact.py`/`_evaluate`/`verdict_class` untouched; magnitude teeth/anti-laundering pins green; a `supports` sets exactly one `independent_sources`; `inconclusive`/`contradicts` set zero.

---

## 12. The loud residual (stated honestly)

**Code now witnesses** (no longer say-so): the asserted number appears in a real, resolvable source, in a substantial referent-binding verbatim quote, in a dimensionally-compatible **factor-verified** unit, after a **code-owned** conversion, for a quantity with referent token-overlap, under conditions whose **numeric subset** (temperature, field) is code-confirmed compatible. The two transforms most likely to be silently wrong — the **unit conversion** and the **regime relevance** — are code's, not the model's.

**Loud, not gated** (the honest remaining mile): (1) **semantic conditions** beyond the parsed numeric subset (e.g. "single-crystal vs powder", "applied vs remanent") — surfaced via the verbatim quote + `source_conditions`; (2) **symbol↔prose quantity** correspondence (`µ_eff` ↔ "effective moment") where token-overlap misses and errs to `unconfirmed` (safe but lossy); (3) whether the source *endorses* this exact use; (4) **field type `Oe`(H) vs `T`(B):** `Oe` measures the H-field but is tabled at the vacuum B-equivalence (`1 Oe ↔ 1e-4 T`, §7), so an `Oe`-vs-`T` comparison conflates H with B when source and claim mean different field types — **exact** when both mean the *applied* field, but **unbounded** inside a magnetized sample (`B = H + 4πM`, so `B/H = 1 + 4πχ` exceeds 10× for χ≳0.72; the "≤2×" intuition is false for high-susceptibility samples). The §5 quantity gate does not separate an H "field" referent from a B "field" referent. This stays a residual (not a gate) because the worst case is a false **`contradicts`** (loud, downgrades to `uncertain`, never an attack — the *safe* direction), **not** a false `supports`; it is surfaced via the verbatim `Oe` token in the basis quote for human adjudication. These travel **loudly** in the basis (quote + `source_quantity` + `claim_conditions` + `source_conditions` side-by-side); a human adjudicates them; the gate stays model-free. v1.x can strengthen the quantity axis toward a graded symbol-aware correspondence (the Tier-2 object) and add `emu·mol⁻¹` once its factor is reference-verified.

---

## 13. Decision log
- **G-D1** Tier-1 scope = the three magnitude `*_source` numeric inputs only; symbolic `expected_source` and grounder `[A1]` deferred (different ontology).
- **G-D2** One sourced value per `comparison_kind`, so one `source_quantity`/`claim_conditions`/`source_unit` and one grounded `Source` per check.
- **G-D3 (firewall — code owns the unit conversion)** The LLM extracts the number in the source's own units, unit token quoted verbatim; **code** converts via a closed `SCALE_TABLE`. v1-light (LLM converts) was rejected: unit conversion is the transform most likely to be silently wrong, and the loud quote can't catch it (quote in source units, asserted value in claim units). "Code owns the conversion" is non-negotiable; "how many pairs" is deferred.
- **G-D4 (the model reads, code judges; anti-anchoring on value AND conditions)** Every LLM assertion (quote, number, unit, referent, conditions) is code-checked against the fetched bytes. The extraction agent is shown **neither the asserted value nor the claim's conditions** and reports the source's **primary** value + its stated conditions, so it cannot tune the number or cherry-pick a regime — `contradicts` is a real disagreement and a `supports` regime-match is the source's, not the model's selection.
- **G-D5 (TWO-axis relevance — the slice's purpose; conditions, not just quantity)** A number+unit match is insufficient. **(a) Quantity gate:** `referent` token-overlaps `source_quantity` (necessary floor; symbol↔prose misses → `unconfirmed`, safe). **(b) Conditions gate:** a **code** predicate parses the numeric subset of `claim_conditions` vs the quote-backed `source_conditions` (temperature, field) using **its own quantity-aware unit ladders** (`K/mK/µK` for temperature, `T/mT/G/Oe` for field) and **MUST NOT call `SCALE_TABLE.convert`** (whose `K` is energy-via-k_B, `T` is Tesla — reusing it conflates a temperature `K` with an energy `K`, F3); requires the source regime to lie within the claim regime; an `=` claim clause needs an exact point match (no tolerance — a conditions tolerance reopens this hole). **(c) Symmetry (the F1 field hole):** a parsed **source** clause on a v1-handled axis the **claim does not constrain**, with a **non-zero** value (`B = 5 T` against a field-silent claim), also fails confirmation — the source regime is more specific (a field-polarized moment ≠ a zero-field moment); `B = 0`/absent stays permissive. *Rev-1 hole fixed:* the original predicate checked only the quantity *name* and would have let a 300 K source `supports` a `T<1 K` claim. *Rev-2 hole (F1, caught by adversarial verification) fixed:* the predicate iterated **claim** clauses only, so a non-zero **source** clause on a claim-silent axis (the canonical field-stated-source / field-silent-claim pairing) sailed to `supports` — the same wrong-regime laundering one axis over. Day-one probes: wrong-temperature, claim-silent-field-with-non-zero-source, sub-Kelvin `mK`. The easy different-*dimension* case is caught by the dimensional gate and is **not** the real probe.
- **G-D6 (four outcomes; showability split from gate-clearing; adjudicate never replace)** `supports` (conditions-confirmed AND ratio<2) → one `Source`, `independent_sources=1`, discount lifts. `inconclusive` (gates pass, but conditions unconfirmed OR ratio∈[2,10)) → quote travels in the basis (**showable**) but `independent_sources=0` — pure-code grounding tops out here for most real papers, which is the honest deliverable. `unconfirmed` (a gate failed) → nothing shown, discounted. `contradicts` (conditions-confirmed AND ratio≥10) → verdict `uncertain`, loud, **never an attack**. Grounding never re-runs the arithmetic or swaps the value.
- **G-D7 (three numeric zones; `supports_factor=2`, `contradict_factor=10`)** Magnitude claims are order-of-magnitude. `<2×` → supports-eligible, `[2×,10×)` → `inconclusive` (numeric no-info — avoids false-contradicting a fine verdict on normal literature scatter), `≥10×` → contradicts-eligible. The harmful direction is a false `supports`; it is guarded by **all of** the quote+unit+quantity+**conditions** gates *before* the numeric step (the conditions gate is what makes this asymmetry actually true — without it, the wrong-conditions false-support was unguarded).
- **G-D8** Determinism boundary: fetch+extraction non-deterministic (agent layer); adjudicator pure; the **result is snapshotted into the artifact** (quote, converted value, source_conditions, locator, text-hash, status+reason) and read by the gate, never re-fetched.
- **G-D9 (scale-table factors are adversarially verified or out-of-table; named tolerance; whole-token match)** A wrong factor is a code-owned, reproducible, confidently-wrong conversion — worse than LLM-converts. Every entry carries a both-directions reference test passing within **`reference_rel_tol` (1e-3)**; an unverified entry is out-of-table (`unconfirmed`). The energy row must use **full-precision** factors — at 4 sig figs `1 meV` computes to `11.600 K` vs the `11.6045 K` reference (a `3.2e-4` miss that fails a tight tolerance and would silently no-op every energy grounding; `1e-3` clears full-precision yet catches a real slip — `1e-4` is a fail-closed footgun). The in-table lookup is a **whole-token** match (F2): a token with a denominator (`emu/g`, `emu/cm³`, `emu·mol⁻¹`, `emu/Oe`) is out-of-table — only **bare `emu`** = moment — so a per-mass value can't launder as an absolute moment. `emu·mol⁻¹` molar susceptibility (4π/per-mole/Oe) is deferred until its factor is reference-verified.
- **G-D10 (`internally_validated` stays ≥1; ≥2 is *unsound now*, not merely deferred)** One fully-confirmed grounding sets `independent_sources=1` and clears the per-claim bar — now a *real* code-adjudicated, conditions-checked source (strict upgrade over the prior LLM-asserted ≥1). ≥2 must **not** be flipped on now: this lens is the only one that code-adjudicates external *support*; the grounder `[A1]`, executor, and prover all write the count from say-so/proof, so a ≥2 bar would force a say-so co-signer — re-admitting the say-so the project excludes. Revisit only when a second lens code-adjudicates *support* on the same check (and pair it with the symbol-aware quantity axis to defeat shared-blind-spot laundering). `artifact.py` untouched here.

---

## 14. Build slices (order)
1. **The adjudicator + scale-table + quote rule + conditions predicate (pure code, no network/LLM) — the honesty core.** `valagents/grounding.py`: `SCALE_TABLE` + `convert` (whole-token, full-precision energy factors); the **G-D9 factor-verification test** (`reference_rel_tol`); `_quote_valid` (§6, full-compound-token); `_quantity_overlap` (§5 item 4 / G-D5a); `_conditions_compatible` (§5 item 5 / G-D5b — its **own** T/field ladders, never `SCALE_TABLE.convert`, with the G-D5c source-axis symmetry). `ground_value(...) -> GroundingResult` (§5 four-outcome tree). Exhaustive unit tests incl. every §11 probe needing no network — especially **wrong-conditions**, **claim-silent-field-with-non-zero-source (G-D5c)**, **sub-Kelvin `mK`**, **emu-denominator out-of-table**, **no-false-contradict-from-wrong-regime**, the **contradicts interior+boundary pins**, and the **factor-verification** test.
2. **Fetch + extraction agent.** Extend `references.py` with `fetch_source_text(locator) -> (text, meta) | None` (full text); add `valagents/agents/value_grounder.py` + `VALUE_GROUNDER` (anti-anchoring: no value, no claim_conditions). Tests with `FakeResolver` + `FakeLLM`.
3. **Data model + designer.** `ComputationPlan` gains `source_quantity`, `claim_conditions`, `source_unit`; `*_source` documented as a resolvable locator; `MAGNITUDE_DESIGNER` updated; `GroundCfg` gains `supports_factor`, `contradict_factor`, `quote_min_tokens`. Backward-compat tests.
4. **Gate integration.** Wire grounding between `_run_magnitude`'s verdict and `verdict_to_check`/`verdict_to_attack`: `supports` → one Source + `independent_sources=1` + loud basis; `inconclusive` → loud basis, `independent_sources=0`; `unconfirmed` → unchanged; `contradicts` → `uncertain` + loud. Snapshot into the `CheckRecord` (G-D8). End-to-end gate tests (discount lifts only on supports; inconclusive shows but doesn't clear; no-regression on backend=none; gate purity pins green).
