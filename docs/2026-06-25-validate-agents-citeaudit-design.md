# validate-agents — CiteAudit Design (verify the LLM-named narrative references)

- **Date:** 2026-06-25
- **Status:** Approved design, pending implementation plan
- **Builds on:** `references.py` (`Reference`, `DefaultResolver`, `normalize_id`, `build_references`, `to_bibtex`), `web_search.ArxivBackend` (query-search), `grounding.py` (`_content_tokens`, `_norm`), `cli.py` (`run_cli`, `render_report`, `_render_supporting_layer`).
- **Source:** the CiteAudit card + "build first (three)" reorder in `docs/2026-06-25_papers_for_validate_agents_report.md`. First of three cheap pure-code wins (CiteAudit → VeriGuard → NLI), ahead of Popper.
- **One-line goal:** Stop the report from presenting an LLM-asserted, unverified paper name as an established citation. Each rendered narrative reference is either **resolved** to a real catalogued record (and annotated with that record) or marked **`[unverified]`** — adjudicated by deterministic field-match, never by model say-so.

---

## 1. Scope

**In scope — the two *rendered, genuinely-citable* narrative fields:**
- `prior_art_positioning.closest_prior` (str) — rendered at `cli.py:202`.
- `prior_art_positioning.must_cite` (list[str]) — rendered at `cli.py:206`.

**Out of scope (decisions):**
- **`theory_bridge.nearest_theories` — NOT resolved (CA-D2).** A theory (BCS, RVB, Fermi-liquid) is not a single paper; resolving it to one arXiv/Crossref record is a category error, it is where both residuals (false-attach, doubt-casting `[unverified]`) concentrate, and "Fermi-liquid theory [unverified]" misreads as doubting the theory. It renders as **plain text, no marker** — unchanged from today.
- **`Novelty.closest_prior`** — not rendered anywhere in `cli.py`, so out of scope for an output-integrity feature. Revisit only if it becomes rendered.
- **The gate.** CiteAudit is **output integrity only**. It never feeds `internally_validated`, `independent_sources`, or any verdict. `artifact.py` is untouched.
- **Entailment / "does this work support the claim."** CiteAudit verifies a citation *exists and is correctly identified*, never that it supports anything (the paper's own concession). That frontier is the grounder's, not this.

---

## 2. The match rule (deterministic — the cardinal-rule core)

With *verify-and-annotate* (§5), the harm to prevent is a **false attach** — printing a real-but-*wrong* paper as "the closest prior." So the attach gate is high-precision and pure code, mirroring the grounding floor's require-ALL:

`_title_match(name: str, candidate_title: str, min_name_tokens: int) -> bool`:
1. **Title-like gate:** `len(_content_tokens(name)) >= min_name_tokens` (default **3**). A short/generic name ("prior model", "Born reciprocity" → 2 content tokens) is not treated as a verifiable paper title → never attaches → `[unverified]`.
2. **Require-ALL:** `_content_tokens(name) <= _content_tokens(candidate_title)` — every content token of the name appears in the candidate's normalized title.

Reuses `grounding._content_tokens` / `_norm` (same NFKC + casefold + stopword tokenizer, for consistency; "magnetic" is stopped on both sides, so the comparison stays symmetric). **No LLM anywhere** — `Consistent()` is a pure string/set rule.

**Why this is high-precision:** a wrong paper would have to carry *every* content token of the name in its title — near-impossible for multi-distinctive-token names. The residual is documented in §6.

`min_name_tokens` is a config knob (`CiteAuditCfg.min_name_tokens: int = 3`) — bump to 4 to kill most generic-3-token over-specification if `nearest_theories` is ever re-added or false-attaches are observed.

---

## 3. Search backends (network — propose candidates only, never adjudicate)

The model/network *proposes* candidate records; §2's pure code *adjudicates*. Same split as grounding.

- **arXiv:** reuse `web_search.ArxivBackend.search(query=name, max_results=cfg.arxiv_rows)` → `Article{title, summary, url, published}`.
- **Crossref:** new `_crossref_title_search(name, rows=cfg.crossref_rows)` — `httpx` GET `https://api.crossref.org/works?query.bibliographic=<name>&rows=<rows>`, parse `message.items[*]` → `{title: item.title[0], authors, year, doi/url}`. Keyless (same Crossref host the `DoiResolver` already uses, different endpoint).
- **Config (`CiteAuditCfg`):** `min_name_tokens: int = 3`, `arxiv_rows: int = 5`, `crossref_rows: int = 5`. The pure `_title_match` takes `min_name_tokens` as a primitive (the grounding-helpers-take-primitives convention); the `CiteAuditor` unpacks the rows.
- Both keyless, no new dependency. Order: **arXiv first, then Crossref; the first candidate passing `_title_match` wins** (with that backend's metadata). Physics "closest prior" is often journal/classic (Crossref) — arXiv-only would miss it — so both are queried.
- **Authors caveat:** `web_search.ArxivBackend.Article` carries `{title, summary, url, published}` — **no authors**. So an arXiv-matched `Reference` has `authors=[]` (render shows title + year + url, no author clause); Crossref candidates do carry authors. `_title_match` uses only the title, so matching is unaffected. (Accepted v1 limitation; not worth a second arXiv call to backfill authors.)
- **Fail-soft:** any backend exception (network, rate-limit, parse) → treated as "no candidate from that backend" (logged `warning`, like `safe_search`); never crashes the run. Both failing → `[unverified]`.

---

## 4. The auditor (`valagents/citeaudit.py`, injected, off by default)

```python
@dataclass
class CiteResult:
    name: str
    status: str                 # "resolved" | "unverified"
    reference: Reference | None = None   # populated iff resolved (origin="asserted")

class CiteAuditor:
    """Injected dependency. None at the call site → CiteAudit OFF (report identical to today).
    Holds the backends + cfg; tests inject a fake with canned candidates."""
    async def audit(self, name: str) -> CiteResult: ...
```

`audit(name)`:
- Query arXiv then Crossref; for each candidate in order, if `_title_match(name, candidate.title, cfg.min_name_tokens)` → build `Reference(origin="asserted", title=candidate.title, authors=..., year=..., url=..., locator=normalize_id(candidate.url_or_doi))` and return `CiteResult(name, "resolved", ref)`.
- No match (or empty/short name, or all backends failed) → `CiteResult(name, "unverified")`.
- **On/off is the injected object** (`None` = off), not a config flag — mirrors `value_grounder`'s resolver. The CLI builds a live `CiteAuditor` only when grounding/network is wanted; tests inject a fake.

A thin orchestration helper `audit_narrative_refs(art, auditor) -> dict[str, CiteResult]` collects the in-scope field values (`closest_prior` + each `must_cite`), dedups by `_norm(name)`, and returns `{name → CiteResult}`. Returns `{}` when `auditor is None`.

---

## 5. Data flow + output

**`cli.py` `run_cli`** gains a `citeauditor=None` parameter (alongside `resolver`). After `build_references`:
1. `audit_map = await audit_narrative_refs(art, citeauditor)` (`{}` if off).
2. Merge resolved refs into the bibliography: `build_references` gains an optional `asserted_refs: list[Reference] | None = None` parameter; the resolved refs are folded into its `by_locator` dict (keyed by `normalize_id`) **before** the existing numbering/key/sort tail — so they receive `[n]` + a BibTeX entry. Resolved refs have `cited_by=[]`, so the existing `(0 if cited_by else 1, locator)` sort places them after claim-cited refs.
   - **Collision rule — EXISTING WINS (`setdefault`, CA-D8):** if an asserted ref's `normalize_id` locator already exists in `by_locator` (a narrative `closest_prior` that resolves to a paper a claim already cited, or that the provided list already holds), the **existing entry is kept and the asserted ref discarded** — `by_locator.setdefault(asserted.locator, asserted)`. Do NOT overwrite: clobbering would blank the existing `cited_by` (demoting a claim-cited paper to the uncited tail) and flip its `origin` (`retrieved`/`provided` → `asserted`), corrupting provenance and the BibTeX note on a paper a claim genuinely cited. The narrative name still resolves correctly — the inline `[n]` join (below) goes through the locator to whichever entry survives.
3. **Name → `[n]` join (explicit):** for a resolved name, the marker number is `audit_map[name].reference.locator` → look that locator up in the numbered `refs` (`{normalize_id(r.locator): r.number}`) → `r.number`. The join is by *locator*, not by object identity, so it returns the surviving entry's number even when the asserted ref was discarded by the collision rule. `render_report` receives `refs` + `audit_map` and builds this locator→number map once.
4. `render_report(art, refs, audit_map)` and `to_bibtex(refs)` as before.

**Render annotation (`_render_supporting_layer`):** for `closest_prior` and each `must_cite`, look up `audit_map[name]`:
- `resolved` → `name — {Title}, {Authors[0] et al.} {Year} ({url}) ✓ [n]` (the resolved record is shown **loud** so a human can sanity-check the match).
- `unverified` (or off → no entry) → `name [unverified]` (off → just `name`, unchanged).

**Bibliography provenance.** Extend `references.Reference.origin` Literal `["provided", "retrieved"]` → add `"asserted"` (in `references.py`; **`artifact.py` untouched**). `to_bibtex`'s note line already emits `origin=…`, so an asserted ref reads `note = {origin=asserted; relation=unknown}`.

**A one-line in-report gloss** is printed once under the Prior-Art Positioning block when any `[unverified]` marker appears: *"`[unverified]` = not resolved to a catalogued record; not a claim of fabrication."* (CA-D4 — prevents misreading a working feature as broken.)

---

## 6. Expected behavior (honesty note — write it down so it's not read as a bug)

**`[unverified]` will be the common outcome, and that is correct.** require-ALL means an LLM free-text name (a paraphrase, a description, a short label) rarely token-subsets a canonical title verbatim, even for a real paper. This is the same fail-closed direction as the grounder's "promotes rarely" — a report peppered with `[unverified]` is the feature *working*, not failing. The marker wording (§5 gloss) and the loud resolved-record make the distinction legible: resolved = a real record we matched; `[unverified]` = we could not resolve it, **not** "we think it's fake."

**Stoplist amplifies `[unverified]` (known cause, not a bug):** `_title_match` reuses `grounding._content_tokens`/`_STOP`, which was tuned for claim↔quote topicality and stops domain words like "magnetic". For citation titles that strips meaningful tokens at *both* the min-token gate and require-ALL — "magnetic order" → `{order}` (1 token → fails min-3), "magnetic spin model" → `{spin, model}` (2 → `[unverified]`). Safe direction (fail-closed), fine for v1; the named **recall knob** if `[unverified]` proves too common in practice is a lighter citation-specific stoplist (don't change now — documented so it's a known cause, not a surprise).

**Residual — generic-token over-specification (accepted, mild — CA-D5):** because the search queries the name itself, its top hits share its words; for a generic 3-common-token name ("spin liquid model"), require-ALL can then attach a *real-but-arbitrary* topical paper. This is **over-specification, not fabrication**: the attached paper is real, and the loud resolved-title lets a human catch a mis-attach. Bounded by `min_name_tokens` (raise to 4 to kill most). Out-of-scope generics like `nearest_theories` — the worst offenders — are not resolved at all (§1, CA-D2).

---

## 7. Off / errors / determinism

- **Off:** `citeauditor=None` → `audit_map = {}` → render and bibliography are **byte-identical to today** (regression-pinned). The default `main()` wiring builds a live auditor only when appropriate; no flag flips behavior silently.
- **Fail-soft:** any backend error → that name `unverified`; never crashes a run (literature search is best-effort, like `safe_search`).
- **Determinism boundary:** the search (candidate proposal) is network/non-deterministic; the **adjudication (`_title_match`) and the marker decision are pure deterministic code**. The resolved record is recorded in the `Reference` (shown + BibTeX'd) for reproducibility/human-check.

---

## 8. Testing

- **`_title_match` (pure unit):** title-like gate (2-token name → False regardless of candidate); require-ALL pass ("phase space quantum mechanics" ⊆ "Quantum Mechanics in Phase Space" → True); **false-attach rejected** (name with a token absent from the candidate title → False); generic over-specification documented (a 3-token generic name *does* match an arbitrary same-token title → True, asserted as accepted behavior, not a bug); `min_name_tokens` knob respected (same name False at 4, True at 3).
- **`CiteAuditor` + fake backends (canned candidates, no network):** resolved-attach builds a `Reference(origin="asserted", …)` with the matched metadata; no-match → `unverified`; arXiv-miss-Crossref-hit → resolved via Crossref; backend raising → `unverified` (fail-soft); short/empty name → `unverified` without a search.
- **`audit_narrative_refs`:** collects only `closest_prior` + `must_cite` (NOT `nearest_theories`); dedups by normalized name; `auditor=None` → `{}`.
- **Bibliography merge (the correctness-critical cases):** (1) duplicate asserted refs (two narrative names resolving to the same locator) → one entry; (2) **collision with a claim-cited ref (CA-D8) — a narrative `closest_prior` whose resolved locator equals a `retrieved` claim-cited ref → ONE bibliography entry, the claim's `[n]` reused, the existing `cited_by` and `origin="retrieved"` PRESERVED (asserted ref discarded); the narrative marker still shows that `[n]`.** (3) asserted ref with a fresh locator → new entry, `origin="asserted"`, sorted after claim-cited refs.
- **cli integration (fake auditor):** resolved name → inline `✓ [n]` marker + the resolved ref present in `refs`/BibTeX with `origin=asserted`; unverified name → `[unverified]` + the gloss line; `nearest_theories` line unchanged (plain text); **`citeauditor=None` → report + bib byte-identical to a no-auditor run** (regression pin).

---

## 9. Cardinal-rule fit

Extends the cardinal rule to the **output layer**: stop printing unverified model claims about what papers exist. The adjudication is a deterministic field-match (no say-so); a false-attach is blocked by require-ALL + the min-token gate; `[unverified]` is honest ("not resolved", never "fabricated"). It does **not** touch the gate — a fabricated reference never earned credit there anyway (the grounder's sources are retrieval-backed + quote-checked); this closes the *narrative* surface the report renders verbatim. Network proposes; code adjudicates — the same firewall as grounding.

---

## 10. Decision log
- **CA-D1 (output integrity, not the gate)** CiteAudit annotates the rendered report; it never feeds `internally_validated`/`independent_sources`. `artifact.py` untouched.
- **CA-D2 (drop `nearest_theories`)** A theory is not a single paper — resolving it to one record is a category error and the highest false-attach / doubt-casting risk. Render it as plain text, no marker. Only `closest_prior` + `must_cite` (genuinely citable works) are resolved.
- **CA-D3 (verify-and-annotate, fail-loud)** On a deterministic title-match, attach the resolved record (loud, for human-check); else `[unverified]`. Never drop a name; never call it "fabricated."
- **CA-D4 (`[unverified]` is common + glossed)** require-ALL makes `[unverified]` the frequent, correct fail-closed outcome; a one-line in-report gloss states it means "not resolved to a record," not "suspected fake."
- **CA-D5 (require-ALL + `min_name_tokens=3`, knob)** High-precision attach (every name token in the candidate title) + a title-like gate. Residual: generic-3-token over-specification (real-but-arbitrary paper) — accepted as mild (real + human-checkable via the loud title); `min_name_tokens` is a knob (→4 kills most).
- **CA-D6 (arXiv + Crossref, keyless; injected, off-by-default)** Both backends queried (preprints + journal/classic); first require-ALL match wins; the auditor is an injected dependency (`None` = off → report identical to today), so it never fires headless/no-network and tests inject a fake.
- **CA-D7 (`origin="asserted"` provenance)** Resolved narrative refs join the bibliography as `origin="asserted"` (distinct from `provided`/`retrieved`); extends the `references.Reference` Literal only — `artifact.py` untouched.
- **CA-D8 (merge collision — existing wins)** On a locator collision in `build_references`, the **existing** entry (`retrieved`/`provided`) is kept and the asserted ref discarded (`setdefault`, not assignment). Overwriting would blank a claim-cited paper's `cited_by` (demoting it to the uncited tail) and corrupt its `origin`/BibTeX provenance. The narrative `[n]` join is by *locator* → the surviving entry's number, so the marker is still correct. (Regression-tested in §8.)

---

## 11. Build slices (order)
1. **Pure-code match + Crossref search (no orchestration).** In `valagents/citeaudit.py`: `_title_match` (+ reuse `grounding._content_tokens`/`_norm`), `_crossref_title_search`, `CiteAuditCfg.min_name_tokens`; extend `references.Reference.origin` to add `"asserted"`. Exhaustive `_title_match` unit tests + a Crossref-parse test against a canned JSON fixture.
2. **The auditor + orchestration.** `CiteResult`, `CiteAuditor` (arXiv+Crossref, first-match-wins, fail-soft), `audit_narrative_refs`; unit tests with fake backends (resolved/unverified/crossref-fallback/fail-soft/short-name).
3. **CLI wiring + render + bibliography.** `run_cli(citeauditor=None)`, merge resolved refs into `build_references` numbering, `render_report`/`_render_supporting_layer` inline markers + gloss line, `main()` builds a live auditor. Integration tests (fake auditor): markers + asserted ref in refs/BibTeX; `nearest_theories` unchanged; `None` → byte-identical report (regression pin).
