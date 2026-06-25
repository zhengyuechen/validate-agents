# validate-agents — LLM-planned, code-scoped arXiv retrieval (query planner)

**Status:** design, approved for spec (2026-06-25). Awaiting external review before plan/implementation.
**One line:** an LLM proposes a *structured* arXiv query (1–2 archives + 2–4 distinctive terms) per claim; **code validates, renders, and retrieves** — fixing the confirmed off-domain-retrieval root cause without touching the grounder's adjudication or the gate.

---

## 1. Problem (confirmed, not hypothesized)

The grounder retrieves literature by handing `search_articles(backend, claim.statement)` the **full claim sentence** with no category scoping. On real condensed-matter claims this returns **entirely off-domain papers** — particle physics (hep-ex B-meson/ATLAS), gravitational waves (gr-qc GWTC, astro-ph IceCube/LIGO) — for a superconductivity/Hall-coefficient claim. The grounder then correctly fail-closes to `uncertain` (indep=0), but **for the wrong reason: retrieval failure, not absent literature** (the CM literature is abundant). Every grounding-quality guarantee built so far (Tier-1, Tier-2, the on-property floor) sits *downstream* of this broken step, so on a real target the grounder is effectively non-functional.

**Confirmed root cause (live arXiv API, 2026-06-25).** It is *not* merely "no `cat:` scoping retrieves poorly." A 35-word claim sentence is dominated by generic-physics vocabulary — *momentum, energy, conservation, magnetic field, transition, observation* — which is **more characteristic of the high-volume hep-ex/gr-qc/astro corpora than of superconductivity**. arXiv search is keyword/Lucene relevance, not semantic. So:

- generic terms match enormous high-term-frequency particle/GW corpora strongly;
- the few domain-specific terms (*hole, Hall coefficient, superconductor*) are a minority of the query and get diluted;
- with no category constraint the **whole arXiv corpus competes**, and the fields whose papers best match the bulk (generic) vocabulary win the relevance ranking.

This was reproduced and the fix direction confirmed against the real API:

```
search_query = "the full claim sentence"            → hep-ex / gr-qc / astro-ph (the bug)
search_query = cat:cond-mat* AND (hole AND "Hall coefficient")
                                                    → cond-mat.str-el / .supr-con / .mes-hall (real literature)
```

**Implication:** shortening the query alone cannot work — the generic-physics words stay and keep winning. The lever is **corpus scoping (constrain the archive) PLUS distinctive-term extraction (search the salient terms, not the sentence).** Since the system is general (it does not know a claim's domain a priori), the fix is a query-construction step that *infers* the archive and *extracts* the salient terms.

---

## 2. Cardinal-rule firewall

The LLM proposes a **query**, never a verdict. This is exactly the computation-designer pattern already in the system: model proposes, **code retrieves and adjudicates**. The grounder's existing code-witnessed adjudication — quote admissibility, the on-property distinctive-token floor, the dedup'd ≥1-source bar, the contradiction guard — is **completely unchanged**. `artifact.py` and the gate are untouched. A bad query can only change *what literature is surveyed*; it can never launder into credit, because credit still requires a code-validated verbatim on-property passage from a retrieved abstract.

So the query planner carries **zero gate risk** and is correctly **default-ON** (contrast NLI, which swaps a behavior and is default-OFF): it is a pure-win, fail-soft retrieval improvement whose only cost is one extra LLM call.

---

## 3. Architecture

A new query-planning step sits **between** the grounder and `search_articles`:

```
ground_claim(claim) ─▶ plan_query(claim.statement, llm, cfg) ─▶ PlannedQuery(archives, terms)
                                      │
                                      ▼  (code validates archives, renders the query string)
                       render_query(planned, backend, widen) ─▶ "cat:cond-mat* AND (hole AND \"Hall coefficient\")"
                                      │
                                      ▼
                       search_articles(backend, rendered)  ─▶ pool ─▶ [UNCHANGED grounder adjudication]
```

**New unit — `valagents/agents/query_planner.py`** (one focused module):

- `async def plan_query(text: str, llm, cfg) -> PlannedQuery` — one LLM call. Emits a KEY: value tail (same parsing idiom as the grounder, via `checked`): `ARCHIVES: cond-mat, quant-ph | TERMS: hole, "Hall coefficient", superconductor`. Code splits on commas, **validates each archive against `VALID_ARCHIVES`** (drops hallucinated/unknown ones), keeps 2–4 terms. Returns `PlannedQuery(archives=[valid only], terms=[...])`. On LLM failure / unparseable tail → `PlannedQuery(archives=[], terms=[])`.
- `VALID_ARCHIVES: frozenset[str]` — the arXiv top-level archive set (see §6). Anti-hallucination allow-list.
- `def render_query(planned, backend, widen: bool = False) -> str` — pure, backend-aware (see §5). Returns the search string handed to `search_articles`.

**`PlannedQuery`** — a small frozen dataclass `{archives: list[str], terms: list[str]}`. No Pydantic/artifact coupling; it never enters `IdeaArtifact`.

**Wiring** — `ground_claim` and `ground_novelty` both call the planner before `search_articles` (same bug, same fix, reuse). Backed by the 3-rung ladder in §4.

---

## 4. Control flow — the 3-rung fail-soft ladder + one widen step

The two render levers are **scope** (the `cat:` filter — what fixes the bug) and **terms** (the AND/OR precision). The rule: **widen the keywords, never the scope.** Scope is sacred (relaxing it re-admits hep-ex/gr-qc); term precision is the safe thing to loosen.

```python
planned = await plan_query(text, llm, cfg)        # archives already validated; may be empty
arxiv = backend_label(backend) == "arxiv"         # reuse the existing helper in web_search.py

if planned.terms and planned.archives and arxiv:
    rung = "scoped"                                # RUNG 1 (best): cat:-scoped, terms AND'd
elif planned.terms:
    rung = "terms_only"                            # RUNG 2: no valid archive (or non-arXiv backend) → terms-only unscoped
else:
    rung = "raw"                                   # RUNG 3 (last resort): total planner collapse → today's behavior

if rung == "raw":
    fmt, arts = await search_articles(backend, text)              # raw sentence, unscoped (= current behavior)
    widened = False
else:
    q = render_query(planned, backend, widen=False)
    fmt, arts = await search_articles(backend, q)
    widened = False
    if arxiv and len(arts) < cfg.grounding.widen_min_results:     # WIDEN KEYWORDS, scope fixed
        q = render_query(planned, backend, widen=True)            # AND → OR across terms; cat:-scope unchanged
        fmt, arts = await search_articles(backend, q)
        widened = True
```

**Why a ladder, not a binary fallback.** The earlier binary ("planner fails *or* 0 valid archives → raw sentence") collapsed two different cases and one of them **reinstated the bug**: *valid terms but no valid archive* falling back to the raw sentence puts the full-sentence contamination right back. RUNG 2 (terms-only unscoped) reuses the render branch already needed for Tavily and is contamination-free, so the buggy raw-sentence path (RUNG 3) now fires **only on total planner collapse** (no usable terms) — making "the fix can only help" almost always literally true.

**Why widen fires often (accepted).** AND'ing several distinctive terms is restrictive — a paper matching *all* of them is rare — so the widen step will fire on many claims; "2 API calls" is closer to the common case than the worst case. Mitigation: the planner emits **2–4** terms (not 3–6), keeping the AND default tighter and the widen less frequent. This is an accepted cost, not a defect — the OR-widen still keeps the `cat:` scope, so it is still on-domain.

---

## 5. Render — the load-bearing string (empirically pinned)

**arXiv backend:**

```
(cat:cond-mat* OR cat:quant-ph*) AND ("term one" AND term2 AND term3)
```

- **Wildcard form: `cat:<archive>*` — no dot.** Confirmed against the live API (2026-06-25):

  | form | totalResults | coverage |
  |---|---|---|
  | `cat:cond-mat*` (no dot) | **187** | leaves **+** legacy bare-`cond-mat` umbrella tags; zero contamination |
  | `cat:cond-mat.*` (with dot) | 183 | leaves only; **drops** the 4 legacy-umbrella papers |
  | `cat:cond-mat` (no wildcard) | **4** | legacy umbrella only — effectively broken |

  Both wildcard forms are contamination-free (every primary category `cond-mat.*`) and overlap 9/10 in the top 10; the **no-dot form is strictly broader** (superset, includes pre-subcategory-era papers). **The `*` is mandatory** — `cat:cond-mat` without it returns 4 papers, not 187. Chosen form: **`cat:<archive>*`**.

- **1–2 archives, OR'd:** `(cat:a1* OR cat:a2*)`. Hedges the archive guess without dropping to the unreliable leaf level. OR syntax confirmed valid against the API. Cap at 2 (YAGNI; more dilutes the scope).
- **Phrase-quote multiword terms:** a term containing a space is wrapped in `"…"` (`Hall coefficient` → `"Hall coefficient"`), else it tokenizes as `Hall AND coefficient`. Single-word terms unquoted.
- **`widen=True`:** join terms with `OR` instead of `AND`; the `cat:` clause is **unchanged**.

**Non-arXiv backend (Tavily/other):** `cat:` is arXiv-only syntax. Render = the focused terms space-joined as a natural query (e.g. `hole Hall coefficient superconductor`) — still a large improvement over the full sentence. `widen` is a no-op for non-arXiv (Tavily relevance, not Lucene AND/OR). This is the same terms-only branch RUNG 2 uses.

---

## 6. `VALID_ARCHIVES` (complete top-level set)

Validate the archive token against the full arXiv archive list so a legitimate cross-disciplinary claim is **not** silently dropped to terms-only:

```
cond-mat, quant-ph, hep-th, hep-ex, hep-ph, hep-lat, gr-qc, astro-ph,
nucl-th, nucl-ex, physics, math, cs, eess, nlin, q-bio, q-fin, stat, econ
```

(Drop anything else as a hallucination.) The planner is told to emit a bare top-level archive (e.g. `cond-mat`, not `cond-mat.supr-con`); if it emits a leaf, code truncates at the dot to the archive prefix before validating. Rationale for archive-level, not leaf: the LLM is reliable at the archive level ("this is condensed matter") and unreliable at the leaf level (supr-con vs str-el), and cross-listing already blurs leaf boundaries — while the measured contamination lives entirely at the **archive** level (cond-mat vs hep-ex/gr-qc).

---

## 7. Audit synergy (transparency, free)

The `.candidates/<run_id>.jsonl` audit log (shipped 2026-06-25, commit `26c0c66`) already records, per claim, the **full retrieved pool with titles + URLs + per-article disposition** (`credited / quote_failed / contradicts / contradicts_unverified / uncited`) — so "save what it looked at" (titles/URLs of the pool) is **already closed**.

The query planner adds the missing half — *what query produced that pool* — by passing a `query` block into the existing `emit_candidates(**fields)` call:

```json
{ "claim": "C3", "query": { "rung": "scoped", "archives": ["cond-mat"],
    "terms": ["hole", "Hall coefficient", "superconductor"],
    "rendered": "cat:cond-mat* AND (hole AND \"Hall coefficient\" AND superconductor)",
    "widened": false, "n_hits": 7 },
  "n_retrieved": 7, "n_credited": 1, "contradicted": false, "candidates": [ … pool … ] }
```

Net result, end to end: **"here's the query it ran, here's the pool it got back, here's what it cited and why"** — exactly the qual transparency story, now complete. Diagnostic only; the gate never reads it.

---

## 8. Config (`valagents/config.py`, grounding sub-model)

```python
query_planner: bool = True        # default-ON: pure-win, fail-soft, no heavy dependency
widen_min_results: int = 3        # below this hit count, widen keywords (AND→OR), scope fixed
```

When `query_planner` is False, the grounder uses the current behavior (raw `claim.statement`) — i.e. RUNG 3 unconditionally. This is the kill switch / A-B lever, not a recommended default.

---

## 9. Error handling / fail-soft guarantees

- **Planner LLM failure or unparseable tail** → `PlannedQuery([], [])` → RUNG 3 (raw sentence). Run never breaks. Worst case = today's behavior.
- **All archives invalid/hallucinated, terms OK** → RUNG 2 (terms-only unscoped), contamination-free.
- **`search_articles` itself fails** (network/rate-limit) → already fail-soft (`("", [])`) per its existing contract; the grounder degrades to ungrounded exactly as today.
- **Strictly additive:** there is no input on which the planner makes retrieval *worse* than the current full-sentence behavior — the only path that uses the full sentence is total planner collapse, which is the current behavior.

---

## 10. Testing

**`plan_query` (FakeLLM):**
- parses `ARCHIVES: … | TERMS: …` tail into `PlannedQuery`; keeps 2–4 terms.
- drops a hallucinated archive (`ARCHIVES: cond-mat, frobnicate` → `["cond-mat"]`).
- truncates a leaf to archive (`cond-mat.supr-con` → `cond-mat`) before validating.
- unparseable / empty tail → `PlannedQuery([], [])`.

**`render_query` (pure):**
- arXiv scoped: `(cat:cond-mat* OR cat:quant-ph*) AND (hole AND "Hall coefficient")` — wildcard present, multiword phrase-quoted, single-word unquoted.
- `widen=True` → terms OR'd, `cat:` clause byte-identical to non-widened.
- no archives → terms-only, no `cat:`.
- non-arXiv backend → space-joined terms, no `cat:`, widen is no-op.

**Ladder (fake backend, FakeLLM):**
- RUNG 1: valid archives+terms → scoped query string sent; thin result triggers ONE widen call with scope preserved (assert the second query keeps `cat:` and flips AND→OR).
- RUNG 2: 0 valid archives, terms present → terms-only query sent, never the raw sentence.
- RUNG 3: planner returns no terms → raw `claim.statement` sent (current behavior).
- `query_planner=False` → raw sentence sent regardless.

**Grounder integration (with `run_log.bind`):**
- the `.candidates` record carries the `query` block (`rung`, `archives`, `terms`, `rendered`, `widened`, `n_hits`) alongside the pool — and the pool still carries titles/URLs (regression guard on the shipped audit trail).
- planner failure leaves the existing grounder verdict path intact (fail-soft).

**Live API (manual, not CI):** the §5 wildcard table is a manual probe, recorded here as evidence; not a network-dependent unit test.

---

## 11. Scope guard (YAGNI)

In scope: one planner module, the 3-rung ladder, one widen step, the two config fields, the audit `query` block, reuse in `ground_novelty`.

**Out of scope:** leaf-subcategory selection; a 3+-rung widen ladder; per-claim query caching; a new search backend; semantic/embedding retrieval; query rewriting beyond AND→OR. Each is a separate decision if RUNG-1 retrieval later proves insufficient — measured via the audit log's `n_hits`/disposition mix, not guessed.

---

## 12. Design decisions (traceable)

- **QP-D1 — archive-level scope, not leaf.** LLM reliable at archive, unreliable at leaf; contamination lives at archive level; cross-listing blurs leaves. (User-chosen Option 1.)
- **QP-D2 — 1–2 archives OR'd.** Hedge the archive guess without leaf-level fragility. Cap 2. (User refinement.)
- **QP-D3 — widen keywords, never scope.** Scope is what fixes the bug; relaxing it re-admits contamination. Widen = code-side AND→OR, one step, no extra LLM call. (User refinement.)
- **QP-D4 — `cat:<archive>*`, no dot.** Empirically the broader, contamination-free superset (187 vs 183 vs 4); the `*` is mandatory. (Verified live, §5 — load-bearing string the user flagged.)
- **QP-D5 — 3-rung fail-soft ladder.** scoped → terms-only-unscoped → raw-sentence. Terms-only before raw sentence so the buggy full-sentence path fires only on total planner collapse. (User redline.)
- **QP-D6 — 2–4 terms, not 3–6.** Keeps the AND default tighter so widen fires less. (User note.)
- **QP-D7 — `VALID_ARCHIVES` is the complete top-level set** incl. `eess, nlin, q-bio, q-fin, stat, econ` so a legitimate cross-disciplinary archive isn't silently dropped. (User note.)
- **QP-D8 — default-ON.** Pure-win, fail-soft, only cost is one LLM call — unlike NLI (behavior swap, default-OFF). (User confirmation.)
- **QP-D9 — backend-aware render.** `cat:` for arXiv; terms-only space-join for Tavily/other (the RUNG-2 branch). `cat:` is arXiv-only syntax.
- **QP-D10 — firewall preserved.** Model proposes a query; code validates/renders/retrieves/adjudicates. `artifact.py` and the gate untouched; zero gate risk.

---

## 13. Open questions

None blocking. One acknowledged tradeoff (QP-D6 / §4): the AND default + frequent widen means many claims cost 2 arXiv calls; accepted, mitigated by 2–4 terms, revisit only if the audit log shows widen firing near-universally.
