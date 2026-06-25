# validate-agents — Citation circle: author-cluster the grounder's independent-source count (S2-disambiguated)

**Status:** design, approved-in-principle (2026-06-25). Awaiting external review before plan/implementation.
**One line:** count the grounder's "independent sources" as **distinct author clusters**, not papers — so a self-citing author's N papers collapse to one — with Semantic Scholar `authorId`s as exact cluster keys and arXiv name-matching as the always-available fallback.

---

## 1. Problem — the second independence leak

The gate credits a claim when `independent_sources >= 1` (and, on the roadmap, `>= 2`). "Independent sources" has **two** ways to be a lie, on different axes:

1. **Say-so independence** — the `[A1]` support relation labelled `independent` by the *model*. Tier-2 closed this: code now witnesses a verbatim on-property quote, and the field name carries the honest caveat that independence is the model's label, not code-witnessed.
2. **Count distinctness** — "2 sources" actually meaning two *distinct groups*, not one author counted twice. **This is open.** `_dedup_articles` (grounder.py:24) collapses by arXiv-id / DOI / normalized title — so the *same paper* counts once, but **forty papers by one author count forty times**. Raise the bar to `>= 2` today and a self-citer clears it trivially: two Hirsch papers → "2 sources" → pass.

These are orthogonal. Tier-2 fixed axis 1; **this fixes axis 2**. Together they are the precondition that makes a `>= 2` bar *sound* — which is why this is not a "broad tightener," it is **the enabling condition for the tightening the roadmap has wanted and kept deferring**.

**Why now:** the bar was never raised partly because thin abstract-only retrieval mostly demotes (everything would fail `>= 2` vacuously). The query planner (shipped 2026-06-25) changed retrieval richness. So the discipline is: ship clustering at **bar = 1** (honesty + count-correction + measurement), then read the real cluster-count distribution under the new retrieval, *then* decide if `>= 2` is achievable or vacuous. Measure, don't assume.

---

## 2. Cardinal-rule framing

Author identity (arXiv names, S2 `authorId`s) is **ground-truth metadata**, not a judgment. Code counts distinct clusters. Nothing here lets a model — or a heuristic — decide a verdict:

- S2/arXiv are **metadata sources** in the same role as `ArxivBackend` is for retrieval: they supply facts; code adjudicates. An `authorId` is a fact about who wrote a paper, not an opinion about whether the claim is supported.
- The **dominant-cluster heuristic never gates.** "The biggest cluster is probably the self-citer" is an inference; letting it flip a verdict would re-introduce exactly the say-so on the independence axis that Tier-2 killed. It is **surfaced to the human, never gated** — the same surface-don't-gate rule the Popper spec adopted.
- `artifact.py` and the gate's `>= 1` logic are **untouched**. This changes only the integer `code_witnessed` feeds into `min(...)`, computed by pure code from ground-truth authorship.

**v1 is verdict-neutral at bar = 1** (proof in §6): clustering can only *reduce* `code_witnessed`, and never below 1 while any credited article exists, so `min(self_report, n_clusters) >= 1` holds exactly when it did before. The count change is invisible to the gate, visible to the human and the measurement log. The actual verdict movement waits for the evidence-gated `>= 2` decision (roadmap).

---

## 3. The three-axis independence picture (context, not all in scope)

Clustering cleans up the **weakest** channel (literature). It is deliberately one leg of a tripod the roadmap already implies:

1. **Verify the math** (prover / symbolic executor) — strongest; can't be fooled by reputation or repetition. Independent of the literature entirely.
2. **Test against *independent* physics/data** — run the claim through standard theory (BCS) or real measurements, **not the author's own model**. Independence has to live in the physics you test with.
3. **Count groups, not papers** (this spec) — the literature channel, made honest.

The honest ceiling (§11): name-only clustering can't catch *disjoint-author* collusion (a citation ring with no shared names looks independent). S2 `authorId`s sharpen merges/splits but don't change that ceiling. That residue is why channels 1–2 remain the strong ones — and why the system's job is to be **honest about how far each claim got**, not to claim it solved circularity.

---

## 4. Architecture & data flow

```
ground_claim
  └─ planned_search → credited Articles (each Article.url = arXiv entry id)
       └─ _dedup_articles (existing: collapse same work)
            └─ S2Client.enrich_authors([arxiv ids])  ── one batched call, global 1/s throttle, cached
                 → {arxiv_id: [(authorId, name)]}     (── {} on any failure → name-only)
            └─ _cluster_by_author(deduped, id_map)    ── union-find on shared author keys (grounding.py)
                 → clusters
       code_witnessed = len(clusters)                  ── distinct groups, not papers (bar stays 1)
       independent_sources = min(self_report, code_witnessed)
       run_log.emit_candidates(... clusters={n_clusters, largest_share, s2_enriched} ...)
```

**Module decomposition** (follows the existing separation: API clients are their own modules, pure grounding helpers live in `grounding.py`):
- `valagents/semantic_scholar.py` *(new)* — the throttled, cached batch client (I/O).
- `valagents/grounding.py` — `_author_key`, `_cluster_by_author` (pure; beside `_dedup_articles`/`_norm`, per the C1 decision — not a new package, not a new flat file).
- `valagents/web_search.py` — `Article.authors` field + populate in `ArxivBackend`.
- `valagents/agents/grounder.py` — enrich + cluster + count.
- `valagents/citeaudit.py` — consume `Article.authors` (close the documented caveat).
- `valagents/config.py` — `S2Cfg`.

---

## 5. Component A — `Article.authors` (closes a defect citeaudit already confesses)

`Article` (web_search.py:18) carries no authors. This is not new scope — `citeaudit.py:82` literally documents the gap: `# web_search.Article carries no authors (the authors caveat) -> authors=[]`. One repair, two consumers.

- Add `authors: list[str] = field(default_factory=list)` to `Article`.
- `ArxivBackend.search`: populate `authors=[a.name for a in r.authors]` — the exact extraction already used at `references.py:79` (`[author.name for author in result.authors]`), lifted verbatim. arXiv always returns authors, so this channel is free and always present.
- `TavilyBackend`: leave `authors=[]` (web results carry no structured authors); clustering degrades to "every article its own cluster" for that backend, which is safe (never over-merges).
- `citeaudit.py:83`: change `authors=[]` to `authors=list(a.authors)` — the confessed caveat retired; CiteAudit's title/author field-match gets real authors for free.

---

## 6. Component B — `_cluster_by_author` (grounding.py, pure)

```python
def _author_key(name: str, author_id: str | None = None) -> str:
    if author_id:
        return f"s2:{author_id}"                 # exact — S2 already merged name variants
    parts = _norm(name).split()                  # name fallback: greedy last|first-initial
    if not parts:
        return "anon"
    return f"name:{parts[-1]}|{parts[0][:1]}"    # SAFE OVER-MERGE: two 'J. Wang' merge (under-counts = conservative)

def _cluster_by_author(articles: list, id_map: dict) -> list[list]:
    """Union-find over articles; two articles share a cluster iff they share ANY author key.
    id_map: {arxiv_id: [(authorId, name)]} from S2; absent -> fall back to Article.authors names."""
    # keys(article) = s2 authorId keys if enriched, else name keys from article.authors
    # union all articles sharing a key; return the connected components (each a list of articles)
```

Properties:
- **Exact keys when enriched** (`s2:<authorId>`): eliminates false-merge (two different "J. Wang") *and* false-split ("Jorge Hirsch" vs "J. E. Hirsch"). This is the proper replacement for the dead ORCID branch.
- **Name fallback is fail-safe by over-merging**: greedy `last|initial` can merge two distinct same-surname authors → *under*-counts independence → conservative for the gate (never inflates). The gate's bias is correct: when unsure, count fewer independent groups.
- An article with **no authors at all** (Tavily, or S2-unresolved + empty arXiv authors) gets a unique per-article key → its own cluster (never silently merged into another).

**Verdict-neutrality at bar = 1 (the proof):** `code_witnessed = len(clusters)`. Clustering only merges, so `1 <= len(clusters) <= len(deduped)` whenever `deduped` is non-empty. Thus `min(self_report, len(clusters)) >= 1` ⇔ `min(self_report, len(deduped)) >= 1` for the `>= 1` gate. No verdict moves. The only operation that could drive a cluster count to 0 is **origin-subtraction**, which is roadmap (§10), not v1.

---

## 7. Component C — `semantic_scholar.py` (the one S2 use in v1: exact cluster keys)

A module-level singleton client. **One job:** turn a claim's credited arXiv ids into `authorId`s via the batch endpoint.

**The batch endpoint (the only one that survives 1 req/s):**
```python
POST https://api.semanticscholar.org/graph/v1/paper/batch
  params = {"fields": "externalIds,authors"}              # authors -> [{authorId, name}]
  json   = {"ids": [f"arXiv:{bare_id}" for bare_id in ...]}   # bare id = strip URL prefix + vN suffix
  headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}        # from .env, never hard-coded
# returns a list IN INPUT ORDER, null for unresolved ids; up to 500 papers per call
```
For a claim's handful of credited papers, that is **one request**. Bare id derivation reuses the arXiv parsing already in `references.normalize_id` / `detect_kind`, plus an explicit trailing-`vN` strip.

**Rate-limit engineering (1 req/s is cumulative across all endpoints — this *will* bite under parallel claim grounding):**
- **Global async throttle** — a single module-level `asyncio.Lock` + monotonic last-call timestamp wrapping *every* S2 request, spacing them `>= 1.0 s` apart. Shared across all concurrent claim groundings in the run's event loop, so parallel claims queue instead of colliding into 429s.
- **Per-process cache** keyed by bare arXiv id — never fetch the same paper twice in a run (or across runs in the same server process).
- **429 backoff** — on HTTP 429, sleep (honoring `Retry-After` if present, else linear backoff) and retry a bounded number of times.
- **Batch is the budget win** — N papers → 1 token of the 1/s budget.

**Graceful degradation (S2 is an enhancement, never a hard dependency):**
- No `SEMANTIC_SCHOLAR_API_KEY` set, network error, timeout, or exhausted retries → `enrich_authors` returns `{}` → clustering falls back to arXiv name keys → **this is exactly the safe name-only v1**. Nothing breaks; the run never fails on S2.
- The `.candidates` audit records whether enrichment happened (`s2_enriched: <count>/<total>`), so a degraded run is visible, not silent.

---

## 8. Component D — grounder integration & surfacing

In `ground_claim`, replace the count site (grounder.py:107-109):
```python
deduped = _dedup_articles(passing)
id_map  = await enrich_authors([a.url for a in deduped], cfg)   # {} on any failure -> name-only; bare ids stripped internally
clusters = _cluster_by_author(deduped, id_map)
code_witnessed = min(len(clusters), len(articles))             # distinct GROUPS, not papers
independent_sources = min(as_int(tail["independent_sources"]), code_witnessed)
```

**Surfacing (surface-don't-gate):**
- **Audit** — extend the `.candidates` `query`/audit block (shipped 2026-06-25) with `clusters: {n_clusters, n_papers, largest_cluster_share, s2_enriched}`. This is what makes the `>= 2` decision measurable: read the cluster-count distribution across real claims straight from the log.
- **Human basis** — when one cluster dominates (e.g. `largest_cluster_share >= 0.7`), append a flag to the grounder basis: `[dominance: k/n credited supports are one author group]`. The gate still says what it said; the human sees the echo. No heuristic touches the verdict.

---

## 9. Config (`valagents/config.py`)

```python
class S2Cfg(BaseModel):
    enabled: bool = True                 # master switch; False -> always name-only
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    min_interval_s: float = 1.0          # global throttle spacing (the cumulative 1 req/s limit)
    max_retries: int = 2                 # 429 backoff attempts
    timeout_s: float = 20.0
```
Key is read from `os.environ["SEMANTIC_SCHOLAR_API_KEY"]` via the existing `load_dotenv()` (config.py:59); **absent key → name-only**, not an error (unlike Tavily, which errors — S2 is optional). Add to `Config` as `s2: S2Cfg = S2Cfg()`.

---

## 10. Roadmap (designed, NOT built in v1)

- **Raise the bar to `>= 2` — evidence-gated, separate commit.** After v1 ships, run real condensed-matter claims under the query-planner retrieval and read the cluster-count distribution from `.candidates`. If claims routinely surface `>= 2` independent clusters, raise the bar (then `1 cluster < 2` → insufficient, and the self-citer dies *as one group* — no origin identification needed). If they barely clear 1, the bar stays and clustering remains the honesty-fix-plus-echo-catch. **Do not raise the bar blind.**
- **Origin-subtraction — ground-truth only.** When a claim has a *declared* origin paper (named in the seed, or surfaced as `positioning.must_cite`/`closest_prior`), one S2 lookup yields its exact `authorId`s; drop that cluster from the count. This is the Hirsch-echo catch made precise — and it is cardinal-clean **only because the origin is declared (ground-truth), resolved to exact IDs**, never the dominant-cluster heuristic. It flips verdicts (can drive a count to 0) and needs the declared-origin plumbing, so it is its own spec. Note: `cluster-collapse + (>= 2)` already kills the pure self-citer without it, so origin-subtraction is added precision, not a prerequisite.

---

## 11. Things S2 exposes that we deliberately DON'T use

- **`hIndex`** is on the author object — **kept out of the gate**, exactly as decided: it would boost Hirsch, who literally invented it. Author prestige is not independence.
- **Citation edges** (`/citations`, `/references`) are available — **kept out of clustering**: a citation can be a *refutation*, so a citation edge is not a dependence-merge signal. Only ever useful later as a *surfaced* "who-cites-whom" view, never as a merge input.
- **The honest ceiling stays**: disjoint-author collusion (a ring with no shared names/IDs) reads as independent to any name- or ID-based clustering. That is the irreducible residue; channels 1–2 (math, independent physics) are the answer to it, and the verdict language must stay honest about it.

---

## 12. Testing

**`_author_key` / `_cluster_by_author` (pure):**
- exact key when `author_id` present (`s2:<id>`); name fallback `name:<last>|<initial>` otherwise.
- false-split fixed: `("Jorge Hirsch", "A1")` and `("J. E. Hirsch", "A1")` (same authorId) → one cluster.
- false-merge fixed: two "J. Wang" with *different* authorIds → two clusters.
- name fallback over-merges safely: two same-surname authors, no IDs → one cluster (conservative).
- forty same-author papers → one cluster; mixed (Hirsch×38 + one independent group) → two clusters.
- empty-authors article → its own singleton cluster (never merged).

**`semantic_scholar.S2Client` (httpx mocked, no network):**
- batch request shape: `arXiv:<bare id>` ids (URL prefix + `vN` stripped), `fields=externalIds,authors`, key header.
- results parsed in input order; `null` entries skipped, not crashed.
- global throttle spaces two calls `>= min_interval_s` apart (monotonic-clock assertion).
- cache: second enrich of the same id makes no second HTTP call.
- 429 → bounded retry then give up; key absent / network error / timeout → `{}` (degrade), never raises.

**Grounder integration (FakeLLM + fake backend + mocked S2):**
- bar = 1 verdict-neutrality: a 2-paper-same-author credited set yields `code_witnessed = 1` but the `>= 1` verdict is unchanged from pre-clustering.
- `.candidates` records `clusters.{n_clusters, largest_cluster_share, s2_enriched}`.
- S2 disabled / failing → name-only clustering still runs; run completes.
- dominance flag appears in the basis when one cluster dominates; absent otherwise.

**`Article.authors` plumbing:** `ArxivBackend` populates from `r.authors`; `citeaudit` candidate carries real authors (caveat-closure regression: `citeaudit.py:83` no longer hard-codes `[]`).

---

## 13. Design decisions

- **CC-D1 — count distinct author clusters, not papers.** Closes the count-distinctness leak (axis 2); the precondition for a sound `>= 2` bar.
- **CC-D2 — clustering lives in `grounding.py`** beside `_dedup_articles`/`_norm` (same genre of pure helper). No new package, no new flat file. (C1.)
- **CC-D3 — no `orcid` param.** A branch that can never fire is speculative flexibility the repo forbids; ORCID is added the day the raw Atom feed is parsed for it, not before. (C2.)
- **CC-D4 — `Article.authors` is a defect repair, not new scope.** Closes citeaudit's literally-commented "authors caveat"; one plumbing line copied from `references.py:79`, two consumers. (C3.)
- **CC-D5 — S2 `authorId` as exact cluster key; name-matching as always-available fallback.** Eliminates false-merge and false-split; proper replacement for the dead ORCID branch. S2 is an enhancement, never a hard dependency.
- **CC-D6 — batch endpoint behind a global 1 req/s throttle + per-id cache + 429 backoff.** The only request shape that survives the cumulative rate limit under parallel claim grounding.
- **CC-D7 — bar stays 1 in v1; verdict-neutral.** Honesty + count-correction + measurement now; the `>= 2` move is a separate evidence-gated commit read from `.candidates`. Measure, don't assume.
- **CC-D8 — dominance is surfaced, never gated.** The dominant-cluster heuristic warns the human; it never flips a verdict (surface-don't-gate, per Popper). Origin-subtraction, if ever built, is **ground-truth (declared origin → exact authorId) only** — never the heuristic.
- **CC-D9 — hIndex and citation edges deliberately excluded** from the gate and from clustering (prestige ≠ independence; a citation can be a refutation).

---

## 14. Open questions

None blocking for v1. Two are explicitly deferred by design: (a) whether `>= 2` is achievable or vacuous — answered by reading `.candidates` after v1 ships; (b) origin-subtraction's declared-origin source (seed field vs `positioning.must_cite`) — its own spec when triggered.
