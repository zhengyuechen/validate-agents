# arXiv Query Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix off-domain arXiv retrieval by inserting an LLM-planned, code-rendered query step before retrieval — the model proposes 1–2 archives + 2–4 distinctive terms, code validates/renders/retrieves/adjudicates.

**Architecture:** A new focused module `valagents/agents/query_planner.py` holds the pure renderer, the planner LLM call, and a `planned_search` ladder helper that wraps `search_articles`. The grounder swaps its two raw `search_articles(backend, sentence)` calls for `planned_search`, and threads the resulting audit query-block into the existing `.candidates` log. The grounder's quote-adjudication and the gate are untouched.

**Tech Stack:** Python 3, Pydantic v2 (config only), the existing `checked` strict-tail parser, `FakeLLM`/monkeypatch for tests, `pytest` (asyncio auto mode).

## Global Constraints

- **Firewall (cardinal rule):** the LLM proposes a *query*, never a verdict. Code validates/renders/retrieves/adjudicates. **NEVER modify `valagents/artifact.py`** or any gate logic. Zero gate risk.
- **Wildcard form is load-bearing:** render arXiv category scope as `cat:<archive>*` — **no dot, the `*` is mandatory** (`cat:cond-mat`=4 hits, `cat:cond-mat*`=187). Verified live.
- **`VALID_ARCHIVES` is the complete top-level set:** `cond-mat, quant-ph, hep-th, hep-ex, hep-ph, hep-lat, gr-qc, astro-ph, nucl-th, nucl-ex, physics, math, cs, eess, nlin, q-bio, q-fin, stat, econ`.
- **Caps:** 1–2 archives, 2–4 terms.
- **3-rung fail-soft ladder:** `scoped` (cat:+terms) → `terms_only` (no valid archive, terms unscoped — never the raw sentence) → `raw` (total planner collapse → today's behavior).
- **Widen keywords, never scope:** when arXiv hits < `widen_min_results`, retry once with terms joined by `OR` instead of `AND`; the `cat:` clause is unchanged. arXiv-only.
- **Config defaults:** `query_planner: bool = True` (default-ON), `widen_min_results: int = 3`.
- **Commits:** plain messages — **NO** `Co-Authored-By` / `Claude-Session` / "Generated with Claude" / any attribution trailer.
- **Test command:** `conda run -n cosci-reproduce python -m pytest tests/ -q` (asyncio auto mode — async tests need no decorator).
- **Backward compatibility:** existing grounder tests use a `FakeLLM` that returns the grounder body for every agent; for agent `query_planner` that body has no `ARCHIVES/TERMS` tail → `plan_query` returns empty → rung `raw` → `search_articles(backend, sentence)` — i.e. exactly current behavior. Do not break this.

---

### Task 1: Pure render core — `PlannedQuery`, `VALID_ARCHIVES`, `render_query`

**Files:**
- Create: `valagents/agents/query_planner.py`
- Test: `tests/test_query_planner.py`

**Interfaces:**
- Consumes: `backend_label` from `valagents/web_search.py` (returns `"arxiv"` for an `ArxivBackend`, `"none"` for `None`).
- Produces:
  - `PlannedQuery` — frozen dataclass `{archives: list[str], terms: list[str]}`, both defaulting to `[]`.
  - `VALID_ARCHIVES: frozenset[str]` — the top-level archive allow-list.
  - `render_query(planned: PlannedQuery, backend, widen: bool = False) -> str` — pure; only called with non-empty `planned.terms`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_query_planner.py
from valagents.agents.query_planner import PlannedQuery, VALID_ARCHIVES, render_query
from valagents.web_search import ArxivBackend


def test_render_scoped_arxiv_two_archives():
    p = PlannedQuery(archives=["cond-mat", "quant-ph"], terms=["hole", "Hall coefficient"])
    assert render_query(p, ArxivBackend()) == '(cat:cond-mat* OR cat:quant-ph*) AND (hole AND "Hall coefficient")'


def test_render_widen_or_keeps_cat_scope_identical():
    p = PlannedQuery(archives=["cond-mat"], terms=["hole", "Hall coefficient"])
    tight = render_query(p, ArxivBackend(), widen=False)
    wide = render_query(p, ArxivBackend(), widen=True)
    assert tight == '(cat:cond-mat*) AND (hole AND "Hall coefficient")'
    assert wide == '(cat:cond-mat*) AND (hole OR "Hall coefficient")'
    assert tight.split(" AND (", 1)[0] == wide.split(" AND (", 1)[0]   # cat: clause byte-identical


def test_render_normalizes_prequoted_term():
    p = PlannedQuery(archives=["cond-mat"], terms=['"Hall coefficient"'])
    assert render_query(p, ArxivBackend()) == '(cat:cond-mat*) AND ("Hall coefficient")'   # not ""Hall coefficient""


def test_render_terms_only_when_no_archives():
    p = PlannedQuery(archives=[], terms=["hole", "Hall coefficient"])
    assert render_query(p, ArxivBackend()) == '(hole AND "Hall coefficient")'              # no cat:


def test_render_nonarxiv_space_join_no_operators():
    p = PlannedQuery(archives=["cond-mat"], terms=["hole", "Hall coefficient"])
    assert render_query(p, None) == "hole Hall coefficient"                                 # backend_label(None)=="none"


def test_valid_archives_complete():
    for a in ("cond-mat", "quant-ph", "eess", "nlin", "q-bio", "q-fin", "stat", "econ"):
        assert a in VALID_ARCHIVES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_query_planner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'valagents.agents.query_planner'`

- [ ] **Step 3: Write the module**

```python
# valagents/agents/query_planner.py
"""Query planner: an LLM proposes a focused arXiv query (1–2 archives + 2–4 distinctive terms);
code validates, renders, and retrieves. Fixes off-domain retrieval — a long generic-physics claim
sentence is a better keyword match for the high-volume hep-ex/gr-qc corpus than for the real domain,
so unscoped relevance ranks particle/GW papers first. Same firewall as the designers: model proposes
a query, code adjudicates; artifact.py and the gate are untouched."""
from __future__ import annotations

from dataclasses import dataclass, field

from valagents.web_search import backend_label


@dataclass(frozen=True)
class PlannedQuery:
    archives: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


# arXiv top-level archives (anti-hallucination allow-list). Complete set incl. the physics-adjacent
# ones a legitimate cross-disciplinary claim may land in — so a real archive is never silently dropped.
VALID_ARCHIVES = frozenset({
    "cond-mat", "quant-ph", "hep-th", "hep-ex", "hep-ph", "hep-lat", "gr-qc", "astro-ph",
    "nucl-th", "nucl-ex", "physics", "math", "cs", "eess", "nlin", "q-bio", "q-fin", "stat", "econ",
})


def _norm_term(t: str) -> str:
    """Strip surrounding whitespace and any quotes the LLM already added, then phrase-quote if the term
    is multi-word (else arXiv tokenizes 'Hall coefficient' as 'Hall AND coefficient'). Idempotent — a
    pre-quoted term never becomes '""Hall coefficient""'."""
    t = t.strip().strip('"').strip()
    return f'"{t}"' if " " in t else t


def render_query(planned: PlannedQuery, backend, widen: bool = False) -> str:
    """Render the backend-specific search string. Called only with non-empty planned.terms."""
    if backend_label(backend) != "arxiv":
        # non-arXiv (e.g. Tavily): focused terms as a natural query — no cat:/Lucene operators.
        return " ".join(t.strip().strip('"').strip() for t in planned.terms)
    op = " OR " if widen else " AND "
    term_q = "(" + op.join(_norm_term(t) for t in planned.terms) + ")"
    if planned.archives:
        cat_q = "(" + " OR ".join(f"cat:{a}*" for a in planned.archives) + ")"   # wildcard MANDATORY
        return f"{cat_q} AND {term_q}"
    return term_q                                                                # terms-only (rung 2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_query_planner.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add valagents/agents/query_planner.py tests/test_query_planner.py
git commit -m "Query planner: pure render core (PlannedQuery, VALID_ARCHIVES, render_query)"
```

---

### Task 2: `plan_query` + the `QUERY_PLANNER` prompt

**Files:**
- Modify: `valagents/prompts.py` (add `QUERY_PLANNER` after `GROUNDER_NOVELTY`, ~line 127)
- Modify: `valagents/agents/query_planner.py` (add imports + `plan_query`)
- Test: `tests/test_query_planner.py` (append)

**Interfaces:**
- Consumes: `PlannedQuery`, `VALID_ARCHIVES` (Task 1); `COMMON_RUBRIC` from `valagents/prompts.py`; `build_messages`, `split_list` from `valagents/agents/base.py`; `checked` from `valagents/parse.py`.
- Produces: `async def plan_query(text: str, llm, cfg, context: str = "") -> PlannedQuery`. `context` (the broader formal claim) feeds ARCHIVE inference; `text` (the sub-claim) feeds TERM extraction. Returns `PlannedQuery()` (empty) on unparseable/failed tail.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_query_planner.py  (append)
from valagents.config import Config
from valagents.agents.query_planner import plan_query
from tests.fake_llm import FakeLLM


def _cfg():
    return Config(default_model="fake")


async def test_plan_query_parses_archives_and_terms():
    llm = FakeLLM(lambda a, m: 'ARCHIVES: cond-mat, quant-ph | TERMS: hole, "Hall coefficient", superconductor')
    p = await plan_query("a metal superconducts only if its carriers are holes", llm, _cfg())
    assert p.archives == ["cond-mat", "quant-ph"]
    assert p.terms == ["hole", '"Hall coefficient"', "superconductor"]


async def test_plan_query_truncates_leaf_and_drops_hallucinated_archive():
    llm = FakeLLM(lambda a, m: "ARCHIVES: cond-mat.supr-con, frobnicate | TERMS: hole, gap")
    p = await plan_query("x", llm, _cfg())
    assert p.archives == ["cond-mat"]                    # leaf -> archive; 'frobnicate' dropped


async def test_plan_query_caps_two_archives_and_four_terms():
    llm = FakeLLM(lambda a, m: "ARCHIVES: cond-mat, quant-ph, hep-th | TERMS: a, b, c, d, e, f")
    p = await plan_query("x", llm, _cfg())
    assert p.archives == ["cond-mat", "quant-ph"]
    assert p.terms == ["a", "b", "c", "d"]


async def test_plan_query_forwards_context_into_prompt():
    seen = {}
    def router(agent, messages):
        seen["user"] = messages[-1]["content"]
        return "ARCHIVES: cond-mat | TERMS: moment, anisotropy"
    p = await plan_query("the effective moment is 1.2 muB", FakeLLM(router), _cfg(),
                         context="a frustrated magnet realizes a quantum spin liquid")
    assert "frustrated magnet" in seen["user"]           # context reached the planner prompt
    assert p.archives == ["cond-mat"]


async def test_plan_query_failsoft_empty_on_unparseable():
    p = await plan_query("x", FakeLLM(lambda a, m: "I cannot help."), _cfg())
    assert p == PlannedQuery() and p.archives == [] and p.terms == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_query_planner.py -q`
Expected: FAIL — `ImportError: cannot import name 'plan_query'`

- [ ] **Step 3: Add the prompt**

In `valagents/prompts.py`, after the `GROUNDER_NOVELTY` block (the line ending `... | POSITION: new|special_case|restatement"""`), add:

```python
QUERY_PLANNER = COMMON_RUBRIC + """

Role: turn a claim into a focused arXiv search query so retrieval returns on-domain literature
instead of whatever shares the claim's generic-physics vocabulary.
Checklist:
- ARCHIVES: name 1-2 arXiv TOP-LEVEL archives the claim's literature lives in (e.g. cond-mat, quant-ph,
  hep-th, astro-ph). Use the bare archive, NOT a sub-category (cond-mat, not cond-mat.supr-con).
- TERMS: 2-4 DISTINCTIVE terms — the entities/observables that pick out THIS claim (e.g. hole,
  "Hall coefficient", superconductor), not generic physics words (energy, momentum, field, transition).
- Quote a multi-word term in double quotes so it matches as a phrase.
- Use CONTEXT only to decide the archive; draw the TERMS from the CLAIM itself.

CLAIM: {text}
CONTEXT (broader claim this was decomposed from; may be empty): {context}

End with exactly:
ARCHIVES: <archive1, archive2> | TERMS: <term1, term2, term3>"""
```

- [ ] **Step 4: Add `plan_query`**

In `valagents/agents/query_planner.py`, add to the imports at the top:

```python
from valagents.prompts import QUERY_PLANNER
from valagents.agents.base import build_messages, split_list
from valagents.parse import checked
```

and append the function:

```python
async def plan_query(text: str, llm, cfg, context: str = "") -> PlannedQuery:
    """Ask the LLM for a focused arXiv query. Validates archives in CODE (leaf->archive truncation,
    allow-list, cap 2) and caps terms (4). On any failure returns an empty PlannedQuery -> rung 'raw'."""
    user = QUERY_PLANNER.format(text=text, context=context or "(none)")
    tail = await checked(
        "query_planner",
        build_messages("You build focused literature-search queries.", user),
        ["ARCHIVES", "TERMS"],
        llm=llm,
    )
    if tail is None:
        return PlannedQuery()
    archives: list[str] = []
    for a in split_list(tail["archives"]):
        arch = a.split(".")[0].strip().lower()           # leaf (cond-mat.supr-con) -> archive (cond-mat)
        if arch in VALID_ARCHIVES and arch not in archives:
            archives.append(arch)
    terms = [t for t in split_list(tail["terms"]) if t][:4]
    return PlannedQuery(archives=archives[:2], terms=terms)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_query_planner.py -q`
Expected: PASS (11 tests total)

- [ ] **Step 6: Commit**

```bash
git add valagents/prompts.py valagents/agents/query_planner.py tests/test_query_planner.py
git commit -m "Query planner: plan_query + QUERY_PLANNER prompt (validate archives, cap terms, context for archive inference)"
```

---

### Task 3: `planned_search` ladder helper + config fields

**Files:**
- Modify: `valagents/config.py:8-14` (add two fields to `GroundCfg`)
- Modify: `valagents/agents/query_planner.py` (add `search_articles` import + `planned_search`)
- Test: `tests/test_query_planner.py` (append), `tests/test_config.py` (append)

**Interfaces:**
- Consumes: `plan_query`, `render_query` (Tasks 1–2); `search_articles`, `backend_label` from `valagents/web_search.py`; `cfg.grounding.query_planner`, `cfg.grounding.widen_min_results`.
- Produces: `async def planned_search(backend, text: str, llm, cfg, context: str = "") -> tuple[str, list, dict]` — returns `(formatted, articles, query_block)`. `query_block = {rung, archives, terms, rendered, widened, n_hits}`, `rung ∈ {scoped, terms_only, raw}`.

- [ ] **Step 1: Write the config field test**

```python
# tests/test_config.py  (append)
def test_grounding_query_planner_defaults():
    from valagents.config import Config
    c = Config(default_model="m")
    assert c.grounding.query_planner is True
    assert c.grounding.widen_min_results == 3
```

- [ ] **Step 2: Add the config fields**

In `valagents/config.py`, inside `class GroundCfg`, after the `subject_saturation_frac` line (line 14), add:

```python
    query_planner: bool = True            # default-ON: LLM proposes cat:+terms, code retrieves (QP-D8)
    widen_min_results: int = 3            # below this hit count, widen keywords AND->OR; cat: scope fixed (QP-D3)
```

- [ ] **Step 3: Write the failing `planned_search` tests**

```python
# tests/test_query_planner.py  (append)
import pytest
from valagents.agents.query_planner import planned_search
from valagents.web_search import ArxivBackend, Article


def _pool(n):
    return [Article(title=f"T{i}", summary="s", url=f"http://arxiv.org/abs/x{i}v1", published="2025") for i in range(n)]


async def test_planned_search_scoped_then_widens_keywords_not_scope(monkeypatch):
    pool, queries, calls = _pool(5), [], {"n": 0}
    async def fake_search(self, query, max_results=10):
        queries.append(query); calls["n"] += 1
        return pool[:1] if calls["n"] == 1 else pool        # thin first hit -> widen fires
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: 'ARCHIVES: cond-mat | TERMS: hole, "Hall coefficient"')
    fmt, arts, block = await planned_search(ArxivBackend(), "claim text", llm, _cfg())
    assert queries == ['(cat:cond-mat*) AND (hole AND "Hall coefficient")',
                       '(cat:cond-mat*) AND (hole OR "Hall coefficient")']
    assert block["rung"] == "scoped" and block["widened"] is True and block["n_hits"] == 5
    assert block["archives"] == ["cond-mat"] and block["rendered"] == queries[1]


async def test_planned_search_terms_only_when_no_valid_archive(monkeypatch):
    queries = []
    async def fake_search(self, query, max_results=10):
        queries.append(query); return _pool(5)
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: "ARCHIVES: frobnicate | TERMS: hole, gap")
    fmt, arts, block = await planned_search(ArxivBackend(), "the full claim sentence", llm, _cfg())
    assert block["rung"] == "terms_only"
    assert queries[0] == "(hole AND gap)"                   # no cat:, and NEVER the raw sentence
    assert "claim" not in queries[0]


async def test_planned_search_raw_when_planner_disabled(monkeypatch):
    cfg = Config(default_model="fake"); cfg.grounding.query_planner = False
    queries = []
    async def fake_search(self, query, max_results=10):
        queries.append(query); return _pool(1)
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: "ARCHIVES: cond-mat | TERMS: hole")     # ignored: planner off
    fmt, arts, block = await planned_search(ArxivBackend(), "the full claim sentence", llm, cfg)
    assert block["rung"] == "raw" and queries == ["the full claim sentence"]   # single call, current behavior


async def test_planned_search_raw_on_planner_collapse(monkeypatch):
    queries = []
    async def fake_search(self, query, max_results=10):
        queries.append(query); return _pool(1)
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: "no machine-readable tail here")        # plan_query -> empty
    fmt, arts, block = await planned_search(ArxivBackend(), "raw claim", llm, _cfg())
    assert block["rung"] == "raw" and queries == ["raw claim"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_query_planner.py tests/test_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'planned_search'` (and the config test fails until Step 2 is applied)

- [ ] **Step 5: Add `planned_search`**

In `valagents/agents/query_planner.py`, extend the `web_search` import and append the function:

```python
from valagents.web_search import backend_label, search_articles   # replace the existing backend_label-only import
```

```python
async def planned_search(backend, text: str, llm, cfg, context: str = "") -> tuple[str, list, dict]:
    """Plan -> validate -> render -> retrieve via the 3-rung fail-soft ladder with one widen step.
    Returns (formatted, articles, query_block). query_block is the audit record of what actually ran."""
    arxiv = backend_label(backend) == "arxiv"

    planned = PlannedQuery()
    if cfg.grounding.query_planner:
        planned = await plan_query(text, llm, cfg, context=context)

    if not planned.terms:                                          # RUNG 3: planner collapse -> today's behavior
        fmt, arts = await search_articles(backend, text)
        return fmt, arts, {"rung": "raw", "archives": [], "terms": [],
                           "rendered": text, "widened": False, "n_hits": len(arts)}

    rung = "scoped" if (planned.archives and arxiv) else "terms_only"
    q = render_query(planned, backend, widen=False)
    fmt, arts = await search_articles(backend, q)
    widened = False
    if arxiv and len(arts) < cfg.grounding.widen_min_results:      # widen KEYWORDS (AND->OR); cat: scope fixed
        q = render_query(planned, backend, widen=True)
        fmt, arts = await search_articles(backend, q)
        widened = True
    return fmt, arts, {"rung": rung, "archives": list(planned.archives), "terms": list(planned.terms),
                       "rendered": q, "widened": widened, "n_hits": len(arts)}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_query_planner.py tests/test_config.py -q`
Expected: PASS (16 query_planner + the config suite)

- [ ] **Step 7: Commit**

```bash
git add valagents/config.py valagents/agents/query_planner.py tests/test_query_planner.py tests/test_config.py
git commit -m "Query planner: planned_search 3-rung ladder + widen step; config fields (default-ON, widen_min_results=3)"
```

---

### Task 4: Grounder wiring — use `planned_search`, record the query block

**Files:**
- Modify: `valagents/agents/grounder.py` (import swap; `ground_claim` line ~53 + the `emit_candidates` call ~117–125; `ground_novelty` line ~148)
- Test: `tests/test_run_log_candidates.py` (append — reuses its fixtures)

**Interfaces:**
- Consumes: `planned_search` (Task 3); the shipped `run_log.emit_candidates(claim_id, **fields)`; fixtures `CLAIM`, `FC`, `A_SUPPORT`, `A_SYNTH`, `A_CONTRA`, `_cfg` from `tests/test_grounding_support_agent.py`.
- Produces: `ground_claim`/`ground_novelty` retrieve through the planner; the `.candidates` record gains a `query` block.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_run_log_candidates.py  (append)
import pytest
from valagents.web_search import ArxivBackend


async def test_ground_claim_records_query_block(tmp_path, monkeypatch):
    run_log.bind(tmp_path / ".logs" / "run-q.jsonl")
    queries = []
    async def fake_search(self, query, max_results=10):
        queries.append(query)
        return [A_SUPPORT, A_SYNTH, A_CONTRA]               # 3 >= widen_min_results -> no widen
    monkeypatch.setattr(ArxivBackend, "search", fake_search)

    def router(agent, messages):
        if agent == "query_planner":
            return 'ARCHIVES: cond-mat | TERMS: "noise PSD", "temperature-independent", YbZn2GaO5'
        return ("CLAIM: c1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | BASIS: x\n"
                "```json\n{\"citations\": []}\n```")
    await ground_claim(CLAIM, FC, ArxivBackend(), FakeLLM(router), _cfg())

    rec = json.loads((tmp_path / ".candidates" / "run-q.jsonl").read_text().splitlines()[0])
    assert rec["query"]["rung"] == "scoped"
    assert rec["query"]["archives"] == ["cond-mat"]
    assert "cat:cond-mat*" in rec["query"]["rendered"] and rec["query"]["widened"] is False
    assert any("cat:cond-mat*" in q for q in queries)       # the scoped query actually hit the backend
    assert rec["candidates"][0]["title"]                    # pool (titles/URLs) still recorded — regression guard
```

This test imports `ground_claim`, `FakeLLM`, `json`, and the fixtures already imported at the top of `tests/test_run_log_candidates.py`; add `import pytest` and the `ArxivBackend` import shown if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_run_log_candidates.py::test_ground_claim_records_query_block -q`
Expected: FAIL — `KeyError: 'query'` (the `.candidates` record has no `query` block yet)

- [ ] **Step 3: Swap the import in `grounder.py`**

In `valagents/agents/grounder.py`, replace line 14:

```python
from valagents.web_search import search_articles
```

with:

```python
from valagents.agents.query_planner import planned_search
```

(`search_articles` is no longer used in this file after this task — `ground_claim` and `ground_novelty` were its only two callers.)

- [ ] **Step 4: Wire `ground_claim`**

In `valagents/agents/grounder.py`, change the retrieval line (currently line 53):

```python
    formatted, articles = await search_articles(backend, claim.statement)
```

to:

```python
    formatted, articles, query_block = await planned_search(
        backend, claim.statement, llm, cfg, context=formal_claim.statement)
```

Then, in the `run_log.emit_candidates(...)` call (currently lines ~117–125), add the `query=query_block` field:

```python
    run_log.emit_candidates(
        claim.id, tick=tick, n_retrieved=len(articles), n_credited=independent_sources,
        contradicted=contradicted, query=query_block,
        candidates=[
            {"label": f"A{i}", "title": a.title, "url": a.url,
             "published": str(a.published)[:10], "disposition": disposition.get(f"A{i}", "uncited")}
            for i, a in enumerate(articles, start=1)
        ],
    )
```

- [ ] **Step 5: Wire `ground_novelty`**

In `valagents/agents/grounder.py`, change the novelty retrieval line (currently line 148):

```python
    formatted, _ = await search_articles(backend, formal_claim.statement)
```

to:

```python
    formatted, _, _ = await planned_search(backend, formal_claim.statement, llm, cfg)
```

- [ ] **Step 6: Run the targeted test, then the two grounder suites**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_run_log_candidates.py tests/test_grounding_support_agent.py tests/test_grounding_agent.py tests/test_scheduler_checks.py -q`
Expected: PASS — the new query-block test passes; existing grounder tests still pass (their FakeLLM yields no `ARCHIVES/TERMS` tail → rung `raw` → current behavior).

- [ ] **Step 7: Commit**

```bash
git add valagents/agents/grounder.py tests/test_run_log_candidates.py
git commit -m "Grounder: retrieve via planned_search and record the query block in the .candidates audit log"
```

---

### Task 5: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the whole suite**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS — all prior tests plus the new query-planner and grounder-wiring tests. (Pre-existing numerical-overflow `RuntimeWarning`s in the simulation suite are unrelated and expected.)

- [ ] **Step 2: If green, no commit needed.** If any pre-existing test regressed, it is almost certainly a grounder test whose `FakeLLM` now needs to tolerate the `query_planner` agent — fix by confirming that agent's body has no `ARCHIVES/TERMS` tail (so it falls to rung `raw`), not by changing production code.

---

## Self-Review

**1. Spec coverage:**
- §3 module (`PlannedQuery`, `VALID_ARCHIVES`, `plan_query` w/ context, `render_query`) → Tasks 1–2. ✓
- §4 3-rung ladder + one widen step → Task 3 (`planned_search`). ✓
- §5 render: `cat:<archive>*` no-dot wildcard, 1–2 archives OR'd, phrase-quote, pre-quote normalization, non-arXiv space-join → Task 1 `render_query` + tests. ✓
- §6 complete `VALID_ARCHIVES` (incl. eess/nlin/q-bio/q-fin/stat/econ) → Task 1 + `test_valid_archives_complete`. ✓
- §7 audit `query` block into `emit_candidates` → Task 4. ✓ (pool titles/URLs already shipped; regression-guarded.)
- §8 config `query_planner=True`, `widen_min_results=3` → Task 3 + `test_grounding_query_planner_defaults`. ✓
- §9 fail-soft ladder (rung 2 before rung 3; planner failure → raw) → Task 3 tests. ✓
- §10 testing (plan_query parse/validate/context/failsoft; render table; ladder rungs; grounder integration) → Tasks 1–4 tests. ✓
- QP-D11 context for archive inference → Task 2 (`context` param + `test_plan_query_forwards_context_into_prompt`). ✓
- QP-D12 pre-quote normalization → Task 1 (`_norm_term` + `test_render_normalizes_prequoted_term`). ✓
- Firewall / artifact.py untouched → no task touches `valagents/artifact.py`. ✓

**2. Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code; every test step shows real assertions. ✓

**3. Type consistency:** `PlannedQuery(archives, terms)`, `render_query(planned, backend, widen)`, `plan_query(text, llm, cfg, context)`, `planned_search(backend, text, llm, cfg, context) -> (str, list, dict)`, `query_block` keys `{rung, archives, terms, rendered, widened, n_hits}` — identical across Tasks 1→4. The grounder unpacks the 3-tuple in both call sites. ✓

## Execution notes (model selection)

- Tasks 1–3 are mechanical with complete code/tests in-plan → cheapest implementer tier.
- Task 4 is integration (two call sites + an existing emit call, backward-compat sensitivity) → standard tier.
- Task 5 is a verification run → cheapest tier.
