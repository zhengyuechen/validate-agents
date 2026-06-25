# CiteAudit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the LLM-named narrative references (`prior_art_positioning.closest_prior` + `must_cite`) against real catalogued records by deterministic title-match, so the report annotates each as a resolved citation or `[unverified]` instead of printing an unverified model claim as fact.

**Architecture:** A new `valagents/citeaudit.py` with a pure-code `_title_match` (require-ALL + min-token gate) and a pure Crossref parser (Task 1); a `CiteAuditor` that resolves a name via arXiv then Crossref, first match wins, fail-soft (Task 2); and CLI wiring that merges resolved refs into the bibliography (existing-wins collision rule) and annotates the report (Task 3). Network *proposes* candidates; pure code *adjudicates*. Output integrity only — never touches the gate.

**Tech Stack:** Python 3, Pydantic v2, pytest (asyncio_mode=auto), conda env `cosci-reproduce`, `httpx` (already a dep), `arxiv` (already a dep via `web_search.ArxivBackend`). Reuses `grounding._content_tokens`/`_norm` and the `references.py` resolver/bibliography machinery.

## Global Constraints

- **NEVER modify `valagents/artifact.py`.** CiteAudit is output-integrity only; it never feeds `internally_validated`/`independent_sources`/any verdict.
- **Commits: plain messages, NO attribution trailer** — no `Co-Authored-By`, no `Claude-Session`, no "Generated with Claude". Audit each commit message.
- **Pure helpers take primitives** (`min_name_tokens: int`), not `cfg` — consistent with `grounding._quote_valid`/`_support_quote_valid`. The `CiteAuditor` unpacks `cfg.citeaudit.*`.
- **`citeauditor=None` = OFF** → `run_cli` produces a report + bib **byte-identical** to today. This is a regression pin.
- **Reuse `grounding._content_tokens` / `_norm`** for tokenization (NFKC + casefold + the existing `_STOP`).
- **Merge collision rule (CA-D8): EXISTING WINS** — `by_locator.setdefault(...)`, never overwrite (would blank a claim-cited ref's `cited_by` and corrupt its `origin`).
- **Match config defaults:** `CiteAuditCfg.min_name_tokens = 3`, `arxiv_rows = 5`, `crossref_rows = 5`.
- **Test command:** `conda run -n cosci-reproduce python -m pytest tests/ -q` (full); focused: `conda run -n cosci-reproduce python -m pytest tests/test_citeaudit.py -q`.
- **Spec:** `docs/2026-06-25-validate-agents-citeaudit-design.md` (decision log CA-D1..D8).

---

### Task 1: Pure match + Crossref parser + config + `Reference.origin`

**Files:**
- Create: `valagents/citeaudit.py`
- Modify: `valagents/config.py` (add `CiteAuditCfg`; add `citeaudit` field to `Config`, around lines 8–46)
- Modify: `valagents/references.py:23` (`Reference.origin` Literal)
- Test: `tests/test_citeaudit.py` (create)

**Interfaces:**
- Consumes: `grounding._content_tokens` (existing).
- Produces:
  - `_Candidate` dataclass `{title: str, authors: list[str], year: str, url: str}`
  - `_title_match(name: str, candidate_title: str, min_name_tokens: int) -> bool`
  - `_crossref_candidates(data: dict) -> list[_Candidate]`
  - `config.CiteAuditCfg{min_name_tokens=3, arxiv_rows=5, crossref_rows=5}`; `Config.citeaudit`
  - `references.Reference.origin` now `Literal["provided", "retrieved", "asserted"]`

- [ ] **Step 1: Add the config knobs**

In `valagents/config.py`, add after `GroundCfg` (after line 13):

```python
class CiteAuditCfg(BaseModel):
    min_name_tokens: int = 3   # a narrative name needs >= this many content tokens to be "title-like" (CA-D5)
    arxiv_rows: int = 5        # candidates fetched per arXiv title-search
    crossref_rows: int = 5     # candidates fetched per Crossref title-search
```

In the `Config` class (after `grounding: GroundCfg = GroundCfg()`, line 43), add:

```python
    citeaudit: CiteAuditCfg = CiteAuditCfg()
```

- [ ] **Step 2: Extend `Reference.origin`**

In `valagents/references.py`, line 23, change:

```python
    origin: Literal["provided", "retrieved"] = "retrieved"
```
to:
```python
    origin: Literal["provided", "retrieved", "asserted"] = "retrieved"
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_citeaudit.py`:

```python
from valagents.citeaudit import _title_match, _crossref_candidates, _Candidate


def test_title_match_requires_all_tokens():
    # name content-tokens (4) all present in the candidate title -> match
    assert _title_match("phase space quantum mechanics", "Quantum Mechanics in Phase Space", 3) is True


def test_title_match_min_token_gate():
    # short names (2 content tokens) are not title-like -> never attach, regardless of candidate
    assert _title_match("Born reciprocity", "Born Reciprocity and Reciprocal Relativity", 3) is False
    assert _title_match("prior model", "A Prior Model of Everything", 3) is False


def test_title_match_false_attach_rejected():
    # a name token absent from the candidate title -> no match (the harm we prevent)
    assert _title_match("spin liquid noise spectroscopy", "Spin Liquid Magnetization Study", 3) is False


def test_title_match_generic_overspecification_is_accepted_residual():
    # CA-D5: a generic 3-token name DOES match an arbitrary same-token title (real paper, human-checkable)
    assert _title_match("spin liquid model", "A New Spin Liquid Model Hamiltonian", 3) is True


def test_title_match_knob_4_kills_generic():
    # raising min_name_tokens to 4 demotes the 3-token generic name to unverified
    assert _title_match("spin liquid model", "A New Spin Liquid Model Hamiltonian", 4) is False


def test_crossref_candidates_parse():
    data = {"message": {"items": [
        {"title": ["Quantum Mechanics in Phase Space"],
         "author": [{"given": "C.", "family": "Zachos"}, {"given": "D.", "family": "Fairlie"}],
         "published-print": {"date-parts": [[2005]]}, "DOI": "10.1142/5287"},
        {"title": [], "author": []},  # titleless item -> skipped
    ]}}
    cands = _crossref_candidates(data)
    assert len(cands) == 1
    assert cands[0].title == "Quantum Mechanics in Phase Space"
    assert cands[0].authors == ["C. Zachos", "D. Fairlie"]
    assert cands[0].year == "2005"
    assert cands[0].url == "https://doi.org/10.1142/5287"


def test_crossref_candidates_empty():
    assert _crossref_candidates({}) == []
    assert _crossref_candidates({"message": {"items": []}}) == []
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_citeaudit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'valagents.citeaudit'`.

- [ ] **Step 5: Create `valagents/citeaudit.py` (the pure core)**

```python
"""CiteAudit — verify the LLM-named narrative references (closest_prior, must_cite) against real
catalogued records by deterministic title-match. Output integrity only; never feeds the gate (CA-D1).
Network proposes candidates; _title_match (pure code) adjudicates."""
from __future__ import annotations

from dataclasses import dataclass, field

from valagents.grounding import _content_tokens


@dataclass
class _Candidate:
    title: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    url: str = ""


def _title_match(name: str, candidate_title: str, min_name_tokens: int) -> bool:
    """High-precision deterministic match (CA-D5): the name must be 'title-like' (>= min_name_tokens
    content tokens) AND every content token of the name must appear in the candidate's title
    (require-ALL). Reuses grounding._content_tokens (NFKC + casefold + _STOP). No LLM.
    A wrong-paper attach would need a title carrying *every* name token — near-impossible."""
    name_tokens = _content_tokens(name)
    if len(name_tokens) < min_name_tokens:
        return False
    return name_tokens <= _content_tokens(candidate_title)


def _crossref_candidates(data: dict) -> list[_Candidate]:
    """Pure parser for a Crossref /works response: message.items[*] -> _Candidate. Titleless items
    are skipped. Mirrors the field handling in references.DoiResolver."""
    out: list[_Candidate] = []
    for item in (data.get("message", {}).get("items") or []):
        titles = item.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            continue
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in (item.get("author") or [])
        ]
        authors = [a for a in authors if a]
        published = item.get("published-print") or item.get("published-online") or {}
        parts = published.get("date-parts", [[""]])
        year = str(parts[0][0]) if parts and parts[0] else ""
        doi = item.get("DOI", "")
        url = f"https://doi.org/{doi}" if doi else item.get("URL", "")
        out.append(_Candidate(title=title, authors=authors, year=year, url=url))
    return out
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_citeaudit.py -q`
Expected: PASS (7 passed).

- [ ] **Step 7: Run the touched-area suites (no regression from the config/Reference change)**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_references.py -q`
Expected: PASS (the `origin` Literal widened; existing values still valid).

- [ ] **Step 8: Commit**

```bash
git add valagents/citeaudit.py valagents/config.py valagents/references.py tests/test_citeaudit.py
git commit -m "citeaudit: pure title-match + crossref parser + CiteAuditCfg + Reference origin=asserted"
```

---

### Task 2: `CiteAuditor` + `audit_narrative_refs`

**Files:**
- Modify: `valagents/citeaudit.py` (add network search, `CiteResult`, `CiteAuditor`, `audit_narrative_refs`)
- Test: `tests/test_citeaudit.py` (extend)

**Interfaces:**
- Consumes: `_title_match`, `_crossref_candidates`, `_Candidate` (Task 1); `references.Reference`, `references.normalize_id`; `grounding._norm`; `web_search.ArxivBackend`-shaped object (anything with `async search(query, max_results) -> list[Article]`, `Article` having `.title/.url/.published`).
- Produces:
  - `CiteResult{name: str, status: str, reference: Reference | None}`
  - `async _crossref_title_search(name: str, rows: int) -> list[_Candidate]`
  - `CiteAuditor(arxiv_backend, crossref_search=_crossref_title_search, cfg=None)` with `async audit(name) -> CiteResult`
  - `async audit_narrative_refs(art, auditor) -> dict[str, CiteResult]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_citeaudit.py`:

```python
import pytest
from valagents.citeaudit import CiteResult, CiteAuditor, audit_narrative_refs
from valagents.config import Config
from valagents.artifact import IdeaArtifact, PriorArtPositioning


def _cfg():
    return Config(default_model="fake")


class _Art:  # minimal arxiv Article stand-in (title/url/published)
    def __init__(self, title, url="http://arxiv.org/abs/2501.00001v1", published="2025-01-01"):
        self.title = title
        self.url = url
        self.published = published
        self.summary = ""


class _FakeArxiv:
    def __init__(self, arts, raises=False):
        self._arts = arts
        self._raises = raises

    async def search(self, query, max_results=5):
        if self._raises:
            raise RuntimeError("network down")
        return list(self._arts)


async def _crossref_none(name, rows):
    return []


def _crossref_with(cands):
    async def _search(name, rows):
        return list(cands)
    return _search


async def test_audit_resolves_via_arxiv():
    arx = _FakeArxiv([_Art("Quantum Mechanics in Phase Space", url="http://arxiv.org/abs/hep-th/0110114")])
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "resolved"
    assert r.reference.origin == "asserted"
    assert r.reference.title == "Quantum Mechanics in Phase Space"
    assert r.reference.locator == "arxiv:hep-th/0110114" or r.reference.locator.startswith("arxiv:")


async def test_audit_falls_back_to_crossref():
    from valagents.citeaudit import _Candidate
    arx = _FakeArxiv([])  # arXiv finds nothing
    cross = _crossref_with([_Candidate("Quantum Mechanics in Phase Space", ["C. Zachos"], "2005",
                                       "https://doi.org/10.1142/5287")])
    auditor = CiteAuditor(arx, cross, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "resolved" and r.reference.locator == "10.1142/5287"
    assert r.reference.authors == ["C. Zachos"]


async def test_audit_unverified_when_no_match():
    arx = _FakeArxiv([_Art("An Unrelated Paper About Something Else Entirely")])
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "unverified" and r.reference is None


async def test_audit_short_name_skips_search():
    arx = _FakeArxiv([_Art("Born Reciprocity and Reciprocal Relativity")], raises=True)  # would raise if searched
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    r = await auditor.audit("Born reciprocity")  # 2 content tokens < 3 -> no search, unverified
    assert r.status == "unverified"


async def test_audit_fail_soft_on_arxiv_error():
    arx = _FakeArxiv([], raises=True)  # arXiv raises -> swallowed
    cross = _crossref_with([])
    auditor = CiteAuditor(arx, cross, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "unverified"  # no crash


async def test_audit_narrative_refs_scope_and_dedup():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(
        closest_prior="phase space quantum mechanics",
        must_cite=["phase space quantum mechanics", "spin liquid noise spectroscopy"],  # 1st dup of closest_prior
    ))
    arx = _FakeArxiv([_Art("Quantum Mechanics in Phase Space")])
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    out = await audit_narrative_refs(art, auditor)
    assert set(out) == {"phase space quantum mechanics", "spin liquid noise spectroscopy"}
    assert out["phase space quantum mechanics"].status == "resolved"


async def test_audit_narrative_refs_off_returns_empty():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(closest_prior="x y z"))
    assert await audit_narrative_refs(art, None) == {}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_citeaudit.py -q`
Expected: FAIL with `ImportError: cannot import name 'CiteResult'`.

- [ ] **Step 3: Implement the auditor**

Append to `valagents/citeaudit.py`. Add these imports at the top (with the existing `from valagents.grounding import _content_tokens`):

```python
import logging

from valagents.grounding import _content_tokens, _norm
from valagents.references import Reference, normalize_id

log = logging.getLogger(__name__)
```

(Replace the existing single grounding import line with the combined one above.) Then add:

```python
@dataclass
class CiteResult:
    name: str
    status: str                      # "resolved" | "unverified"
    reference: Reference | None = None


async def _crossref_title_search(name: str, rows: int) -> list[_Candidate]:
    """Network: Crossref bibliographic title search. Fail-soft -> [] on any error."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": name, "rows": rows},
                timeout=15,
            )
            resp.raise_for_status()
            return _crossref_candidates(resp.json())
    except Exception as exc:
        log.warning("crossref title search failed (%s)", exc)
        return []


def _article_to_candidate(a) -> _Candidate:
    # web_search.Article carries no authors (CA: authors caveat) -> authors=[]
    return _Candidate(title=a.title, authors=[], year=str(a.published)[:4], url=a.url)


def _candidate_to_ref(c: _Candidate) -> Reference:
    return Reference(
        locator=normalize_id(c.url) if c.url else normalize_id(c.title),
        title=c.title, authors=c.authors, year=c.year, url=c.url, origin="asserted",
    )


class CiteAuditor:
    """Injected dependency; `None` at the call site -> CiteAudit OFF. Resolves a narrative name to a
    real record by deterministic title-match: arXiv first, then Crossref, first match wins. Network
    proposes candidates; _title_match adjudicates. Fail-soft: a backend error yields no candidate."""

    def __init__(self, arxiv_backend, crossref_search=_crossref_title_search, cfg=None):
        self._arxiv = arxiv_backend
        self._crossref = crossref_search
        self._cfg = cfg

    async def audit(self, name: str) -> CiteResult:
        ca = self._cfg.citeaudit
        if len(_content_tokens(name)) < ca.min_name_tokens:     # not title-like -> skip the search entirely
            return CiteResult(name, "unverified")
        try:
            arts = await self._arxiv.search(name, max_results=ca.arxiv_rows)
        except Exception as exc:
            log.warning("arxiv title search failed (%s)", exc)
            arts = []
        for a in arts:
            if _title_match(name, a.title, ca.min_name_tokens):
                return CiteResult(name, "resolved", _candidate_to_ref(_article_to_candidate(a)))
        for c in await self._crossref(name, ca.crossref_rows):
            if _title_match(name, c.title, ca.min_name_tokens):
                return CiteResult(name, "resolved", _candidate_to_ref(c))
        return CiteResult(name, "unverified")


async def audit_narrative_refs(art, auditor) -> dict[str, CiteResult]:
    """Collect the in-scope narrative names (closest_prior + must_cite; NOT nearest_theories — CA-D2),
    dedup the audit CALL by normalized name, return {original_name -> CiteResult}. {} when auditor is None."""
    if auditor is None:
        return {}
    pos = getattr(art, "prior_art_positioning", None)
    names: list[str] = []
    if pos:
        if pos.closest_prior.strip():
            names.append(pos.closest_prior)
        names.extend(m for m in pos.must_cite if m.strip())
    seen: dict[str, CiteResult] = {}
    out: dict[str, CiteResult] = {}
    for name in names:
        key = _norm(name)
        if key not in seen:
            seen[key] = await auditor.audit(name)
        out[name] = seen[key]
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_citeaudit.py -q`
Expected: PASS (14 passed — 7 from Task 1 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add valagents/citeaudit.py tests/test_citeaudit.py
git commit -m "citeaudit: CiteAuditor (arxiv->crossref first-match-wins, fail-soft) + audit_narrative_refs"
```

---

### Task 3: CLI wiring + bibliography merge + render annotation

**Files:**
- Modify: `valagents/references.py` (`build_references` gains `asserted_refs=`; setdefault merge)
- Modify: `valagents/cli.py` (`run_cli` `citeauditor=`; `render_report`/`_render_supporting_layer` markers + gloss; `main()` builds a live auditor)
- Test: `tests/test_citeaudit_cli.py` (create)

**Interfaces:**
- Consumes: `audit_narrative_refs`, `CiteAuditor` (Task 2); `references.build_references`, `normalize_id`.
- Produces: `build_references(artifact, provided_path=None, resolver=None, asserted_refs=None)`; `render_report(art, refs=None, audit_map=None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_citeaudit_cli.py`:

```python
import pytest
from valagents.references import Reference, build_references, normalize_id
from valagents.artifact import IdeaArtifact, AtomicClaim, CheckRecord, Source, PriorArtPositioning
from valagents.citeaudit import CiteResult
from valagents.cli import render_report


def _resolved_ref(locator, title, url):
    return Reference(locator=normalize_id(locator), title=title, url=url, year="2005",
                     authors=["A. Author"], origin="asserted")


async def test_build_references_merges_asserted_fresh_locator():
    art = IdeaArtifact(raw_idea="s")
    asserted = [_resolved_ref("https://doi.org/10.1142/5287", "Quantum Mechanics in Phase Space",
                              "https://doi.org/10.1142/5287")]
    refs = await build_references(art, asserted_refs=asserted)
    assert len(refs) == 1 and refs[0].origin == "asserted" and refs[0].number == 1


async def test_build_references_collision_existing_wins():
    # a claim-cited (retrieved) ref AND an asserted ref share a locator -> ONE entry, retrieved kept
    art = IdeaArtifact(raw_idea="s", claim_graph=[AtomicClaim(
        id="c1", statement="s", type="empirical",
        checks=[CheckRecord(lens="grounder", verdict="pass", independent_sources=1,
                            sources=[Source(locator="arxiv:2501.00001", title="Retrieved Title",
                                            url="http://arxiv.org/abs/2501.00001", relation="independent")])])])
    asserted = [_resolved_ref("http://arxiv.org/abs/2501.00001v2", "Asserted Title",
                              "http://arxiv.org/abs/2501.00001v2")]
    refs = await build_references(art, asserted_refs=asserted)
    assert len(refs) == 1
    r = refs[0]
    assert r.origin == "retrieved"          # existing wins (CA-D8)
    assert r.cited_by == ["c1"]             # preserved, not blanked
    assert r.title == "Retrieved Title"     # not clobbered by the asserted ref


def test_render_annotates_resolved_and_unverified_with_gloss():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(
        closest_prior="phase space quantum mechanics", must_cite=["made up nonexistent theorem paper"]))
    ref = _resolved_ref("https://doi.org/10.1142/5287", "Quantum Mechanics in Phase Space",
                        "https://doi.org/10.1142/5287")
    ref.number = 1
    audit_map = {
        "phase space quantum mechanics": CiteResult("phase space quantum mechanics", "resolved", ref),
        "made up nonexistent theorem paper": CiteResult("made up nonexistent theorem paper", "unverified"),
    }
    report = render_report(art, [ref], audit_map)
    assert "Quantum Mechanics in Phase Space" in report and "[1]" in report   # resolved -> loud title + [n]
    assert "[unverified]" in report
    assert "not resolved to a catalogued record" in report                    # the gloss


def test_render_off_no_markers():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(closest_prior="phase space qm"))
    report = render_report(art, [], None)   # audit_map None -> off
    assert "[unverified]" not in report and "not resolved to a catalogued record" not in report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_citeaudit_cli.py -q`
Expected: FAIL (`build_references` has no `asserted_refs` kwarg; `render_report` ignores `audit_map`).

- [ ] **Step 3: Add the `asserted_refs` merge to `build_references`**

In `valagents/references.py`, replace `build_references` (lines 199–212) with:

```python
async def build_references(artifact, provided_path=None, resolver: Resolver | None = None,
                          asserted_refs: list[Reference] | None = None) -> list[Reference]:
    by_locator = {ref.locator: ref for ref in collect_retrieved(artifact)}
    if provided_path:
        resolver = resolver or DefaultResolver()
        for provided in await load_provided(provided_path, resolver):
            if provided.locator in by_locator:
                provided.cited_by = by_locator[provided.locator].cited_by
            by_locator[provided.locator] = provided
    for asserted in (asserted_refs or []):                 # CA-D8: existing wins, never overwrite
        by_locator.setdefault(normalize_id(asserted.locator), asserted)

    refs = sorted(by_locator.values(), key=lambda ref: (0 if ref.cited_by else 1, ref.locator))
    for number, ref in enumerate(refs, start=1):
        ref.number = number
        ref.key = _bibtex_key(ref, number)
    return refs
```

- [ ] **Step 4: Add the render annotation + gloss to `cli.py`**

In `valagents/cli.py`, add this helper above `_render_supporting_layer` (before line 164):

```python
_UNVERIFIED_GLOSS = "_`[unverified]` = not resolved to a catalogued record; not a claim of fabrication._"


def _cite_marker(name: str, audit_map: dict, num_by_loc: dict) -> str:
    """Inline marker for a narrative reference name. '' when CiteAudit is off or the name isn't in scope."""
    from valagents.references import normalize_id
    res = (audit_map or {}).get(name)
    if res is None:
        return ""
    if res.status == "resolved" and res.reference is not None:
        ref = res.reference
        n = num_by_loc.get(normalize_id(ref.locator))
        who = f"{ref.authors[0]} et al. " if ref.authors else ""
        marker = f" — {ref.title}, {who}{ref.year} ({ref.url}) ✓"
        return marker + (f" [{n}]" if n else "")
    return " [unverified]"
```

Change `_render_supporting_layer(lines, art)` (line 164) to `_render_supporting_layer(lines, art, audit_map, num_by_loc)`. In its `prior_art_positioning` block (lines 198–208), change the `closest_prior` and `must_cite` lines to annotate:

```python
    if art.prior_art_positioning:
        pos = art.prior_art_positioning
        lines += [
            "## Prior-Art Positioning",
            f"**Closest prior:** {pos.closest_prior}{_cite_marker(pos.closest_prior, audit_map, num_by_loc)}",
            f"**Similarity:** {pos.similarity}",
            f"**Difference:** {pos.difference}",
            f"**What is new:** {pos.what_is_new}",
            "**Must cite/discuss:** " + (
                ", ".join(m + _cite_marker(m, audit_map, num_by_loc) for m in pos.must_cite) or "none"),
            "",
        ]
        if audit_map and any(r.status == "unverified" for r in audit_map.values()):
            lines += [_UNVERIFIED_GLOSS, ""]
```

(The `theory_bridge.nearest_theories` line at 190 is UNCHANGED — CA-D2.)

In `render_report` (line 228), change the signature to `def render_report(art, refs=None, audit_map=None) -> str:` and, where it calls `_render_supporting_layer(lines, art)`, build the locator→number map and pass both:

```python
    num_by_loc = {normalize_id(r.locator): r.number for r in refs}
    _render_supporting_layer(lines, art, audit_map, num_by_loc)
```

Ensure `normalize_id` is imported in `cli.py` (add `normalize_id` to the `from valagents.references import (...)` block near line 13 if not already present).

- [ ] **Step 5: Wire `run_cli` + `main()`**

In `valagents/cli.py` `run_cli` (the `art = await run(...)` / `refs = await build_references(...)` region near line 277), replace those two lines with:

```python
    art = await run(seed, llm, cfg, backend=backend)
    audit_map = await audit_narrative_refs(art, citeauditor)
    asserted = [r.reference for r in audit_map.values() if r.status == "resolved" and r.reference]
    refs = await build_references(art, references_path, resolver, asserted_refs=asserted)
```

and change the `render_report(art, refs)` call to `render_report(art, refs, audit_map)`. Add `citeauditor=None` to the `run_cli` signature (after `resolver=None`). Add the imports at the top of `cli.py`:

```python
from valagents.citeaudit import CiteAuditor, audit_narrative_refs
from valagents.web_search import ArxivBackend
```

In `main()` (near line 302), build a live auditor and pass it:

```python
    citeauditor = CiteAuditor(ArxivBackend(), cfg=cfg)
```
and add `citeauditor=citeauditor,` to the `run_cli(...)` call.

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `conda run -n cosci-reproduce python -m pytest tests/test_citeaudit_cli.py -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Run the full suite (no regressions; the off-path is byte-identical)**

Run: `conda run -n cosci-reproduce python -m pytest tests/ -q`
Expected: PASS. Existing `test_cli.py` calls `render_report(art)` and `run_cli(...)` without `audit_map`/`citeauditor`; both default to off → unchanged output. If any existing test constructed `build_references` positionally past `resolver`, it still works (the new param is keyword-last).

- [ ] **Step 8: Commit**

```bash
git add valagents/references.py valagents/cli.py tests/test_citeaudit_cli.py
git commit -m "citeaudit: wire into cli — asserted_refs merge (existing-wins) + inline markers/gloss + live auditor"
```

---

## Self-Review

**1. Spec coverage** (CiteAudit design §1–§11):
- §1 scope (closest_prior + must_cite; nearest_theories NOT audited; gate untouched) → Task 2 `audit_narrative_refs` + `test_audit_narrative_refs_scope_and_dedup`; artifact.py never imported for write. ✓
- §2 match rule (require-ALL + min-token, reuse `_content_tokens`, no LLM) → Task 1 `_title_match` + tests. ✓
- §3 search (arXiv reuse + Crossref keyless; first-match-wins; fail-soft; authors caveat; cfg rows) → Task 2 `CiteAuditor`/`_crossref_title_search` + `_article_to_candidate` (authors=[]). ✓
- §4 auditor (CiteResult, injected None=off, audit_narrative_refs) → Task 2. ✓
- §5 data flow (run_cli citeauditor, merge before numbering, name→[n] by locator, inline markers, gloss, origin=asserted) → Task 3. ✓
- §5 collision EXISTING-WINS (CA-D8) → Task 3 `setdefault` + `test_build_references_collision_existing_wins`. ✓
- §6 [unverified]-common + over-specification + stoplist → documented in spec; `test_title_match_generic_overspecification_is_accepted_residual` pins the residual behavior. ✓
- §7 off/errors/determinism → `test_render_off_no_markers` + fail-soft tests + the full-suite byte-identical pin. ✓
- §8 test matrix → Tasks 1–3 cover every listed case (match, parse, resolved/unverified/crossref-fallback/fail-soft/short-name, merge dup+collision+fresh, render markers+gloss+off). ✓
- §10 CA-D1..D8 → all realized. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; every run step has the exact command + expected result. ✓

**3. Type consistency:** `_title_match(name, candidate_title, min_name_tokens)`, `_crossref_candidates(data) -> list[_Candidate]`, `CiteAuditor(arxiv_backend, crossref_search, cfg).audit(name) -> CiteResult`, `audit_narrative_refs(art, auditor) -> dict[str, CiteResult]`, `build_references(..., asserted_refs=)`, `render_report(art, refs, audit_map)` — identical at definition and call sites. `_Candidate`/`Reference` field names match `references.py`. `Article.title/.url/.published` match `web_search.py` (no authors → `_article_to_candidate` sets `authors=[]`). ✓

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-06-25-validate-agents-citeaudit.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh implementer per task, two-stage review between tasks (with the CA-D8 collision rule and the None=off byte-identical pin as the adversarial focal points), final whole-branch review.
2. **Inline execution** — execute the 3 tasks here with checkpoints.

Which approach?
