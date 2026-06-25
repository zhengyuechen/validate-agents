"""CiteAudit — verify the LLM-named narrative references (closest_prior, must_cite) against real
catalogued records by deterministic title-match. Output integrity only; never feeds the gate (CA-D1).
Network proposes candidates; _title_match (pure code) adjudicates."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from valagents.grounding import _content_tokens, _norm
from valagents.references import Reference, normalize_id

log = logging.getLogger(__name__)


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
    # web_search.Article carries no authors (the authors caveat) -> authors=[]
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
