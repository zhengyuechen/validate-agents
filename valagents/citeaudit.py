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
