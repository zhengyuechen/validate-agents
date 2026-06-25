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
