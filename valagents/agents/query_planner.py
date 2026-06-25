"""Query planner: an LLM proposes a focused arXiv query (1–2 archives + 2–4 distinctive terms);
code validates, renders, and retrieves. Fixes off-domain retrieval — a long generic-physics claim
sentence is a better keyword match for the high-volume hep-ex/gr-qc corpus than for the real domain,
so unscoped relevance ranks particle/GW papers first. Same firewall as the designers: model proposes
a query, code adjudicates; artifact.py and the gate are untouched."""
from __future__ import annotations

from dataclasses import dataclass, field

from valagents.web_search import backend_label, search_articles
from valagents.prompts import QUERY_PLANNER
from valagents.agents.base import build_messages, split_list
from valagents.parse import checked


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
    """Render the backend-specific search string. The ladder only calls this with non-empty terms;
    guard the precondition so a future direct caller gets "" rather than a malformed 'AND ()' clause."""
    if not planned.terms:
        return ""
    if backend_label(backend) != "arxiv":
        # non-arXiv (e.g. Tavily): focused terms as a natural query — no cat:/Lucene operators.
        return " ".join(t.strip().strip('"').strip() for t in planned.terms)
    op = " OR " if widen else " AND "
    term_q = "(" + op.join(_norm_term(t) for t in planned.terms) + ")"
    if planned.archives:
        cat_q = "(" + " OR ".join(f"cat:{a}*" for a in planned.archives) + ")"   # wildcard MANDATORY
        return f"{cat_q} AND {term_q}"
    return term_q                                                                # terms-only (rung 2)


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
