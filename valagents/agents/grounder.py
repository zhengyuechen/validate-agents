"""Grounder: per-claim literature grounding with citation metadata capture."""
from __future__ import annotations

import re

from valagents.artifact import CheckRecord, Source, Novelty, AtomicClaim, FormalClaim
from valagents.parse import checked
from valagents.prompts import GROUNDER_CLAIM, GROUNDER_NOVELTY
from valagents.agents.base import build_messages, map_support_to_verdict, as_int
from valagents.web_search import search_articles


def _parse_source_labels(sources_str: str) -> list[str]:
    """Return the raw token strings from the SOURCES field (split on comma)."""
    if sources_str.strip().lower() in ("none", ""):
        return []
    return [s.strip() for s in sources_str.split(",") if s.strip()]


def _extract_label(token: str) -> str | None:
    """Extract the [Ai] label from a token like 'A1(Smith)' or '[A1]'."""
    m = re.match(r"\[?A(\d+)\]?", token, re.IGNORECASE)
    return f"A{m.group(1)}" if m else None


async def ground_claim(
    claim: AtomicClaim, formal_claim, backend, llm, cfg, tick: int = 0
) -> CheckRecord:
    """Ground a single atomic claim in retrieved literature.

    Uses search_articles so that each cited [Ai] label can be mapped back to the
    Article that produced it, enriching the Source record with title/url/year.
    A label with no matching article keeps a bare Source(locator=token).
    D8 independence cap: independent_sources = min(LLM-reported, matched sources with real URLs).
    No retrieval → matched=0 → min(LLM,0)=0 → uncertain. The model cannot self-certify independence.
    """
    formatted, articles = await search_articles(backend, claim.statement)

    # Build a label → Article mapping: [A1] → articles[0], etc.
    label_to_article = {f"A{i}": a for i, a in enumerate(articles, start=1)}

    user = GROUNDER_CLAIM.format(
        ctype=claim.type,
        statement=claim.statement,
        articles=formatted or "(none)",
        cid=claim.id,
    )
    tail = await checked(
        "grounder",
        build_messages("You ground claims in literature.", user),
        ["CLAIM", "SUPPORT", "INDEPENDENT_SOURCES", "SOURCES", "BASIS"],
        llm=llm,
    )
    if tail is None:
        return CheckRecord(lens="grounder", verdict="uncertain", basis="(unparseable)", tick=tick)

    tokens = _parse_source_labels(tail["sources"])
    srcs = []
    for token in tokens:
        label = _extract_label(token)
        article = label_to_article.get(label) if label else None
        if article is not None:
            srcs.append(Source(
                locator=article.url,
                title=article.title,
                url=article.url,
                year=str(article.published)[:4],
                relation="independent",
            ))
        else:
            srcs.append(Source(locator=token, relation="independent"))

    # D8 independence cap (unconditional): the model can downgrade but never claim MORE
    # independence than the count of matched sources that carry a real URL.
    # With no backend / no retrieved articles, matched_independent=0 → min(llm,0)=0
    # → map_support_to_verdict("supported", 0) → "uncertain".
    # This is intentional: no real external retrieval ⇒ a claim cannot reach "pass".
    llm_independent = as_int(tail["independent_sources"])
    matched_independent = sum(1 for s in srcs if s.relation == "independent" and s.url)
    independent_sources = min(llm_independent, matched_independent)
    verdict = map_support_to_verdict(tail["support"], independent_sources)

    return CheckRecord(
        lens="grounder",
        verdict=verdict,
        basis=tail["basis"],
        sources=srcs,
        independent_sources=independent_sources,
        tick=tick,
    )


async def ground_novelty(formal_claim: FormalClaim, backend, llm, cfg) -> Novelty | None:
    """Position a formal claim against closest prior art and return the novelty delta."""
    formatted, _ = await search_articles(backend, formal_claim.statement)
    user = GROUNDER_NOVELTY.format(formal=formal_claim.statement, articles=formatted or "(none)")
    tail = await checked(
        "grounder",
        build_messages("You position claims against prior art.", user),
        ["CLOSEST_PRIOR", "DELTA", "POSITION"],
        llm=llm,
    )
    if tail is None:
        return None
    return Novelty(
        closest_prior=[tail["closest_prior"]],
        delta=tail["delta"],
        position=tail["position"].strip().lower(),
    )
