"""Grounder: per-claim literature grounding with citation metadata capture."""
from __future__ import annotations

import re

from valagents.artifact import CheckRecord, Source, Novelty, AtomicClaim, FormalClaim
from valagents.parse import checked, checked_body
from valagents.prompts import GROUNDER_CLAIM, GROUNDER_NOVELTY
from valagents.agents.base import build_messages, map_support_to_verdict, as_int, choice
from valagents.agents.value_grounder import _extract_json
from valagents.grounding import (
    _quote_admissible, _support_quote_valid, _retrieval_saturated_tokens, _norm,
)
from valagents.agents.query_planner import planned_search
from valagents import references, run_log


def _extract_label(token: str) -> str | None:
    """Extract the [Ai] label from a token like 'A1(Smith)' or '[A1]'."""
    m = re.match(r"\[?A(\d+)\]?", token, re.IGNORECASE)
    return f"A{m.group(1)}" if m else None


def _dedup_articles(articles: list) -> list:
    """§7: collapse to distinct works. Key on normalize_id(url) for recognized arXiv/DOI ids, else the
    normalized title. Order-preserving. Protects the deferred ≥2 bar from per-version double counting."""
    seen: set[str] = set()
    out: list = []
    for a in articles:
        key = references.normalize_id(a.url) if references.detect_kind(a.url) != "unknown" else _norm(a.title)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


async def ground_claim(
    claim: AtomicClaim, formal_claim, backend, llm, cfg, tick: int = 0
) -> CheckRecord:
    """Ground a single atomic claim against retrieved literature (Tier-2: code-witnessed support).

    The LLM emits a SUPPORT/INDEPENDENT_SOURCES/BASIS tail PLUS a citations JSON (per-source
    {label, direction, quote}) — and NOTHING ELSE the floor depends on. Pure code (grounding.py) adjudicates
    each quote: anti-fabrication + sentence-bound + substantial (both directions), plus a supports-only
    on-property floor whose distinctive token set is derived in CODE from the CLAIM minus the retrieval-
    saturated subject (T2-D12 — the model cannot choose the set, so it cannot launder credit via a property
    declaration). The credited count is the number of DISTINCT retrieved works carrying a passing SUPPORTS
    quote, capped by both retrieval and the model's self-report. A passing CONTRADICTS quote forces a
    pass→uncertain downgrade. Code witnesses presence + on-property topicality + anti-fabrication — NOT
    entailment, NOT independence (the direction is the model's loud label). artifact.py is untouched.
    """
    formatted, articles, query_block = await planned_search(
        backend, claim.statement, llm, cfg, context=formal_claim.statement)
    label_to_article = {f"A{i}": a for i, a in enumerate(articles, start=1)}

    user = GROUNDER_CLAIM.format(
        ctype=claim.type, statement=claim.statement, articles=formatted or "(none)", cid=claim.id,
    )
    tail, body = await checked_body(
        "grounder",
        build_messages("You ground claims in literature.", user),
        ["CLAIM", "SUPPORT", "INDEPENDENT_SOURCES", "BASIS"],
        llm=llm,
    )
    if tail is None:
        return CheckRecord(lens="grounder", verdict="uncertain", basis="(unparseable)", tick=tick)

    g = cfg.grounding
    data = _extract_json(body) or {}
    raw_citations = data.get("citations")
    citations = raw_citations if isinstance(raw_citations, list) else []

    # §5 (T2-D12): the subject subtractor is the CODE-derived retrieval-saturated set ONLY — model-independent.
    # The distinctive property set (claim − subject) is computed inside _support_quote_valid from claim.statement.
    subject_tokens = _retrieval_saturated_tokens(articles, g.subject_saturation_frac)

    passing: list = []
    contradicted = False
    contradiction_quote = ""
    disposition: dict[str, str] = {}                                 # per-label audit: what happened to each cited article
    for c in citations:
        if not isinstance(c, dict):
            continue
        label = _extract_label(str(c.get("label", "")))
        art = label_to_article.get(label) if label else None
        if art is None:
            continue
        quote = str(c.get("quote", ""))
        direction = str(c.get("direction", "")).strip().lower()
        if direction == "contradicts":
            if _quote_admissible(quote, art.summary, g.quote_min_tokens):   # §6: admissible only, NO property floor
                contradicted = True
                if not contradiction_quote:
                    contradiction_quote = quote
                disposition[label] = "contradicts"
            else:
                disposition.setdefault(label, "contradicts_unverified")
        elif direction == "supports":
            if _support_quote_valid(quote, art.summary, claim.statement,
                                    subject_tokens, g.quote_min_tokens):
                passing.append(art)
                disposition[label] = "credited"
            else:
                disposition.setdefault(label, "quote_failed")

    deduped = _dedup_articles(passing)
    code_witnessed = min(len(deduped), len(articles))               # §7 code cap (cannot exceed retrieval)
    independent_sources = min(as_int(tail["independent_sources"]), code_witnessed)   # model may downgrade, never inflate
    verdict = map_support_to_verdict(tail["support"], independent_sources)
    if contradicted and verdict == "pass":                          # §6 contradiction guard (force-downgrade)
        verdict = "uncertain"

    # Audit trail (run_log .candidates, fail-soft): the FULL retrieved pool + each article's disposition —
    # what the grounder SURVEYED vs cited, incl. rejected (quote_failed) and contradicting papers. The
    # retrieved pool otherwise lived only in the prompt at runtime and was unrecoverable. Diagnostic only;
    # the gate never reads this.
    run_log.emit_candidates(
        claim.id, tick=tick, n_retrieved=len(articles), n_credited=independent_sources,
        contradicted=contradicted, query=query_block,
        candidates=[
            {"label": f"A{i}", "title": a.title, "url": a.url,
             "published": str(a.published)[:10], "disposition": disposition.get(f"A{i}", "uncited")}
            for i, a in enumerate(articles, start=1)
        ],
    )

    srcs = [Source(locator=a.url, title=a.title, url=a.url, year=str(a.published)[:4], relation="independent")
            for a in deduped]
    basis = tail["basis"]
    if independent_sources >= 1:
        # §8 honest boundary IN THE ARTIFACT A HUMAN READS: the credited count is presence + on-property
        # topicality, NOT entailment, NOT independence. The field name (`independent_sources`) and
        # relation="independent" are kept for gate-compat, so the basis is the only place this is said.
        basis = (f"{basis} [grounder credit: {independent_sources} retrieved source(s) carrying a "
                 f"code-witnessed verbatim on-property passage; entailment & independence are the model's "
                 f"label, not code-witnessed]")
    if contradicted:
        basis = f"CONTRADICTION: {contradiction_quote} — {basis}"   # prefix kept FIRST (math-claim handling unchanged)

    return CheckRecord(
        lens="grounder", verdict=verdict, basis=basis, sources=srcs,
        independent_sources=independent_sources, tick=tick,
    )


async def ground_novelty(formal_claim: FormalClaim, backend, llm, cfg) -> Novelty | None:
    """Position a formal claim against closest prior art and return the novelty delta."""
    formatted, _, _ = await planned_search(backend, formal_claim.statement, llm, cfg)
    user = GROUNDER_NOVELTY.format(formal=formal_claim.statement, articles=formatted or "(none)")
    tail = await checked(
        "grounder",
        build_messages("You position claims against prior art.", user),
        ["CLOSEST_PRIOR", "DELTA", "POSITION"],
        llm=llm,
    )
    if tail is None:
        return None
    position = choice(tail["position"], {"new", "special_case", "restatement"})
    if position is None:
        return None
    return Novelty(
        closest_prior=[tail["closest_prior"]],
        delta=tail["delta"],
        position=position,
    )
