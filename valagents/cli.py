"""CLI: valagents "<seed>" -> IdeaArtifact JSON + markdown report."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from valagents import run_log
from valagents.citeaudit import CiteAuditor, audit_narrative_refs
from valagents.config import load_config
from valagents.llm import OpenRouterClient
from valagents.references import (
    DefaultResolver,
    build_references,
    markers_for_claim,
    normalize_id,
    to_bibtex,
)
from valagents.scheduler import run
from valagents.web_search import ArxivBackend, build_backend


LIMIT = (
    "internally_validated means 'survived the checks this system can run,' never 'true' — "
    "every lens shares the base model's blind spots."
)

_VERDICT_GLOSS = {
    "validated": "survived all internal checks — not yet peer-reviewed or experimentally confirmed",
    "refuted": "a load-bearing claim was falsified",
    "draft": "gate not yet reached; more checks needed",
    "ill_posed": "the idea cannot be settled by experiment in its current form — reframe first",
    "challenged": "a serious objection stands; idea needs repair before testing",
    "promising": "promising — a decisive check would settle it",
}

_ILL_POSED_REASON_GLOSS = {
    "not_falsifiable": "the claim is not falsifiable as stated",
    "unformalizable": "the idea could not be pinned to a precise claim",
    "unfaithful_drift": "the formalization drifted from the seed",
    "unfaithful_narrowed": "the formalization drifted from the seed",
    "ill_formed": "the decomposition was degenerate",
}


def _slug(seed: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in seed.lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return (slug[:40].strip("-") or "run")


def _reference_line(ref) -> str:
    title = ref.title or ref.locator
    authors = ", ".join(ref.authors) if ref.authors else "unknown authors"
    year = ref.year or "n.d."
    url = ref.url or ref.locator
    return f"[{ref.number}] {title} - {authors} ({year}). {url}  .{ref.origin} .{ref.relation}"


def _render_verdict_block(lines: list, art, blocker: dict, vc: str, gloss: str) -> None:
    """Block 1: headline verdict + metadata."""
    lines += [
        "# Validation Report",
        "",
        f"**Verdict:** {vc} — {gloss}",
        "",
        f"**Seed:** {art.raw_idea}",
        "",
        f"**Status:** `{art.status}`",
        f"**Load-bearing claim:** `{art.load_bearing}`",
        f"**Blocker:** {blocker.get('reason', '-')} ({blocker.get('claim_id') or '-'})",
        f"**Maturity:** {art.maturity:.2f}",
        "",
    ]
    if vc == "ill_posed":
        reason = blocker.get("reason", "")
        reason_gloss = _ILL_POSED_REASON_GLOSS.get(reason, reason)
        lines += [
            f"_This is not yet a testable claim ({reason_gloss}) — it needs reframing, not an experiment._",
            "",
        ]
    if art.formal_claim:
        lines += [
            f"**Formal claim:** {art.formal_claim.statement}",
            f"_falsifiable: {art.formal_claim.falsifiable}_",
            "",
        ]


def _render_crux_block(lines: list, art) -> None:
    """Block 2: the crux — strongest objection + decisive test, foregrounded."""
    lines.append("## The Crux")
    lines.append("")
    if art.steelman_objection:
        obj = art.steelman_objection
        lines += [
            "### Strongest objection",
            f"{obj.strongest_objection}",
            f"**What would kill it:** {obj.what_would_kill_it}",
        ]
        if obj.threatening_result:
            lines.append(f"**Threatening result:** {obj.threatening_result}")
        lines.append("")
    else:
        lines += ["### Strongest objection", "_No steelman objection produced._", ""]
    if art.validation_plan:
        plan = art.validation_plan
        lines += [
            "### Decisive test",
            f"- {plan.decisive_test}",
            f"- confirm if: {plan.confirm_if}",
            f"- refute if: {plan.refute_if}",
            f"- discriminates from: {plan.discriminates_from or 'n/a'}",
            f"- inferential standard: {plan.inferential_standard or 'n/a'}",
            f"- cost: {plan.cost}",
            "",
        ]


def _render_balanced_case(lines: list, art) -> None:
    """Block 3: case for and case against, adjacent and equal weight."""
    if art.convincing_case:
        case = art.convincing_case
        lines += [
            "### Case for",
            f"**Short version:** {case.elevator_version}",
            f"**Technical version:** {case.technical_version}",
            f"**Why existing theory leaves room:** {case.why_existing_theory_leaves_room}",
            f"**Why plausible:** {case.why_plausible}",
            f"**Skeptic tests:** {', '.join(case.skeptic_tests) or 'none'}",
            "",
        ]
    if art.steelman_objection:
        obj = art.steelman_objection
        lines += [
            "### Case against",
            f"**Strongest objection:** {obj.strongest_objection}",
            f"**Mechanism of failure:** {obj.mechanism_of_failure}",
            f"**Threatening result:** {obj.threatening_result}",
            f"**What would kill it:** {obj.what_would_kill_it}",
            f"**Fair summary:** {obj.fair_summary}",
            "",
        ]


def _render_claims_block(lines: list, art, refs: list) -> None:
    """Block 4: claim graph + limit-recovery checks."""
    if art.claim_graph:
        lines.append("## Claim Graph")
        for claim in art.claim_graph:
            markers = "".join(f"[{n}]" for n in markers_for_claim(refs, claim.id))
            marker_text = f" {markers}" if markers else ""
            lines.append(
                f"- `{claim.id}` [{claim.type}/{claim.role}] **{claim.status}**"
                f"{marker_text} - {claim.statement}"
            )
        lines.append("")
    limit_claims = [c for c in art.claim_graph if c.origin == "limit_recovery"]
    if limit_claims:
        lines.append("## Limit-recovery checks")
        for c in limit_claims:
            lines.append(f"- `{c.id}` **{c.status}** — {c.statement}")
        lines.append("")


_UNVERIFIED_GLOSS = "_`[unverified]` = not resolved to a catalogued record; not a claim of fabrication._"


def _cite_marker(name: str, audit_map: dict, num_by_loc: dict) -> str:
    """Inline marker for a narrative reference name. '' when CiteAudit is off or the name isn't in scope."""
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


def _render_supporting_layer(lines: list, art, audit_map, num_by_loc) -> None:
    """Block 5: completion, theory bridge, positioning, known limits, predictions."""
    if art.completion:
        lines += [
            "## Completed Candidate",
            f"**Completion:** `{art.completion.status}`",
            "",
            art.completion.completed_idea,
            "",
            f"**Mechanism:** {art.completion.mechanism}",
            f"**Weakest link:** {art.completion.weakest_link}",
            "",
        ]
        if art.completion.assumptions:
            lines.append("### Assumptions")
            for assumption in art.completion.assumptions:
                if assumption.status == "novel_load_bearing":
                    lines.append(f"- ⚠ rests on a novel, load-bearing assumption: {assumption.text}")
                else:
                    lines.append(f"- [{assumption.status}] {assumption.text}")
            lines.append("")
    if art.theory_bridge:
        bridge = art.theory_bridge
        lines += [
            "## Theory Bridge",
            f"**Family:** {bridge.theory_family}",
            f"**Nearest theories:** {', '.join(bridge.nearest_theories) or 'none'}",
            f"**Extends:** {bridge.extends}",
            f"**Challenges:** {bridge.challenges}",
            f"**Known limits to recover:** {bridge.recovers_known_limits}",
            f"**Departure point:** {bridge.departure_point}",
            f"**Expert translation:** {bridge.expert_translation}",
            "",
        ]
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
    if art.known_limits:
        lines.append("## Known Limits")
        for item in art.known_limits:
            lines.append(
                f"- **{item.limit}** — recovered: `{item.recovered}`; "
                f"failure if not: {item.failure_if_not}; repair: {item.repair_needed}"
            )
        lines.append("")
    if art.predictions:
        lines.append("## Predictions")
        for pred in art.predictions:
            lines.append(
                f"- {pred.observable} (effect: {pred.effect_size}; "
                f"discriminates from: {pred.discriminates_from or 'n/a'}; "
                f"measurable: {pred.measurable}; detectable: {pred.detectable})"
            )
        lines.append("")


def render_report(art, refs=None, audit_map=None) -> str:
    refs = refs or []
    num_by_loc = {normalize_id(r.locator): r.number for r in refs}
    blocker = art.blocker or {}
    vc = art.verdict_class
    gloss = _VERDICT_GLOSS.get(vc, vc)
    lines: list[str] = []

    # 1. Verdict block (headline)
    _render_verdict_block(lines, art, blocker, vc, gloss)

    # 2. The crux — strongest objection + decisive test, foregrounded
    _render_crux_block(lines, art)

    # 3. Balanced case — "Case for" immediately followed by "Case against"
    _render_balanced_case(lines, art)

    # 4. Claims & checks
    _render_claims_block(lines, art, refs)

    # 5. Completion & positioning (supporting layer, now below the crux)
    _render_supporting_layer(lines, art, audit_map, num_by_loc)

    # 6. References
    if refs:
        lines.append("## References")
        for ref in refs:
            lines.append(_reference_line(ref))
        lines.append("")

    # 7. Honest-limit sentence
    lines += ["---", f"> {LIMIT}"]
    return "\n".join(lines)


async def run_cli(
    seed,
    llm,
    cfg,
    backend=None,
    out_dir=None,
    references_path=None,
    resolver=None,
    citeauditor=None,
    run_id=None,
) -> dict:
    out = Path(out_dir or cfg.results_dir)
    slug = run_id or _slug(seed)
    run_dir = out / slug                       # one folder per run; everything for this run lives here
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log.bind(run_dir / "logs.jsonl")

    art = await run(seed, llm, cfg, backend=backend, run_id=slug)
    audit_map = await audit_narrative_refs(art, citeauditor)
    asserted = [r.reference for r in audit_map.values() if r.status == "resolved" and r.reference]
    refs = await build_references(art, references_path, resolver, asserted_refs=asserted)

    json_path = run_dir / "artifact.json"
    report_path = run_dir / "report.md"
    bib_path = run_dir / "references.bib"
    json_path.write_text(art.model_dump_json(indent=2))
    report_path.write_text(render_report(art, refs, audit_map))
    bib_path.write_text(to_bibtex(refs))
    return {
        "artifact": art,
        "json_path": str(json_path),
        "report_path": str(report_path),
        "bib_path": str(bib_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="valagents")
    parser.add_argument("seed")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--references")
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolver = DefaultResolver() if args.references else None
    citeauditor = CiteAuditor(ArxivBackend(), cfg=cfg)
    out = asyncio.run(
        run_cli(
            args.seed,
            OpenRouterClient(cfg),
            cfg,
            backend=build_backend(cfg),
            references_path=args.references,
            resolver=resolver,
            citeauditor=citeauditor,
        )
    )
    print(f"status: {out['artifact'].status} -> {out['report_path']}")


if __name__ == "__main__":
    main()
