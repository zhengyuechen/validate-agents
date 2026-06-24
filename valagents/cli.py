"""CLI: valagents "<seed>" -> IdeaArtifact JSON + markdown report."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from valagents import run_log
from valagents.config import load_config
from valagents.llm import OpenRouterClient
from valagents.references import (
    DefaultResolver,
    build_references,
    markers_for_claim,
    to_bibtex,
)
from valagents.scheduler import run
from valagents.web_search import build_backend


LIMIT = (
    "internally_validated means 'survived the checks this system can run,' never 'true' — "
    "every lens shares the base model's blind spots."
)


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


def render_report(art, refs=None) -> str:
    refs = refs or []
    blocker = art.blocker or {}
    lines = [
        "# Validation Report",
        "",
        f"**Seed:** {art.raw_idea}",
        "",
        f"**Status:** `{art.status}`",
        f"**Load-bearing claim:** `{art.load_bearing}`",
        f"**Blocker:** {blocker.get('reason', '-') } ({blocker.get('claim_id') or '-'})",
        f"**Maturity:** {art.maturity:.2f}",
        "",
    ]
    if art.formal_claim:
        lines += [
            f"**Formal claim:** {art.formal_claim.statement}",
            f"_falsifiable: {art.formal_claim.falsifiable}_",
            "",
        ]
    if art.claim_graph:
        lines.append("## Claim Graph")
        for claim in art.claim_graph:
            markers = "".join(f"[{n}]" for n in markers_for_claim(refs, claim.id))
            marker_text = f" {markers}" if markers else ""
            lines.append(f"- `{claim.id}` [{claim.type}] **{claim.status}**{marker_text} - {claim.statement}")
        lines.append("")
    if art.validation_plan:
        plan = art.validation_plan
        lines += [
            "## Decisive Test",
            f"- {plan.decisive_test}",
            f"- confirm if: {plan.confirm_if}",
            f"- refute if: {plan.refute_if}",
            f"- cost: {plan.cost}",
            "",
        ]
    if refs:
        lines.append("## References")
        for ref in refs:
            lines.append(_reference_line(ref))
        lines.append("")
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
    run_id=None,
) -> dict:
    out = Path(out_dir or cfg.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = run_id or _slug(seed)
    run_log.bind(out / ".logs" / f"{slug}.jsonl")

    art = await run(seed, llm, cfg, backend=backend)
    refs = await build_references(art, references_path, resolver)

    json_path = out / f"{slug}.json"
    report_path = out / f"{slug}.md"
    bib_path = out / f"{slug}.bib"
    json_path.write_text(art.model_dump_json(indent=2))
    report_path.write_text(render_report(art, refs))
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
    out = asyncio.run(
        run_cli(
            args.seed,
            OpenRouterClient(cfg),
            cfg,
            backend=build_backend(cfg),
            references_path=args.references,
            resolver=resolver,
        )
    )
    print(f"status: {out['artifact'].status} -> {out['report_path']}")


if __name__ == "__main__":
    main()
