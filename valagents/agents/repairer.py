from __future__ import annotations
from valagents.parse import checked_body, parse_tail_lines, StrictTailError
from valagents.prompts import REPAIRER
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render


async def repair(artifact, target_ids: list[str], llm, cfg) -> dict | None:
    if not target_ids:
        return None
    user = REPAIRER.format(artifact=_render(artifact), targets=", ".join(target_ids))
    msgs = build_messages("You repair claims without weakening them.", user)
    summary, body = await checked_body("repairer", msgs, ["REPAIR", "TARGETS", "RATIONALE"], llm=llm)
    if summary is None:
        return None
    try:
        rows = parse_tail_lines(body, ["CLAIM", "STATEMENT"])
    except StrictTailError:
        rows = []
    new_statements = {r["claim"]: r["statement"] for r in rows}
    targets = [t.strip() for t in summary["targets"].split(",") if t.strip()] or target_ids
    return {"repair": summary["repair"], "targets": targets, "rationale": summary["rationale"],
            "new_statements": new_statements}
