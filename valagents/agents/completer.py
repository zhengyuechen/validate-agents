from __future__ import annotations

from valagents.artifact import Assumption, IdeaCompletion, IdeaArtifact
from valagents.agents.base import build_messages, choice
from valagents.agents.redteam import _render
from valagents.parse import checked_body, parse_tail_lines, StrictTailError
from valagents.prompts import IDEA_COMPLETER

_ALLOWED_STATUSES = {"standard", "contested", "novel_load_bearing"}


def _parse_assumptions(body: str) -> list[Assumption]:
    try:
        rows = parse_tail_lines(body, ["ASSUMPTION", "STATUS"])
    except StrictTailError:
        return []
    result = []
    for row in rows:
        raw_status = row.get("status", "").strip().lower()
        status = raw_status if raw_status in _ALLOWED_STATUSES else "standard"
        result.append(Assumption(text=row["assumption"], status=status))
    return result


async def complete_idea(art: IdeaArtifact, llm, cfg) -> IdeaCompletion | None:
    user = IDEA_COMPLETER.format(artifact=_render(art))
    tail, body = await checked_body(
        "completer",
        build_messages("You complete candidate research ideas before validation.", user),
        ["COMPLETION_STATUS", "COMPLETED_IDEA", "MECHANISM", "WEAKEST_LINK"],
        llm=llm,
    )
    if tail is None:
        return None
    status = choice(
        tail["completion_status"],
        {"incomplete", "completed_candidate", "polished_research_plan"},
    ) or "incomplete"
    assumptions = _parse_assumptions(body)
    return IdeaCompletion(
        status=status,
        completed_idea=tail["completed_idea"],
        mechanism=tail["mechanism"],
        assumptions=assumptions,
        weakest_link=tail["weakest_link"],
    )
