from __future__ import annotations

from valagents.artifact import IdeaCompletion, IdeaArtifact
from valagents.agents.base import build_messages, choice, split_list
from valagents.agents.redteam import _render
from valagents.parse import checked
from valagents.prompts import IDEA_COMPLETER


async def complete_idea(art: IdeaArtifact, llm, cfg) -> IdeaCompletion | None:
    user = IDEA_COMPLETER.format(artifact=_render(art))
    tail = await checked(
        "completer",
        build_messages("You complete candidate research ideas before validation.", user),
        ["COMPLETION_STATUS", "COMPLETED_IDEA", "MECHANISM", "ASSUMPTIONS", "WEAKEST_LINK"],
        llm=llm,
    )
    if tail is None:
        return None
    status = choice(
        tail["completion_status"],
        {"incomplete", "completed_candidate", "polished_research_plan"},
    ) or "incomplete"
    return IdeaCompletion(
        status=status,
        completed_idea=tail["completed_idea"],
        mechanism=tail["mechanism"],
        assumptions=split_list(tail["assumptions"]),
        weakest_link=tail["weakest_link"],
    )
