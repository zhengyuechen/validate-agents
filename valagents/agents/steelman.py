from __future__ import annotations

from valagents.artifact import SteelmanObjection, IdeaArtifact
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render
from valagents.parse import checked
from valagents.prompts import STEELMAN_OBJECTION


async def build_steelman_objection(art: IdeaArtifact, llm, cfg) -> SteelmanObjection | None:
    user = STEELMAN_OBJECTION.format(artifact=_render(art))
    tail = await checked(
        "steelman_objection",
        build_messages("You are the idea's most capable critic. Build the strongest honest case that it is wrong.", user),
        [
            "STRONGEST_OBJECTION",
            "MECHANISM_OF_FAILURE",
            "THREATENING_RESULT",
            "WHAT_WOULD_KILL_IT",
            "FAIR_SUMMARY",
        ],
        llm=llm,
    )
    if tail is None:
        return None
    return SteelmanObjection(
        strongest_objection=tail["strongest_objection"],
        mechanism_of_failure=tail["mechanism_of_failure"],
        threatening_result=tail["threatening_result"],
        what_would_kill_it=tail["what_would_kill_it"],
        fair_summary=tail["fair_summary"],
    )
