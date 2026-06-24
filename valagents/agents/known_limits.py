from __future__ import annotations

from valagents.artifact import IdeaArtifact, KnownLimit
from valagents.agents.base import build_messages, choice
from valagents.agents.redteam import _render
from valagents.parse import checked_lines
from valagents.prompts import KNOWN_LIMITS


async def check_known_limits(art: IdeaArtifact, llm, cfg) -> list[KnownLimit]:
    user = KNOWN_LIMITS.format(artifact=_render(art))
    rows = await checked_lines(
        "known_limits",
        build_messages("You check whether candidate theories recover known limits.", user),
        ["LIMIT", "RECOVERED", "FAILURE_IF_NOT", "REPAIR_NEEDED"],
        llm=llm,
    )
    out: list[KnownLimit] = []
    for row in rows or []:
        recovered = choice(row["recovered"], {"yes", "no", "unclear"}) or "unclear"
        out.append(
            KnownLimit(
                limit=row["limit"],
                recovered=recovered,
                failure_if_not=row["failure_if_not"],
                repair_needed=row["repair_needed"],
            )
        )
    return out
