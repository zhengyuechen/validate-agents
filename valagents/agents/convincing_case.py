from __future__ import annotations

from valagents.artifact import ConvincingCase, IdeaArtifact
from valagents.agents.base import build_messages, split_list
from valagents.agents.redteam import _render
from valagents.parse import checked
from valagents.prompts import CONVINCING_CASE


async def build_convincing_case(art: IdeaArtifact, llm, cfg) -> ConvincingCase | None:
    user = CONVINCING_CASE.format(artifact=_render(art))
    tail = await checked(
        "convincing_case",
        build_messages("You build sober scientific cases for candidate ideas.", user),
        [
            "ELEVATOR_VERSION",
            "TECHNICAL_VERSION",
            "WHY_EXISTING_THEORY_LEAVES_ROOM",
            "WHY_PLAUSIBLE",
            "SKEPTIC_TESTS",
        ],
        llm=llm,
    )
    if tail is None:
        return None
    return ConvincingCase(
        elevator_version=tail["elevator_version"],
        technical_version=tail["technical_version"],
        why_existing_theory_leaves_room=tail["why_existing_theory_leaves_room"],
        why_plausible=tail["why_plausible"],
        skeptic_tests=split_list(tail["skeptic_tests"]),
    )
