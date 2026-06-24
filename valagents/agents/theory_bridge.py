from __future__ import annotations

from valagents.artifact import IdeaArtifact, TheoryBridge
from valagents.agents.base import build_messages, split_list
from valagents.agents.redteam import _render
from valagents.parse import checked
from valagents.prompts import THEORY_BRIDGE


async def build_theory_bridge(art: IdeaArtifact, llm, cfg) -> TheoryBridge | None:
    user = THEORY_BRIDGE.format(artifact=_render(art))
    tail = await checked(
        "theory_bridge",
        build_messages("You anchor new ideas in existing theory.", user),
        [
            "THEORY_FAMILY",
            "NEAREST_THEORIES",
            "EXTENDS",
            "CHALLENGES",
            "RECOVERS_KNOWN_LIMITS",
            "DEPARTURE_POINT",
            "EXPERT_TRANSLATION",
        ],
        llm=llm,
    )
    if tail is None:
        return None
    return TheoryBridge(
        theory_family=tail["theory_family"],
        nearest_theories=split_list(tail["nearest_theories"]),
        extends=tail["extends"],
        challenges=tail["challenges"],
        recovers_known_limits=tail["recovers_known_limits"],
        departure_point=tail["departure_point"],
        expert_translation=tail["expert_translation"],
    )
