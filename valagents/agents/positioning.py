from __future__ import annotations

from valagents.artifact import IdeaArtifact, PriorArtPositioning
from valagents.agents.base import build_messages, split_list
from valagents.agents.redteam import _render
from valagents.parse import checked
from valagents.prompts import PRIOR_ART_POSITIONING


async def position_prior_art(art: IdeaArtifact, llm, cfg) -> PriorArtPositioning | None:
    user = PRIOR_ART_POSITIONING.format(artifact=_render(art))
    tail = await checked(
        "positioning",
        build_messages("You position ideas against prior art without reducing novelty to support.", user),
        ["CLOSEST_PRIOR", "SIMILARITY", "DIFFERENCE", "WHAT_IS_NEW", "MUST_CITE"],
        llm=llm,
    )
    if tail is None:
        return None
    return PriorArtPositioning(
        closest_prior=tail["closest_prior"],
        similarity=tail["similarity"],
        difference=tail["difference"],
        what_is_new=tail["what_is_new"],
        must_cite=split_list(tail["must_cite"]),
    )
