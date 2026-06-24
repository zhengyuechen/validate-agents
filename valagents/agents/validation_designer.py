from __future__ import annotations
from valagents.artifact import ValidationPlan, IdeaArtifact
from valagents.parse import checked
from valagents.prompts import VALIDATION_DESIGNER
from valagents.agents.base import build_messages, choice
from valagents.agents.redteam import _render


async def design_validation(art: IdeaArtifact, llm, cfg) -> ValidationPlan | None:
    user = VALIDATION_DESIGNER.format(artifact=_render(art))
    tail = await checked("validation_designer", build_messages("You design decisive tests.", user),
                         ["TEST", "CONFIRM_IF", "REFUTE_IF", "DISCRIMINATES_FROM", "INFERENTIAL_STANDARD", "COST"],
                         llm=llm)
    if tail is None:
        return None
    cost = choice(tail["cost"], {"low", "medium", "high"}) or "medium"
    return ValidationPlan(
        decisive_test=tail["test"],
        confirm_if=tail["confirm_if"],
        refute_if=tail["refute_if"],
        discriminates_from=tail["discriminates_from"],
        inferential_standard=tail["inferential_standard"],
        cost=cost,
    )
