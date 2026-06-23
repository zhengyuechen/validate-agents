from __future__ import annotations
from valagents.artifact import FormalClaim
from valagents.parse import checked
from valagents.prompts import FORMALIZER
from valagents.agents.base import build_messages


async def formalize(raw_idea: str, llm, cfg) -> FormalClaim | None:
    msgs = build_messages("You are a careful formalizer.", FORMALIZER.format(raw_idea=raw_idea))
    tail = await checked("formalizer", msgs, ["CLAIM", "VARIABLES", "REGIME", "FALSIFIABLE"], llm=llm)
    if tail is None:
        return None
    return FormalClaim(
        statement=tail["claim"],
        variables=[v.strip() for v in tail["variables"].split(",") if v.strip()],
        scope="", regime=tail["regime"],
        falsifiable=tail["falsifiable"].strip().lower().startswith("y"),
    )
