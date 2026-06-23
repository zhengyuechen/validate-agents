from __future__ import annotations
from valagents.artifact import Prediction, FormalClaim, Novelty
from valagents.parse import checked_lines
from valagents.prompts import PREDICTOR
from valagents.agents.base import build_messages


async def predict(formal_claim: FormalClaim, novelty: Novelty | None, llm, cfg) -> list[Prediction]:
    user = PREDICTOR.format(formal=formal_claim.statement, delta=(novelty.delta if novelty else ""))
    rows = await checked_lines("predictor", build_messages("You extract falsifiable predictions.", user),
                               ["OBSERVABLE", "EFFECT_SIZE", "DISCRIMINATES_FROM", "MEASURABLE"], llm=llm)
    if not rows:
        return []
    return [Prediction(observable=r["observable"], effect_size=r["effect_size"],
                       discriminates_from=r["discriminates_from"],
                       measurable=r["measurable"].strip().lower().startswith("y")) for r in rows]
