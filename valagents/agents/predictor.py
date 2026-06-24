from __future__ import annotations
from valagents.artifact import Prediction, FormalClaim, Novelty
from valagents.parse import checked_lines
from valagents.prompts import PREDICTOR
from valagents.agents.base import build_messages

_DETECTABLE_VALUES = {"yes", "no", "unclear"}


async def predict(formal_claim: FormalClaim, novelty: Novelty | None, llm, cfg) -> list[Prediction]:
    user = PREDICTOR.format(formal=formal_claim.statement, delta=(novelty.delta if novelty else ""))
    rows = await checked_lines("predictor", build_messages("You extract falsifiable predictions.", user),
                               ["OBSERVABLE", "EFFECT_SIZE", "DISCRIMINATES_FROM", "MEASURABLE", "DETECTABLE"],
                               llm=llm)
    if not rows:
        return []
    result = []
    for r in rows:
        raw_detectable = r.get("detectable", "").strip().lower()
        detectable = raw_detectable if raw_detectable in _DETECTABLE_VALUES else "unclear"
        result.append(Prediction(
            observable=r["observable"],
            effect_size=r["effect_size"],
            discriminates_from=r["discriminates_from"],
            measurable=r["measurable"].strip().lower().startswith("y"),
            detectable=detectable,
        ))
    return result
