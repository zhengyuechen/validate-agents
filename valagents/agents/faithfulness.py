from __future__ import annotations
from valagents.artifact import Faithfulness, FormalClaim
from valagents.parse import checked
from valagents.prompts import FAITHFULNESS
from valagents.agents.base import build_messages, choice


async def faithfulness_check(raw_idea, formal_claim: FormalClaim, llm, cfg, retried=False) -> Faithfulness | None:
    """Check whether a formal claim accurately reflects the seed idea.

    On double parse failure, returns Faithfulness(verdict="no") — fail-closed behavior ensures
    an unparseable faithfulness judgment is not permitted to proceed as faithful.
    """
    user = FAITHFULNESS.format(raw_idea=raw_idea, formal=formal_claim.statement)
    tail = await checked("faithfulness", build_messages("You are an independent reviewer.", user),
                         ["FAITHFUL", "BACK_TRANSLATION"], llm=llm)
    if tail is None:
        return Faithfulness(verdict="no", back_translation="(unparseable)", retried=retried)  # fail closed
    verdict = choice(tail["faithful"], {"yes", "narrowed", "no"})
    if verdict is None:
        verdict = "no"
    return Faithfulness(verdict=verdict, back_translation=tail["back_translation"], retried=retried)
