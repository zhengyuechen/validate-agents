from __future__ import annotations
from valagents.parse import checked
from valagents.prompts import REPAIRER
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render


async def repair(artifact, target_ids: list[str], llm, cfg) -> dict | None:
    user = REPAIRER.format(artifact=_render(artifact), targets=", ".join(target_ids))
    tail = await checked("repairer", build_messages("You repair claims without weakening them.", user),
                         ["REPAIR", "TARGETS", "RATIONALE"], llm=llm)
    if tail is None:
        return None
    targets = [t.strip() for t in tail["targets"].split(",") if t.strip()] or target_ids
    return {"repair": tail["repair"], "targets": targets, "rationale": tail["rationale"]}
