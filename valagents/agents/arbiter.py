from __future__ import annotations
from valagents.parse import checked
from valagents.prompts import ARBITER
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render
from valagents import run_log


async def arbitrate(artifact, llm, cfg) -> dict:
    computed = artifact.status                       # authoritative (I1)
    user = ARBITER.format(artifact=_render(artifact), computed_status=computed)
    tail = await checked("arbiter", build_messages("You assemble the verdict.", user),
                         ["STATUS", "LOAD_BEARING", "DECISIVE_TEST"], llm=llm)
    narrated = (tail or {}).get("status", "").strip().lower()
    agrees = narrated == computed
    if not agrees:
        run_log.emit("arbiter_mismatch", narrated=narrated, computed=computed)
    return {"status": computed, "narrated": narrated, "agrees": agrees,
            "load_bearing": (tail or {}).get("load_bearing", artifact.load_bearing),
            "decisive_test": (tail or {}).get("decisive_test", "")}
