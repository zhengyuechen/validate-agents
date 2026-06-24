from __future__ import annotations
from valagents.parse import checked
from valagents.prompts import ARBITER
from valagents.agents.base import build_messages
from valagents.agents.redteam import _render
from valagents import run_log


async def arbitrate(artifact, llm, cfg) -> dict:
    computed = artifact.status                       # authoritative (I1)
    lb = artifact.load_bearing                       # snapshot before await
    user = ARBITER.format(artifact=_render(artifact), computed_status=computed)
    tail = await checked("arbiter", build_messages("You assemble the verdict.", user),
                         ["STATUS", "LOAD_BEARING", "DECISIVE_TEST"], llm=llm)
    if tail is None:
        run_log.emit("arbiter_parse_failure", computed=computed)
        return {"status": computed, "narrated": "", "agrees": False,
                "load_bearing": lb, "decisive_test": ""}
    narrated = tail.get("status", "").strip().lower()
    agrees = (narrated == computed.strip().lower())
    if not agrees:
        run_log.emit("arbiter_mismatch", narrated=narrated, computed=computed)
    return {"status": computed, "narrated": narrated, "agrees": agrees,
            "load_bearing": tail.get("load_bearing", lb),
            "decisive_test": tail.get("decisive_test", "")}
