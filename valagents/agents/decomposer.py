from __future__ import annotations
from valagents.artifact import AtomicClaim, FormalClaim
from valagents.parse import checked_lines
from valagents.prompts import DECOMPOSER
from valagents.agents.base import build_messages


async def decompose(formal_claim: FormalClaim, llm, cfg) -> list[AtomicClaim]:
    """Decompose a formal claim into atomic sub-claims with dependency edges.

    Returns empty list on parse failure (fail-closed).
    """
    user = DECOMPOSER.format(formal=formal_claim.statement)
    rows = await checked_lines("decomposer", build_messages("You expose structure.", user),
                               ["CLAIM", "TYPE", "DEPENDS_ON", "STATEMENT"], llm=llm)
    if not rows:
        return []
    out = []
    for r in rows:
        deps = [] if r["depends_on"].strip().lower() in ("none", "") else \
               [d.strip() for d in r["depends_on"].split(",") if d.strip()]
        out.append(AtomicClaim(id=r["claim"], statement=r["statement"],
                               type=r["type"].strip().lower(), depends_on=deps))
    return out
