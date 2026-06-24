from __future__ import annotations
import re
from valagents.artifact import AtomicClaim, FormalClaim
from valagents.parse import checked_lines_body
from valagents.prompts import DECOMPOSER
from valagents.agents.base import build_messages, choice


_ROLES = {"background", "bridge", "novel_core", "assumption", "prediction", "test_condition"}


def _roles_by_claim_id(body: str) -> dict[str, str]:
    roles: dict[str, str] = {}
    for line in body.splitlines():
        claim = re.search(r"\bCLAIM\s*:\s*(.+?)\s*(?=\||$)", line, re.IGNORECASE)
        role = re.search(r"\bROLE\s*:\s*(.+?)\s*(?=\||$)", line, re.IGNORECASE)
        if not claim or not role:
            continue
        normalized = choice(role.group(1), _ROLES)
        if normalized is not None:
            roles[claim.group(1).strip()] = normalized
    return roles


async def decompose(formal_claim: FormalClaim, llm, cfg) -> list[AtomicClaim]:
    """Decompose a formal claim into atomic sub-claims with dependency edges.

    Returns empty list on parse failure (fail-closed).
    """
    user = DECOMPOSER.format(formal=formal_claim.statement)
    rows, body = await checked_lines_body("decomposer", build_messages("You expose structure.", user),
                                          ["CLAIM", "TYPE", "DEPENDS_ON", "STATEMENT"], llm=llm)
    if not rows:
        return []
    roles = _roles_by_claim_id(body)
    out = []
    for r in rows:
        ctype = choice(r["type"], {"definitional", "mathematical", "empirical", "mechanistic"})
        if ctype is None:
            continue
        role = roles.get(r["claim"]) or ("background" if ctype == "definitional" else "novel_core")
        deps = [] if r["depends_on"].strip().lower() in ("none", "") else \
               [d.strip() for d in r["depends_on"].split(",") if d.strip()]
        out.append(AtomicClaim(id=r["claim"], statement=r["statement"],
                               type=ctype, role=role, depends_on=deps))
    return out
