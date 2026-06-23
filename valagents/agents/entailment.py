from __future__ import annotations
from valagents.artifact import Coverage, FormalClaim, AtomicClaim
from valagents.parse import checked
from valagents.prompts import ENTAILMENT
from valagents.agents.base import build_messages


async def entailment_check(formal_claim: FormalClaim, claims: list[AtomicClaim], llm, cfg) -> Coverage | None:
    """Check whether sub-claims logically establish the formal claim.

    On parse failure, returns Coverage(verdict="gap") — fail-closed behavior ensures
    unparseable entailment checks are treated as gaps in coverage.
    """
    sub = "\n".join(f"- {c.id}: {c.statement}" for c in claims)
    user = ENTAILMENT.format(formal=formal_claim.statement, subclaims=sub)
    tail = await checked("entailment", build_messages("You check logical coverage.", user),
                         ["COVERS", "MISSING"], llm=llm)
    if tail is None:
        return Coverage(verdict="gap", missing="(unparseable entailment check)")  # fail closed
    missing = None if tail["missing"].strip().lower() in ("none", "") else tail["missing"]
    return Coverage(verdict=tail["covers"].strip().lower(), missing=missing)
