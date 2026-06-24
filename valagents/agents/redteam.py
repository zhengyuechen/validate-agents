from __future__ import annotations
import logging
from valagents.artifact import Attack, AttackSurface, CheckRecord, IdeaArtifact
from valagents.parse import checked_lines_body, parse_tail, StrictTailError
from valagents.prompts import RED_TEAM
from valagents.agents.base import build_messages

log = logging.getLogger(__name__)

_CATS = ["counterexample", "failure_regime", "confound", "magnitude"]
_ATTACK_KEYS = ["ATTACK", "SEVERITY", "STATUS", "TARGET", "BASIS"]


def _explicit_refutation(basis: str) -> bool:
    return basis.strip().upper().startswith(("CONTRADICTION:", "COUNTEREXAMPLE:", "REFUTES:"))


def _render(art: IdeaArtifact) -> str:
    fc = art.formal_claim.statement if art.formal_claim else art.raw_idea
    claims = "\n".join(f"- {c.id} ({c.type}): {c.statement}" for c in art.claim_graph)
    return f"CLAIM: {fc}\nSUB-CLAIMS:\n{claims}"


async def red_team(art: IdeaArtifact, llm, cfg, tick: int = 0):
    user = RED_TEAM.format(artifact=_render(art))
    msgs = build_messages("You are an adversarial reviewer.", user)

    rows, body = await checked_lines_body("redteam", msgs, _ATTACK_KEYS, llm=llm)

    # Parse ATTEMPTED from the same body that attack rows came from.
    attempted: list[str] = []
    try:
        row = parse_tail(body, ["ATTEMPTED"])
        attempted = [c.strip().lower() for c in row["attempted"].split(",") if c.strip()]
    except StrictTailError:
        attempted = []

    attacks, per_claim = [], []
    for r in (rows or []):
        severity = r["severity"].strip().lower()
        status = r["status"].strip().lower()
        if severity not in {"fatal", "major", "minor"} or status not in {"survived", "landed"}:
            log.warning("redteam skipping malformed attack line: severity=%r status=%r", severity, status)
            continue
        tgt = None if r["target"].strip().lower() in ("none", "") else r["target"].strip()
        a = Attack(type=r["attack"].strip().lower(), severity=severity,
                   status=status, target_claim_id=tgt, basis=r["basis"])
        attacks.append(a)
        if a.status == "landed" and a.severity == "fatal" and tgt:
            verdict = "fail" if _explicit_refutation(a.basis) else "uncertain"
            per_claim.append((tgt, CheckRecord(lens="redteam", verdict=verdict,
                                               basis=a.basis, independent_sources=0, tick=tick)))

    surface = AttackSurface(attempted=attempted, skipped=[c for c in _CATS if c not in attempted])
    return attacks, surface, per_claim
