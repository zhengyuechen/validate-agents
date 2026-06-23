from __future__ import annotations
import logging
from valagents.artifact import Attack, AttackSurface, CheckRecord, IdeaArtifact
from valagents.parse import parse_tail_lines, parse_tail, StrictTailError, _reask
from valagents.prompts import RED_TEAM
from valagents.agents.base import build_messages

log = logging.getLogger(__name__)

_CATS = ["counterexample", "failure_regime", "confound", "magnitude"]
_ATTACK_KEYS = ["ATTACK", "SEVERITY", "STATUS", "TARGET", "BASIS"]


def _render(art: IdeaArtifact) -> str:
    fc = art.formal_claim.statement if art.formal_claim else art.raw_idea
    claims = "\n".join(f"- {c.id} ({c.type}): {c.statement}" for c in art.claim_graph)
    return f"CLAIM: {fc}\nSUB-CLAIMS:\n{claims}"


async def red_team(art: IdeaArtifact, llm, cfg, tick: int = 0):
    user = RED_TEAM.format(artifact=_render(art))
    msgs = build_messages("You are an adversarial reviewer.", user)

    # Single call — parse both ATTACK rows and ATTEMPTED from the same body.
    body = await llm.complete("redteam", msgs)
    rows = None
    try:
        rows = parse_tail_lines(body, _ATTACK_KEYS)
    except StrictTailError:
        reask = list(msgs) + [{"role": "assistant", "content": body},
                              {"role": "user", "content": _reask(_ATTACK_KEYS)}]
        body2 = await llm.complete("redteam", reask)
        try:
            rows = parse_tail_lines(body2, _ATTACK_KEYS)
        except StrictTailError:
            log.warning("redteam strict-tail double failure")

    attacks, per_claim = [], []
    if rows:
        for r in rows:
            tgt = None if r["target"].strip().lower() in ("none", "") else r["target"].strip()
            a = Attack(type=r["attack"].strip().lower(), severity=r["severity"].strip().lower(),
                       status=r["status"].strip().lower(), target_claim_id=tgt, basis=r["basis"])
            attacks.append(a)
            if a.status == "landed" and tgt:
                verdict = "fail" if a.severity in ("fatal", "major") else "uncertain"
                per_claim.append((tgt, CheckRecord(lens="redteam", verdict=verdict,
                                                   basis=a.basis, independent_sources=1, tick=tick)))

    # Parse ATTEMPTED from the same first-call body; default to [] if absent.
    attempted: list[str] = []
    try:
        row = parse_tail(body, ["ATTEMPTED"])
        attempted = [c.strip().lower() for c in row["attempted"].split(",") if c.strip()]
    except (StrictTailError, Exception):
        attempted = []

    surface = AttackSurface(attempted=attempted, skipped=[c for c in _CATS if c not in attempted])
    return attacks, surface, per_claim
