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
    claims = "\n".join(f"- {c.id} ({c.type}/{c.role}): {c.statement}" for c in art.claim_graph)
    parts = [f"CLAIM: {fc}", f"SUB-CLAIMS:\n{claims}"]
    if art.completion is not None:
        assumption_texts = ', '.join(a.text for a in art.completion.assumptions) or 'none'
        parts.append(
            "COMPLETED IDEA:\n"
            f"{art.completion.completed_idea}\n"
            f"MECHANISM: {art.completion.mechanism}\n"
            f"ASSUMPTIONS: {assumption_texts}\n"
            f"WEAKEST_LINK: {art.completion.weakest_link}"
        )
    if art.theory_bridge is not None:
        parts.append(
            "THEORY BRIDGE:\n"
            f"FAMILY: {art.theory_bridge.theory_family}\n"
            f"NEAREST: {', '.join(art.theory_bridge.nearest_theories) or 'none'}\n"
            f"EXTENDS: {art.theory_bridge.extends}\n"
            f"CHALLENGES: {art.theory_bridge.challenges}\n"
            f"KNOWN_LIMITS: {art.theory_bridge.recovers_known_limits}\n"
            f"DEPARTURE: {art.theory_bridge.departure_point}\n"
            f"EXPERT_TRANSLATION: {art.theory_bridge.expert_translation}"
        )
    if art.prior_art_positioning is not None:
        parts.append(
            "PRIOR-ART POSITION:\n"
            f"CLOSEST: {art.prior_art_positioning.closest_prior}\n"
            f"SIMILARITY: {art.prior_art_positioning.similarity}\n"
            f"DIFFERENCE: {art.prior_art_positioning.difference}\n"
            f"NEW: {art.prior_art_positioning.what_is_new}\n"
            f"MUST_CITE: {', '.join(art.prior_art_positioning.must_cite) or 'none'}"
        )
    if art.known_limits:
        limits = "\n".join(
            f"- {limit.limit}: {limit.recovered}; failure={limit.failure_if_not}; repair={limit.repair_needed}"
            for limit in art.known_limits
        )
        parts.append(f"KNOWN LIMIT CHECKS:\n{limits}")
    if art.convincing_case is not None:
        parts.append(
            "CONVINCING CASE:\n"
            f"ELEVATOR: {art.convincing_case.elevator_version}\n"
            f"TECHNICAL: {art.convincing_case.technical_version}\n"
            f"ROOM: {art.convincing_case.why_existing_theory_leaves_room}\n"
            f"PLAUSIBLE: {art.convincing_case.why_plausible}\n"
            f"SKEPTIC_TESTS: {', '.join(art.convincing_case.skeptic_tests) or 'none'}"
        )
    return "\n\n".join(parts)


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
