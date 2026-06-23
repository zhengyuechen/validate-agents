"""Prover: per-claim derivation/well-formedness checker.

Handles:
- definitional claims: coherence and non-circularity check.
- mathematical/mechanistic claims: derivation sketch with gap detection.

A prover 'pass' sets independent_sources=1 — the self-standing derivation counts as its
own independent check, allowing purely mathematical or definitional claims (which the
Grounder cannot externally support) to reach claim-status 'pass' on logic alone.
"""
from __future__ import annotations

from valagents.artifact import CheckRecord, Derivation, Gap, AtomicClaim, FormalClaim
from valagents.parse import checked
from valagents.prompts import PROVER
from valagents.agents.base import build_messages


async def prove_claim(
    claim: AtomicClaim, formal_claim, llm, cfg, tick: int = 0
) -> CheckRecord:
    """Check the derivation or well-formedness of a single atomic claim.

    DERIVATION: complete + FATAL_GAP: no  → pass  (independent_sources=1)
    FATAL_GAP: yes                        → fail  (independent_sources=0)
    DERIVATION: gapped, no fatal gap      → uncertain (independent_sources=0)
    """
    user = PROVER.format(ctype=claim.type, statement=claim.statement)
    tail = await checked(
        "prover",
        build_messages("You check derivations.", user),
        ["DERIVATION", "GAPS", "FATAL_GAP"],
        llm=llm,
    )
    if tail is None:
        return CheckRecord(lens="prover", verdict="uncertain", basis="(unparseable)", tick=tick)

    fatal = tail["fatal_gap"].strip().lower().startswith("y")
    gapped = tail["derivation"].strip().lower() == "gapped"
    verdict = "fail" if fatal else ("uncertain" if gapped else "pass")
    # A prover pass is a self-standing check; counts as one independent source.
    indep = 1 if verdict == "pass" else 0
    return CheckRecord(
        lens="prover",
        verdict=verdict,
        basis=tail["gaps"],
        independent_sources=indep,
        tick=tick,
    )


async def build_derivation(formal_claim: FormalClaim, claims: list[AtomicClaim], llm, cfg) -> Derivation:
    """Aggregate per-claim check results into a top-level Derivation record."""
    steps = [c.statement for c in claims]
    gaps = [
        Gap(description=ck.basis, claim_id=c.id, fatal=(ck.verdict == "fail"))
        for c in claims
        for ck in c.checks
        if ck.lens == "prover" and ck.verdict in ("fail", "uncertain") and ck.basis
    ]
    return Derivation(steps=steps, gaps=gaps)
