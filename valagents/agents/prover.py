"""Prover: per-claim derivation/well-formedness checker.

Handles:
- definitional claims: coherence and non-circularity check.
- mathematical/mechanistic claims: derivation sketch with gap detection.

A prover pass counts as an independent internal check only for definitional and
mathematical claims. Mechanistic/empirical claims still need external grounding.
"""
from __future__ import annotations

from valagents.artifact import CheckRecord, Derivation, Gap, AtomicClaim, FormalClaim
from valagents.parse import checked
from valagents.prompts import PROVER
from valagents.agents.base import build_messages, choice


async def prove_claim(
    claim: AtomicClaim, formal_claim, llm, cfg, tick: int = 0
) -> CheckRecord:
    """Check the derivation or well-formedness of a single atomic claim.

    DERIVATION: complete + FATAL_GAP: no  → pass
    FATAL_GAP: yes                        → uncertain with severe-gap basis
    DERIVATION: gapped, no fatal gap      → uncertain (independent_sources=0)
    GAPS begins CONTRADICTION/COUNTEREXAMPLE → fail
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

    derivation_raw = tail["derivation"].strip().lower()
    derivation = "gapped" if derivation_raw.startswith("gapped") else choice(derivation_raw, {"complete"})
    fatal_gap = choice(tail["fatal_gap"], {"yes", "no"})
    if derivation is None or fatal_gap is None:
        return CheckRecord(lens="prover", verdict="uncertain", basis=tail["gaps"], tick=tick)

    gap_basis = tail["gaps"]
    refutes = gap_basis.strip().upper().startswith(("CONTRADICTION:", "COUNTEREXAMPLE:", "REFUTES:"))
    fatal = fatal_gap == "yes"
    gapped = derivation == "gapped"
    verdict = "fail" if refutes else ("uncertain" if fatal or gapped else "pass")
    basis = f"FATAL_DERIVATION_GAP: {gap_basis}" if fatal and not refutes else gap_basis
    indep = 1 if verdict == "pass" and claim.type in {"definitional", "mathematical"} else 0
    return CheckRecord(
        lens="prover",
        verdict=verdict,
        basis=basis,
        independent_sources=indep,
        tick=tick,
    )


async def build_derivation(formal_claim: FormalClaim, claims: list[AtomicClaim], llm, cfg) -> Derivation:
    """Aggregate per-claim check results into a top-level Derivation record."""
    steps = [c.statement for c in claims]
    gaps = [
        Gap(
            description=ck.basis,
            claim_id=c.id,
            fatal=(
                ck.verdict == "fail"
                or (ck.basis or "").strip().upper().startswith("FATAL_DERIVATION_GAP:")
            ),
        )
        for c in claims
        for ck in c.checks
        if ck.lens == "prover" and ck.verdict in ("fail", "uncertain") and ck.basis is not None
    ]
    return Derivation(steps=steps, gaps=gaps)
