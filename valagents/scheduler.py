"""DAG scheduler: entry gates → per-claim lenses → fan-out → repair-versioning → total verdict."""
from __future__ import annotations
from valagents.store import ArtifactStore
from valagents.config import Config
from valagents.agents.formalizer import formalize
from valagents.agents.faithfulness import faithfulness_check
from valagents.agents.decomposer import decompose
from valagents.agents.entailment import entailment_check


def _apply_gate_cfg(art, cfg: Config) -> None:
    art.min_attack_categories = cfg.gate.min_attack_categories
    art.fanout_N = cfg.gate.fanout_N
    art.repair_cap = cfg.gate.repair_cap


async def run_entry_gates(store: ArtifactStore, raw_idea: str, backend, llm, cfg: Config) -> bool:
    art = store.current
    _apply_gate_cfg(art, cfg)

    # 1. Formalizer
    fc = await formalize(raw_idea, llm, cfg)
    if fc is None:
        art.finalized = True
        store.record({"event": "entry_gate", "reason": "unformalizable", "stage": "formalizer"})
        return False
    store.set("formal_claim", fc)
    if not fc.falsifiable:
        art.finalized = True
        store.record({"event": "entry_gate", "reason": "not_falsifiable"})
        return False

    # 2. Faithfulness, with one re-formalization retry on narrowed/no
    f = await faithfulness_check(raw_idea, fc, llm, cfg, retried=False)
    if f.verdict in ("narrowed", "no"):
        fc2 = await formalize(
            raw_idea + "\n(Restate FAITHFULLY to the full seed; do not narrow.)", llm, cfg
        )
        if fc2 is not None:
            store.set("formal_claim", fc2)
            fc = fc2
        f = await faithfulness_check(raw_idea, fc, llm, cfg, retried=True)
    store.set("faithfulness", f)
    if f.verdict in ("narrowed", "no"):
        art.finalized = True
        reason = "unfaithful_drift" if f.verdict == "no" else "unfaithful_narrowed"
        store.record({"event": "entry_gate", "reason": reason})
        return False

    # 3. Decomposer, with one retry on empty
    claims = await decompose(fc, llm, cfg)
    if not claims:
        claims = await decompose(fc, llm, cfg)
    store.set("claim_graph", claims)
    if not claims:
        art.finalized = True
        store.record({"event": "entry_gate", "reason": "ill_formed"})
        return False

    # 4. Entailment — records coverage but does not stop the run
    cov = await entailment_check(fc, claims, llm, cfg)
    store.set("coverage", cov)
    store.record({"event": "entry_ok", "claims": len(claims), "coverage": cov.verdict})
    return True


from valagents.agents.grounder import ground_claim
from valagents.agents.prover import prove_claim

_LENS_BY_TYPE: dict[str, list[str]] = {
    "definitional": ["prover"],
    "mathematical": ["grounder", "prover"],
    "empirical":    ["grounder"],
    "mechanistic":  ["grounder", "prover"],
}


async def _run_lens(name: str, claim, fc, backend, llm, cfg, tick: int):
    if name == "grounder":
        return await ground_claim(claim, fc, backend, llm, cfg, tick=tick)
    return await prove_claim(claim, fc, llm, cfg, tick=tick)


async def run_claim_checks(store: ArtifactStore, backend, llm, cfg: Config, tick0: int = 0) -> None:
    art = store.current
    fc = art.formal_claim
    tick = tick0
    for claim in art.claim_graph:
        lenses = list(_LENS_BY_TYPE.get(claim.type, ["grounder"]))
        for name in lenses:
            rec = await _run_lens(name, claim, fc, backend, llm, cfg, tick)
            tick += 1
            store.add_check(claim.id, rec)
            store.record({"event": "check", "claim": claim.id, "lens": name, "verdict": rec.verdict})

        # Fan-out: load-bearing claim still uncertain after matrix lenses → run diverse-type
        # extra lenses until fanout_N total checks have been recorded. Diverse means lens types
        # NOT already in the coverage matrix for this claim type, so disagreement is detectable.
        if claim.load_bearing and claim.status == "uncertain":
            extra = [n for n in ("grounder", "prover") if n not in lenses]
            if not extra:
                extra = ["grounder"]
            i = 0
            while len(claim.checks) < cfg.gate.fanout_N and i < len(extra) + 2:
                name = extra[i % len(extra)]
                rec = await _run_lens(name, claim, fc, backend, llm, cfg, tick)
                tick += 1
                store.add_check(claim.id, rec)
                store.record({"event": "fanout", "claim": claim.id, "lens": name, "verdict": rec.verdict})
                i += 1

        claim.exhausted = True
