"""DAG scheduler: entry gates → per-claim lenses → fan-out → repair-versioning → total verdict."""
from __future__ import annotations
from valagents.store import ArtifactStore
from valagents.config import Config
from valagents.artifact import IdeaArtifact, AtomicClaim
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
from valagents.agents.grounder import ground_novelty
from valagents.agents.prover import prove_claim, build_derivation
from valagents.agents.completer import complete_idea
from valagents.agents.theory_bridge import build_theory_bridge
from valagents.agents.positioning import position_prior_art
from valagents.agents.known_limits import check_known_limits
from valagents.agents.convincing_case import build_convincing_case
from valagents.agents.steelman import build_steelman_objection
from valagents.agents.predictor import predict
from valagents.agents.redteam import red_team
from valagents.agents.validation_designer import design_validation
from valagents.agents.repairer import repair
from valagents.agents.arbiter import arbitrate

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

        # fan-out: a load-bearing claim still `uncertain` gets DISTINCT diverse-type lenses
        # (each at most once) until fanout_N lenses have run or no diverse type remains.
        if claim.load_bearing and claim.status == "uncertain":
            run_types = set(lenses)
            for name in ("grounder", "prover"):
                if len(claim.checks) >= cfg.gate.fanout_N:
                    break
                if name in run_types:
                    continue
                rec = await _run_lens(name, claim, fc, backend, llm, cfg, tick); tick += 1
                store.add_check(claim.id, rec)
                store.record({"event": "fanout", "claim": claim.id, "lens": name, "verdict": rec.verdict})
                run_types.add(name)
            if len(claim.checks) < cfg.gate.fanout_N:
                store.record({"event": "fanout_limited", "claim": claim.id})

        claim.exhausted = True

    if fc is not None:
        store.set("derivation", await build_derivation(fc, art.claim_graph, llm, cfg))


def _repair_targets(art: IdeaArtifact) -> list[str]:
    targets: set[str] = set()
    for attack in art.attacks:
        if (
            attack.status == "landed"
            and attack.severity in ("fatal", "major")
            and attack.target_claim_id
        ):
            targets.add(attack.target_claim_id)
    if art.derivation is not None:
        targets.update(gap.claim_id for gap in art.derivation.gaps if gap.fatal)
    return sorted(targets)


def _apply_repair_statements(art: IdeaArtifact, repaired: dict | None, targets: list[str]) -> None:
    if repaired is None:
        return
    allowed = set(targets)
    new_statements = repaired.get("new_statements", {})
    for claim in art.claim_graph:
        if claim.id in allowed and claim.id in new_statements:
            claim.statement = new_statements[claim.id]


async def _whole_artifact_lenses(store: ArtifactStore, backend, llm, cfg: Config, tick: int) -> None:
    art = store.current
    if art.formal_claim is None:
        return
    novelty = await ground_novelty(art.formal_claim, backend, llm, cfg)
    if novelty is not None:
        store.set("novelty", novelty)
        store.record({
            "event": "novelty",
            "position": novelty.position,
            "delta": novelty.delta,
        })
    completion = await complete_idea(art, llm, cfg)
    if completion is not None:
        store.set("completion", completion)
        store.record({
            "event": "completion",
            "status": completion.status,
            "weakest_link": completion.weakest_link,
        })
    bridge = await build_theory_bridge(art, llm, cfg)
    if bridge is not None:
        store.set("theory_bridge", bridge)
        store.record({
            "event": "theory_bridge",
            "family": bridge.theory_family,
            "nearest": bridge.nearest_theories,
        })
    positioning = await position_prior_art(art, llm, cfg)
    if positioning is not None:
        store.set("prior_art_positioning", positioning)
        store.record({
            "event": "positioning",
            "closest_prior": positioning.closest_prior,
            "what_is_new": positioning.what_is_new,
        })
    known_limits = await check_known_limits(art, llm, cfg)
    store.set("known_limits", known_limits)
    if known_limits:
        store.record({
            "event": "known_limits",
            "count": len(known_limits),
            "unclear": sum(1 for item in known_limits if item.recovered == "unclear"),
            "failed": sum(1 for item in known_limits if item.recovered == "no"),
        })
    case = await build_convincing_case(art, llm, cfg)
    if case is not None:
        store.set("convincing_case", case)
        store.record({
            "event": "convincing_case",
            "skeptic_tests": case.skeptic_tests,
        })
    objection = await build_steelman_objection(art, llm, cfg)
    if objection is not None:
        store.set("steelman_objection", objection)
        store.record({
            "event": "steelman_objection",
            "fair_summary": objection.fair_summary,
        })
    predictions = await predict(art.formal_claim, novelty, llm, cfg)
    store.set("predictions", predictions)
    store.record({
        "event": "predictions",
        "count": len(predictions),
        "measurable": sum(1 for item in predictions if item.measurable),
    })
    attacks, surface, per_claim = await red_team(art, llm, cfg, tick=tick)
    store.set("attacks", attacks)
    store.set("attack_surface", surface)
    store.record({
        "event": "redteam",
        "attacks": len(attacks),
        "landed": sum(1 for attack in attacks if attack.status == "landed"),
        "attempted": surface.attempted,
    })
    claim_ids = {claim.id for claim in art.claim_graph}
    for claim_id, record in per_claim:
        if claim_id in claim_ids:
            store.add_check(claim_id, record)
            store.record({
                "event": "redteam_check",
                "claim": claim_id,
                "verdict": record.verdict,
            })
    plan = await design_validation(art, llm, cfg)
    store.set("validation_plan", plan)
    if plan is not None:
        store.record({
            "event": "validation_plan",
            "cost": plan.cost,
            "test": plan.decisive_test,
        })


async def inject_limit_checks(store: ArtifactStore, llm, cfg: Config, tick: int) -> None:
    """Promote each known limit into a load_bearing mathematical AtomicClaim and prove it."""
    art = store.current
    limits = art.known_limits
    if not limits:
        return

    if len(limits) > 3:
        store.record({
            "event": "limit_checks_capped",
            "total": len(limits),
            "kept": 3,
        })
        limits = limits[:3]

    existing_ids = {c.id for c in art.claim_graph}
    for i, kl in enumerate(limits, start=1):
        claim_id = f"L{i}"
        # Ensure uniqueness if "L1" etc. already exist
        while claim_id in existing_ids:
            claim_id = f"L{i}_{len(existing_ids)}"
        existing_ids.add(claim_id)

        claim = AtomicClaim(
            id=claim_id,
            statement=f"In the relevant regime, the idea recovers/respects the known limit: {kl.limit}",
            type="mathematical",
            load_bearing=True,
            origin="limit_recovery",
        )
        art.claim_graph.append(claim)
        rec = await prove_claim(claim, art.formal_claim, llm, cfg, tick=tick)
        tick += 1
        store.add_check(claim_id, rec)
        store.record({
            "event": "limit_check",
            "claim": claim_id,
            "limit": kl.limit,
            "verdict": rec.verdict,
        })
        claim.exhausted = True


async def run(raw_idea: str, llm, cfg: Config, backend=None) -> IdeaArtifact:
    store = ArtifactStore(IdeaArtifact(raw_idea=raw_idea))
    if not await run_entry_gates(store, raw_idea, backend, llm, cfg):
        return store.current

    await run_claim_checks(store, backend, llm, cfg)
    await _whole_artifact_lenses(store, backend, llm, cfg, tick=1000)
    await inject_limit_checks(store, llm, cfg, tick=1500)

    while store.current.repairs_spent < cfg.gate.repair_cap:
        targets = _repair_targets(store.current)
        if not targets:
            break

        repaired = await repair(store.current, targets, llm, cfg)
        store.fork_for_repair(targets)
        _apply_gate_cfg(store.current, cfg)
        _apply_repair_statements(store.current, repaired, targets)
        store.record({"event": "repair", "targets": targets, "ok": repaired is not None})

        version = store.current.version_id
        await run_claim_checks(store, backend, llm, cfg, tick0=2000 * version)
        await _whole_artifact_lenses(store, backend, llm, cfg, tick=3000 * version)

    store.current.finalized = True
    verdict = await arbitrate(store.current, llm, cfg)
    store.record({
        "event": "final",
        "status": store.current.status,
        "load_bearing": store.current.load_bearing,
        "agrees": verdict["agrees"],
    })
    return store.current
