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
from valagents.agents.magnitude_designer import design_magnitude
from valagents.agents.prover import prove_claim, build_derivation
from valagents.agents.computation_designer import design_computation
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
from valagents.agents.simulation_designer import design_simulation

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


async def _symbolic_check(claim, fc, llm, cfg, tick: int, run_id):
    """PC-1b: design + EXECUTE a symbolic check for a claim and return a code-witnessed CheckRecord
    (lens='executor'), or None if no plan was designed or the result was uncertain (fall back to the
    Prover). This is the LEGITIMATE derivation credit — a 'complete derivation' becomes a code-witnessed
    computations/ entry, not prover say-so. Shared by run_claim_checks (math claims) and
    inject_limit_checks (limit claims); the model designs the plan, the sandbox adjudicates."""
    plan = await design_computation(claim, fc, llm, cfg)
    if plan is None:
        return None, None
    from valagents.sandbox.executor import run_plan
    from valagents.computation import verdict_to_check
    adir = _computations_dir(cfg, run_id, claim.id)
    verdict = run_plan(plan, cfg, artifacts_dir=adir)
    if verdict.verdict == "uncertain":          # decisive only — uncertain falls back to the Prover (F2/§5)
        return None, verdict
    return verdict_to_check(verdict, tick=tick), verdict


async def run_claim_checks(store: ArtifactStore, backend, llm, cfg: Config, tick0: int = 0, run_id=None) -> None:
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

        # PC-1b: a mathematical claim earns its independent credit from a CODE-WITNESSED symbolic check
        # (the legit path the stripped prover say-so used to fake) — design + execute, mirror limit checks.
        if claim.type == "mathematical":
            rec, verdict = await _symbolic_check(claim, fc, llm, cfg, tick, run_id)
            tick += 1
            if verdict is not None:
                store.record({"event": "symbolic_check", "claim": claim.id, "verdict": verdict.verdict,
                              "computed": verdict.measured})
            if rec is not None:
                store.add_check(claim.id, rec)

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


async def _whole_artifact_lenses(store: ArtifactStore, backend, llm, cfg: Config, tick: int, resolver=None, run_id=None) -> None:
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
    await run_magnitude_checks(store, llm, cfg, tick=tick + 500, resolver=resolver, run_id=run_id)
    await run_simulation_checks(store, llm, cfg, tick=tick + 700, run_id=run_id)
    plan = await design_validation(art, llm, cfg)
    store.set("validation_plan", plan)
    if plan is not None:
        store.record({
            "event": "validation_plan",
            "cost": plan.cost,
            "test": plan.decisive_test,
        })


def _computations_dir(cfg, run_id, *parts) -> str | None:
    """Per-run sandbox-artifacts dir inside the run's own folder: <results>/<run_id>/computations/<parts...>.
    Grouped by RUN (one folder per run holds everything), so runs don't collide. Falls back to
    <results>/computations/<parts...> when run_id is None (callers that don't pass a run id, e.g. tests)."""
    base = getattr(cfg, "results_dir", None)
    if not base:
        return None
    segs = [str(base)]
    if run_id:
        segs.append(str(run_id))
    segs.append("computations")
    segs.extend(str(p) for p in parts)
    return "/".join(segs)


async def run_magnitude_checks(store, llm, cfg, tick: int = 0, resolver=None, run_id=None) -> None:
    art = store.current
    # L2-D11: drop prior bound_check claims (and their checks) before re-injecting, so repeated runs
    # across repair iterations do not accumulate duplicate bound claims (mirrors red_team overwriting attacks).
    art.claim_graph = [c for c in art.claim_graph if c.origin != "bound_check"]
    bn = 0
    for p in art.predictions:
        if not p.measurable:
            continue
        plan = await design_magnitude(p, art, llm, cfg)
        if plan is None:
            continue
        from valagents.sandbox.executor import run_plan
        from valagents.computation import verdict_to_attack, verdict_to_check
        from valagents.agents.value_grounder import ground_plan
        adir = _computations_dir(cfg, run_id, "magnitude")
        verdict = run_plan(plan, cfg, artifacts_dir=adir)
        store.record({"event": "magnitude_executed", "kind": plan.comparison_kind,
                      "verdict": verdict.verdict, "computed": verdict.measured})
        if verdict.verdict == "uncertain":
            continue                                  # FAIL-CLOSED: no attack, no claim, no attempted-mark (L2-D9/F2)
        grounding = await ground_plan(plan, resolver, llm, cfg)   # None iff grounding OFF
        if grounding is not None and grounding.status == "contradicts":
            store.record({"event": "magnitude_grounding", "kind": plan.comparison_kind, "status": "contradicts"})
            continue                                               # suppress: input is literature-contradicted
        if plan.comparison_kind == "bound_check":
            # CLAIM path: inject a load-bearing mathematical claim; violate -> fail -> REFUTED, comply -> pass.
            bn += 1
            claim_id = f"BND{bn}"
            existing = {c.id for c in art.claim_graph}
            suffix = 0
            while claim_id in existing:
                suffix += 1
                claim_id = f"BND{bn}_{suffix}"
            claim = AtomicClaim(
                id=claim_id, type="mathematical", load_bearing=True, origin="bound_check",
                statement=(f"The idea's predicted effect respects the established bound "
                           f"({plan.bound}, source: {plan.bound_source})."))
            art.claim_graph.append(claim)
            store.add_check(claim_id, verdict_to_check(verdict, tick=tick, grounding=grounding))
            claim.exhausted = True
            tick += 1
            continue
        # ATTACK path (sensitivity_ratio / discriminating_margin): decisive verdict -> Attack + mark "magnitude" attempted.
        attack = verdict_to_attack(verdict, plan.target_claim_id, plan.discriminating, tick=tick)
        if grounding is not None and grounding.quote:
            attack = attack.model_copy(update={"basis": attack.basis + f"; grounding={grounding.status} (quote: {grounding.quote})"})
        art.attacks = art.attacks + [attack]
        if art.attack_surface is not None and "magnitude" not in art.attack_surface.attempted:
            art.attack_surface.attempted = art.attack_surface.attempted + ["magnitude"]
        tick += 1


async def run_simulation_checks(store, llm, cfg, tick: int = 0, run_id=None) -> None:
    art = store.current
    claims = [c for c in art.claim_graph if c.type == "mechanistic"][:3]   # no-op when none; cap at 3
    for claim in claims:
        plan = await design_simulation(claim, art, llm, cfg)
        if plan is None:
            continue
        from valagents.sandbox.executor import run_plan
        from valagents.computation import verdict_to_sim_attack
        adir = _computations_dir(cfg, run_id, "simulation", claim.id)
        verdict = run_plan(plan, cfg, artifacts_dir=adir)
        store.record({"event": "simulation_executed", "claim": claim.id,
                      "verdict": verdict.verdict, "computed": verdict.measured})
        if verdict.verdict == "uncertain":
            continue                                   # FAIL-CLOSED: no attack, no mark (L2-D9 / F2)
        fatal_eligible = bool(claim.load_bearing and claim.role == "novel_core")
        attack = verdict_to_sim_attack(verdict, claim.id, fatal_eligible, tick=tick)
        art.attacks = art.attacks + [attack]
        if art.attack_surface is not None and "simulation" not in art.attack_surface.attempted:
            art.attack_surface.attempted = art.attack_surface.attempted + ["simulation"]
        tick += 1


async def inject_limit_checks(store: ArtifactStore, llm, cfg: Config, tick: int, run_id=None) -> None:
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

        # F2 / PC-1b: augment the reasoned Prover with an EXECUTED symbolic check (shared helper)
        rec2, verdict = await _symbolic_check(claim, art.formal_claim, llm, cfg, tick, run_id)
        if verdict is not None:
            store.record({"event": "limit_executed", "claim": claim_id,
                          "verdict": verdict.verdict, "computed": verdict.measured})
        if rec2 is not None:
            store.add_check(claim_id, rec2)
            tick += 1

        claim.exhausted = True


async def run(raw_idea: str, llm, cfg: Config, backend=None, run_id=None) -> IdeaArtifact:
    store = ArtifactStore(IdeaArtifact(raw_idea=raw_idea))
    if not await run_entry_gates(store, raw_idea, backend, llm, cfg):
        return store.current

    resolver = None
    if cfg.grounding.backend != "none":
        from valagents.agents.value_grounder import LiveFetcher
        resolver = LiveFetcher()

    await run_claim_checks(store, backend, llm, cfg, run_id=run_id)
    await _whole_artifact_lenses(store, backend, llm, cfg, tick=1000, resolver=resolver, run_id=run_id)
    await inject_limit_checks(store, llm, cfg, tick=1500, run_id=run_id)

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
        await run_claim_checks(store, backend, llm, cfg, tick0=2000 * version, run_id=run_id)
        await _whole_artifact_lenses(store, backend, llm, cfg, tick=3000 * version, resolver=resolver, run_id=run_id)

    store.current.finalized = True
    verdict = await arbitrate(store.current, llm, cfg)
    store.record({
        "event": "final",
        "status": store.current.status,
        "load_bearing": store.current.load_bearing,
        "agrees": verdict["agrees"],
    })
    return store.current
