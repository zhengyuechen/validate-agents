"""DAG scheduler: entry gates → per-claim lenses → fan-out → repair-versioning → total verdict."""
from __future__ import annotations
from valagents.store import ArtifactStore
from valagents.config import Config
from valagents.artifact import IdeaArtifact
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
    completion = await complete_idea(art, llm, cfg)
    if completion is not None:
        store.set("completion", completion)
    bridge = await build_theory_bridge(art, llm, cfg)
    if bridge is not None:
        store.set("theory_bridge", bridge)
    positioning = await position_prior_art(art, llm, cfg)
    if positioning is not None:
        store.set("prior_art_positioning", positioning)
    store.set("known_limits", await check_known_limits(art, llm, cfg))
    case = await build_convincing_case(art, llm, cfg)
    if case is not None:
        store.set("convincing_case", case)
    store.set("predictions", await predict(art.formal_claim, novelty, llm, cfg))
    attacks, surface, per_claim = await red_team(art, llm, cfg, tick=tick)
    store.set("attacks", attacks)
    store.set("attack_surface", surface)
    claim_ids = {claim.id for claim in art.claim_graph}
    for claim_id, record in per_claim:
        if claim_id in claim_ids:
            store.add_check(claim_id, record)
    store.set("validation_plan", await design_validation(art, llm, cfg))


async def run(raw_idea: str, llm, cfg: Config, backend=None) -> IdeaArtifact:
    store = ArtifactStore(IdeaArtifact(raw_idea=raw_idea))
    if not await run_entry_gates(store, raw_idea, backend, llm, cfg):
        return store.current

    await run_claim_checks(store, backend, llm, cfg)
    await _whole_artifact_lenses(store, backend, llm, cfg, tick=1000)

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
