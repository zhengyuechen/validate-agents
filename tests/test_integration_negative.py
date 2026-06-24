"""Negative-case integration tests: ill_posed and refuted outcomes.

Both tests run end-to-end through scheduler.run() using deterministic FakeLLMs
(no network). They demonstrate the system kills bad ideas visibly.
"""
from valagents.cli import render_report
from valagents.scheduler import run
from tests.fake_llm import FakeLLM


class FakeBackend:
    async def search(self, query, max_results=5):
        return []


# ---------------------------------------------------------------------------
# Test 1 — Un-falsifiable seed → ill_posed
# ---------------------------------------------------------------------------

def _unfalsifiable_router(agent, messages):
    """Formalizer returns FALSIFIABLE: no; all other agents should not be reached."""
    if agent == "formalizer":
        return (
            "CLAIM: consciousness is non-physical | VARIABLES: consciousness "
            "| REGIME: any | FALSIFIABLE: no"
        )
    # Formalizer may be retried with a prompt asking for a faithful re-statement;
    # return the same unfalsifiable claim.
    return (
        "CLAIM: consciousness is non-physical | VARIABLES: consciousness "
        "| REGIME: any | FALSIFIABLE: no"
    )


async def test_unfalsifiable_seed_is_ill_posed(cfg):
    art = await run(
        "consciousness is fundamentally non-physical and cannot be measured",
        FakeLLM(_unfalsifiable_router),
        cfg,
        backend=FakeBackend(),
    )
    assert art.status == "needs_experiment"
    assert art.blocker is not None
    assert art.blocker["reason"] == "not_falsifiable"
    assert art.verdict_class == "ill_posed"


async def test_unfalsifiable_report_does_not_suggest_experiment(cfg):
    art = await run(
        "consciousness is fundamentally non-physical and cannot be measured",
        FakeLLM(_unfalsifiable_router),
        cfg,
        backend=FakeBackend(),
    )
    report = render_report(art)
    # Must contain framing language, not an experiment directive
    assert "not yet a testable claim" in report or "reframe" in report
    # Must NOT tell the user to run an experiment
    assert "run an experiment" not in report.lower()
    assert "needs experiment" not in report.lower()


# ---------------------------------------------------------------------------
# Test 2 — Limit-violating idea → refuted via failed limit-recovery claim
# ---------------------------------------------------------------------------

def _limit_violating_router(agent, messages):
    """
    A faithful, decomposed, falsifiable seed whose main claims pass — but
    known_limits names a limit the idea must recover, and the prover finds a
    contradiction when checking the injected limit_recovery claim (L1).
    """
    content = messages[-1]["content"] if messages else ""

    if agent == "formalizer":
        return (
            "CLAIM: a new quantum gravity theory predicts faster-than-light signalling "
            "| VARIABLES: v, c | REGIME: low energy | FALSIFIABLE: yes"
        )
    if agent == "faithfulness":
        return "FAITHFUL: yes | BACK_TRANSLATION: QG theory allows FTL signals"
    if agent == "decomposer":
        return (
            "CLAIM: A | TYPE: mathematical | DEPENDS_ON: none "
            "| STATEMENT: the theory modifies the causal cone at low energy\n"
            "CLAIM: B | TYPE: empirical | DEPENDS_ON: none "
            "| STATEMENT: measurable FTL signal is predicted above noise floor"
        )
    if agent == "entailment":
        return "COVERS: complete | MISSING: none"
    if agent == "grounder":
        # novelty check (no SUB-CLAIM in content)
        if "SUB-CLAIM" not in content:
            return "CLOSEST_PRIOR: standard QG | DELTA: causal structure | POSITION: new"
        # both main claims: pass (independent sources found)
        return (
            "CLAIM: A | SUPPORT: supported | INDEPENDENT_SOURCES: 2 "
            "| SOURCES: A1,A2 | BASIS: standard modification"
        )
    if agent == "prover":
        # limit-recovery claim L1 — contains "recovers/respects the known limit"
        if "recovers/respects the known limit" in content or "limit_recovery" in content.lower():
            return (
                "DERIVATION: gapped | GAPS: CONTRADICTION: FTL signalling violates "
                "special relativity causality bound | FATAL_GAP: yes"
            )
        # main mathematical claim A passes
        return "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    if agent == "known_limits":
        return (
            "LIMIT: special relativity causality (no FTL signalling) "
            "| RECOVERED: no "
            "| FAILURE_IF_NOT: theory predicts observable causality violation "
            "| REPAIR_NEEDED: must show signals remain sub-luminal"
        )
    if agent == "completer":
        return (
            "COMPLETION_STATUS: completed_candidate | COMPLETED_IDEA: a QG theory with FTL "
            "| MECHANISM: modified causal structure | WEAKEST_LINK: causality recovery"
        )
    if agent == "theory_bridge":
        return (
            "THEORY_FAMILY: quantum gravity | NEAREST_THEORIES: LQG, string theory "
            "| EXTENDS: standard QG | CHALLENGES: causality | RECOVERS_KNOWN_LIMITS: no "
            "| DEPARTURE_POINT: causal cone modification | EXPERT_TRANSLATION: FTL via QG"
        )
    if agent == "positioning":
        return (
            "CLOSEST_PRIOR: standard QG | SIMILARITY: modifies spacetime "
            "| DIFFERENCE: predicts FTL | WHAT_IS_NEW: FTL signal | MUST_CITE: none"
        )
    if agent == "convincing_case":
        return (
            "ELEVATOR_VERSION: QG allows FTL | TECHNICAL_VERSION: causal cone modification "
            "| WHY_EXISTING_THEORY_LEAVES_ROOM: QG is incomplete "
            "| WHY_PLAUSIBLE: Planck-scale effects | SKEPTIC_TESTS: interferometer"
        )
    if agent == "steelman_objection":
        return (
            "STRONGEST_OBJECTION: violates SR causality | MECHANISM_OF_FAILURE: FTL signals "
            "| THREATENING_RESULT: special relativity | WHAT_WOULD_KILL_IT: SR violation "
            "| FAIR_SUMMARY: contradicts established causality"
        )
    if agent == "predictor":
        return (
            "OBSERVABLE: FTL signal timing | EFFECT_SIZE: delta_t > 0 "
            "| DISCRIMINATES_FROM: SR | MEASURABLE: yes | DETECTABLE: unclear"
        )
    if agent == "redteam":
        return (
            "ATTEMPTED: counterexample, failure_regime, magnitude, confound\n"
            "ATTACK: counterexample | SEVERITY: minor | STATUS: survived "
            "| TARGET: none | BASIS: no direct counterexample found"
        )
    if agent == "validation_designer":
        return (
            "TEST: interferometer FTL timing test | CONFIRM_IF: delta_t > 0 measured "
            "| REFUTE_IF: no signal above noise | DISCRIMINATES_FROM: SR "
            "| INFERENTIAL_STANDARD: p=0.01 | COST: high"
        )
    if agent == "arbiter":
        return "STATUS: refuted | LOAD_BEARING: L1 | DECISIVE_TEST: causality violation check"
    return ""


async def test_limit_violating_idea_is_refuted(cfg):
    art = await run(
        "a new quantum gravity theory predicts faster-than-light signalling at low energy",
        FakeLLM(_limit_violating_router),
        cfg,
        backend=FakeBackend(),
    )
    # The injected limit-recovery claim L1 must be fail
    limit_claims = [c for c in art.claim_graph if c.origin == "limit_recovery"]
    assert limit_claims, "Expected at least one limit_recovery claim to be injected"
    l1 = limit_claims[0]
    assert l1.status == "fail", f"Expected L1 status=fail, got {l1.status!r}"

    assert art.status == "refuted"
    assert art.verdict_class == "refuted"
