import json as _json
import pytest

from valagents.scheduler import run
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


def _grounder_body(tail: str, payload: dict) -> str:
    return tail + "\n```json\n" + _json.dumps(payload) + "\n```"


class FakeBackend:
    async def search(self, query, max_results=5):
        return [
            Article(title="A", summary="The revised effect exists and has been experimentally confirmed here.",
                    url="https://example.com/a", published="2026"),
            Article(title="B", summary="A separate group independently studied the same phenomenon.",
                    url="https://example.com/b", published="2026"),
        ]


def scripted(script):
    def route(agent, messages):
        value = script[agent]
        return value(messages) if callable(value) else value
    return FakeLLM(route)


BASE = {
    "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
    "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
    "decomposer": (
        "CLAIM: c1 | TYPE: empirical | ROLE: novel_core | DEPENDS_ON: none | "
        "STATEMENT: effect exists"
    ),
    "entailment": "COVERS: complete | MISSING: none",
    "grounder": _grounder_body(
        "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | BASIS: ok\n"
        "CLOSEST_PRIOR: p | DELTA: d | POSITION: new",
        {"citations": [
             {"label": "A1", "direction": "supports",
              "quote": "The revised effect exists and has been experimentally confirmed here."},
             {"label": "A2", "direction": "supports",
              "quote": "The revised effect exists and has been experimentally confirmed here."}]}
    ),
    "query_planner": "ARCHIVES: cond-mat | TERMS: effect, exists",  # grounder retrieval planner (query-agnostic FakeBackend)
    "completer": (
        "COMPLETION_STATUS: completed_candidate | COMPLETED_IDEA: completed x | "
        "MECHANISM: x causes y under stated assumptions | WEAKEST_LINK: c1\n"
        "ASSUMPTION: a1 | STATUS: standard\n"
        "ASSUMPTION: a2 | STATUS: standard"
    ),
    "theory_bridge": (
        "THEORY_FAMILY: field theory | NEAREST_THEORIES: prior model, analogy | "
        "EXTENDS: prior model | CHALLENGES: standard null | "
        "RECOVERS_KNOWN_LIMITS: recovers baseline when x=0 | "
        "DEPARTURE_POINT: adds mechanism x | EXPERT_TRANSLATION: x is an added coupling"
    ),
    "positioning": (
        "CLOSEST_PRIOR: prior model | SIMILARITY: same regime | "
        "DIFFERENCE: adds x | WHAT_IS_NEW: discriminating mechanism | MUST_CITE: prior model"
    ),
    "known_limits": (
        "LIMIT: x=0 baseline | RECOVERED: yes | FAILURE_IF_NOT: would contradict baseline | "
        "REPAIR_NEEDED: add limiting derivation"
    ),
    "convincing_case": (
        "ELEVATOR_VERSION: x completes y | TECHNICAL_VERSION: x modifies the baseline model | "
        "WHY_EXISTING_THEORY_LEAVES_ROOM: prior model leaves x unconstrained | "
        "WHY_PLAUSIBLE: mechanism is dimensionally consistent | SKEPTIC_TESTS: test t"
    ),
    "steelman_objection": (
        "STRONGEST_OBJECTION: x lacks experimental support | MECHANISM_OF_FAILURE: x fails at scale | "
        "THREATENING_RESULT: prior null model | WHAT_WOULD_KILL_IT: null measurement | "
        "FAIR_SUMMARY: promising but unvalidated"
    ),
    "predictor": (
        "OBSERVABLE: o | EFFECT_SIZE: 2x | DISCRIMINATES_FROM: null | MEASURABLE: yes | DETECTABLE: yes"
    ),
    "validation_designer": "TEST: t | CONFIRM_IF: c | REFUTE_IF: r | DISCRIMINATES_FROM: prior model | INFERENTIAL_STANDARD: p=0.05 | COST: low",
    "arbiter": "STATUS: internally_validated | LOAD_BEARING: c1 | DECISIVE_TEST: t",
    "prover": "DERIVATION: complete | GAPS: none | FATAL_GAP: no",
    "computation_designer": "no plan",
    "magnitude_designer": "",
}


@pytest.mark.asyncio
async def test_full_run_internally_validated(cfg):
    script = dict(BASE)
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: magnitude | SEVERITY: minor | STATUS: survived | "
        "TARGET: none | BASIS: inert-but-ok"
    )
    art = await run("seed", scripted(script), cfg, backend=FakeBackend())
    assert art.status == "internally_validated"
    assert art.completion.status == "completed_candidate"
    assert art.claim_graph[0].role == "novel_core"
    assert art.theory_bridge.theory_family == "field theory"
    assert art.prior_art_positioning.what_is_new == "discriminating mechanism"
    assert art.known_limits[0].recovered == "yes"
    assert "dimensionally consistent" in art.convincing_case.why_plausible


@pytest.mark.asyncio
async def test_fatal_attack_through_cap_needs_experiment(cfg):
    script = dict(BASE)
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: counterexample | SEVERITY: fatal | STATUS: landed | TARGET: c1 | BASIS: breaks"
    )
    script["repairer"] = "REPAIR: tried | TARGETS: c1 | RATIONALE: attempt"

    art = await run("seed", scripted(script), cfg, backend=FakeBackend())

    assert art.status == "needs_experiment"
    assert art.blocker["reason"] == "severe_objection"
    assert art.repairs_spent == cfg.gate.repair_cap
    assert art.finalized is True


@pytest.mark.asyncio
async def test_thin_surface_needs_experiment(cfg):
    script = dict(BASE)
    script["redteam"] = (
        "ATTEMPTED: counterexample\n"
        "ATTACK: counterexample | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: weak"
    )

    art = await run("seed", scripted(script), cfg, backend=FakeBackend())

    assert art.status == "needs_experiment"
    assert art.blocker["reason"] == "thin_attack_surface"


@pytest.mark.asyncio
async def test_repair_updates_target_claim_and_can_recover(cfg):
    redteam_bodies = iter([
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: counterexample | SEVERITY: fatal | STATUS: landed | TARGET: c1 | BASIS: breaks",
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: counterexample | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: fixed",
    ])
    script = dict(BASE)
    script["redteam"] = lambda messages: next(redteam_bodies)
    script["repairer"] = (
        "REPAIR: tightened mechanism | TARGETS: c1 | RATIONALE: fix mechanism\n"
        "CLAIM: c1 | STATEMENT: revised effect exists"
    )

    art = await run("seed", scripted(script), cfg, backend=FakeBackend())

    assert art.status == "internally_validated"
    assert art.repairs_spent == 1
    assert art.claim_graph[0].statement == "revised effect exists"


@pytest.mark.asyncio
async def test_major_attack_stays_needs_experiment(cfg):
    script = dict(BASE)
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: magnitude | SEVERITY: major | STATUS: landed | TARGET: c1 | BASIS: open issue"
    )
    script["repairer"] = "REPAIR: no safe repair | TARGETS: c1 | RATIONALE: still open"
    script["arbiter"] = "STATUS: needs_experiment | LOAD_BEARING: c1 | DECISIVE_TEST: t"

    art = await run("seed", scripted(script), cfg, backend=FakeBackend())

    assert art.status == "needs_experiment"
    assert art.blocker["reason"] == "open_objection"


@pytest.mark.asyncio
async def test_minor_landed_attack_does_not_block_validation(cfg):
    script = dict(BASE)
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: magnitude | SEVERITY: minor | STATUS: landed | TARGET: c1 | BASIS: small caveat"
    )

    art = await run("seed", scripted(script), cfg, backend=FakeBackend())

    assert art.status == "internally_validated"
