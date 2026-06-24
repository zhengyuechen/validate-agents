import pytest

from valagents.scheduler import run
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


class FakeBackend:
    async def search(self, query, max_results=5):
        return [
            Article(title="A", summary="support", url="https://example.com/a", published="2026"),
            Article(title="B", summary="support", url="https://example.com/b", published="2026"),
        ]


def scripted(script):
    def route(agent, messages):
        value = script[agent]
        return value(messages) if callable(value) else value
    return FakeLLM(route)


BASE = {
    "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
    "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
    "decomposer": "CLAIM: c1 | TYPE: empirical | DEPENDS_ON: none | STATEMENT: effect exists",
    "entailment": "COVERS: complete | MISSING: none",
    "grounder": (
        "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | "
        "SOURCES: A1,A2 | BASIS: ok\n"
        "CLOSEST_PRIOR: p | DELTA: d | POSITION: new"
    ),
    "predictor": (
        "OBSERVABLE: o | EFFECT_SIZE: 2x | DISCRIMINATES_FROM: null | MEASURABLE: yes"
    ),
    "validation_designer": "TEST: t | CONFIRM_IF: c | REFUTE_IF: r | COST: low",
    "arbiter": "STATUS: internally_validated | LOAD_BEARING: c1 | DECISIVE_TEST: t",
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


@pytest.mark.asyncio
async def test_fatal_attack_through_cap_refuted(cfg):
    script = dict(BASE)
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: counterexample | SEVERITY: fatal | STATUS: landed | TARGET: c1 | BASIS: breaks"
    )
    script["repairer"] = "REPAIR: tried | TARGETS: c1 | RATIONALE: attempt"

    art = await run("seed", scripted(script), cfg, backend=FakeBackend())

    assert art.status == "refuted"
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
