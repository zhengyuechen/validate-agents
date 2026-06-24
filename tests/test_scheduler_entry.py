import pytest
from valagents.scheduler import run_entry_gates
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact
from tests.fake_llm import FakeLLM


def store():
    return ArtifactStore(IdeaArtifact(raw_idea="seed"))


def router(script):
    def r(agent, messages):
        out = script.get(agent)
        return out(messages) if callable(out) else out
    return r


@pytest.mark.asyncio
async def test_not_falsifiable_terminates(cfg):
    s = store()
    llm = FakeLLM(router({"formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: no"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is False and s.current.status == "refuted" and s.current.blocker["reason"] == "not_falsifiable"


@pytest.mark.asyncio
async def test_unfaithful_retries_then_refuted(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: narrow x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: narrowed | BACK_TRANSLATION: only a special case",
        "decomposer": "CLAIM: A | TYPE: empirical | DEPENDS_ON: none | STATEMENT: s"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is False and s.current.status == "refuted"
    assert s.current.blocker["reason"] == "unfaithful_narrowed" and s.current.faithfulness.retried is True


@pytest.mark.asyncio
async def test_empty_decomposition_ill_formed(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
        "decomposer": "no rows here"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is False and s.current.status == "refuted" and s.current.blocker["reason"] == "ill_formed"


@pytest.mark.asyncio
async def test_unformalizable_terminates(cfg):
    s = store()
    llm = FakeLLM(router({"formalizer": "I cannot parse this into a formal claim."}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is False
    assert s.current.status == "refuted"
    assert s.current.blocker["reason"] == "unformalizable"


@pytest.mark.asyncio
async def test_clean_entry_proceeds(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
        "decomposer": "CLAIM: A | TYPE: empirical | DEPENDS_ON: none | STATEMENT: s",
        "entailment": "COVERS: complete | MISSING: none"}))
    proceed = await run_entry_gates(s, "seed", None, llm, cfg)
    assert proceed is True and len(s.current.claim_graph) == 1 and s.current.coverage.verdict == "complete"


@pytest.mark.asyncio
async def test_malformed_faithfulness_label_refutes_instead_of_raising(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: maybe | BACK_TRANSLATION: ambiguous",
    }))

    proceed = await run_entry_gates(s, "seed", None, llm, cfg)

    assert proceed is False
    assert s.current.status == "refuted"
    assert s.current.blocker["reason"] == "unfaithful_drift"


@pytest.mark.asyncio
async def test_malformed_claim_type_becomes_ill_formed_instead_of_raising(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
        "decomposer": "CLAIM: A | TYPE: math | DEPENDS_ON: none | STATEMENT: s",
    }))

    proceed = await run_entry_gates(s, "seed", None, llm, cfg)

    assert proceed is False
    assert s.current.status == "refuted"
    assert s.current.blocker["reason"] == "ill_formed"


@pytest.mark.asyncio
async def test_malformed_entailment_label_becomes_gap_instead_of_raising(cfg):
    s = store()
    llm = FakeLLM(router({
        "formalizer": "CLAIM: x | VARIABLES: n | REGIME: any | FALSIFIABLE: yes",
        "faithfulness": "FAITHFUL: yes | BACK_TRANSLATION: same",
        "decomposer": "CLAIM: A | TYPE: empirical | DEPENDS_ON: none | STATEMENT: s",
        "entailment": "COVERS: maybe | MISSING: unclear",
    }))

    proceed = await run_entry_gates(s, "seed", None, llm, cfg)

    assert proceed is True
    assert s.current.coverage.verdict == "gap"
