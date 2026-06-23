import pytest
from valagents.agents.faithfulness import faithfulness_check
from valagents.agents.decomposer import decompose
from valagents.agents.entailment import entailment_check
from valagents.artifact import FormalClaim
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="escape time falls with curl term", falsifiable=True)

@pytest.mark.asyncio
async def test_faithfulness_yes(cfg):
    body = "BACK_TRANSLATION: rotation speeds saddle escape\nFAITHFUL: yes | BACK_TRANSLATION: rotation speeds escape"
    f = await faithfulness_check("curl helps escape saddles", FC, FakeLLM(lambda a, m: body), cfg)
    assert f.verdict == "yes"

@pytest.mark.asyncio
async def test_faithfulness_narrowed_records_retried_flag(cfg):
    body = "FAITHFUL: narrowed | BACK_TRANSLATION: only decoherence"
    f = await faithfulness_check("is collapse physical", FC, FakeLLM(lambda a, m: body), cfg, retried=True)
    assert f.verdict == "narrowed" and f.retried is True

@pytest.mark.asyncio
async def test_decompose_builds_graph(cfg):
    body = ("CLAIM: A | TYPE: mathematical | DEPENDS_ON: none | STATEMENT: projection nonzero\n"
            "CLAIM: B | TYPE: mechanistic | DEPENDS_ON: none | STATEMENT: alpha not saturated\n"
            "CLAIM: C | TYPE: empirical | DEPENDS_ON: A | STATEMENT: converges near minima")
    claims = await decompose(FC, FakeLLM(lambda a, m: body), cfg)
    assert [c.id for c in claims] == ["A", "B", "C"]
    assert claims[2].depends_on == ["A"] and claims[0].type == "mathematical"

@pytest.mark.asyncio
async def test_decompose_empty_on_failure(cfg):
    claims = await decompose(FC, FakeLLM(lambda a, m: "no rows"), cfg)
    assert claims == []

@pytest.mark.asyncio
async def test_entailment_gap(cfg):
    body = "COVERS: gap | MISSING: the load-bearing nonzero-projection step"
    cov = await entailment_check(FC, [], FakeLLM(lambda a, m: body), cfg)
    assert cov.verdict == "gap" and "projection" in cov.missing
