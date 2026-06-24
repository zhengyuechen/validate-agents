import pytest
from valagents.agents.formalizer import formalize
from valagents.agents.base import map_support_to_verdict, as_int
from tests.fake_llm import FakeLLM


@pytest.mark.asyncio
async def test_formalizer_pins_claim(cfg):
    body = ("reasoning\nCLAIM: escape time falls with a curl term | VARIABLES: theta, alpha "
            "| REGIME: strict saddles | FALSIFIABLE: yes")
    fc = await formalize("curl term helps escape saddles", FakeLLM(lambda a, m: body), cfg)
    assert fc.falsifiable is True and "curl" in fc.statement


@pytest.mark.asyncio
async def test_formalizer_not_falsifiable(cfg):
    body = "CLAIM: it is elegant | VARIABLES: none | REGIME: any | FALSIFIABLE: no"
    fc = await formalize("seed", FakeLLM(lambda a, m: body), cfg)
    assert fc.falsifiable is False


@pytest.mark.asyncio
async def test_formalizer_double_fail_returns_none(cfg):
    fc = await formalize("seed", FakeLLM(lambda a, m: "no tail at all"), cfg)
    assert fc is None


def test_support_downgrade_without_independent_source():
    assert map_support_to_verdict("supported", 0) == "uncertain"   # D8
    assert map_support_to_verdict("supported", 1) == "pass"   # exact >=1 boundary
    assert map_support_to_verdict("supported", 2) == "pass"
    assert map_support_to_verdict("unsupported", 5) == "uncertain"


def test_as_int_extracts_first_integer():
    assert as_int("3") == 3
    assert as_int("3-5") == 3      # range token -> low end, not 0
    assert as_int("none") == 0
