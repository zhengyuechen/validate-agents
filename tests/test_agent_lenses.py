"""Tests for the Grounder and Prover per-claim lenses (Task 10)."""
import pytest
from valagents.agents.grounder import ground_claim, ground_novelty
from valagents.agents.prover import prove_claim
from valagents.artifact import AtomicClaim, FormalClaim
from valagents.web_search import Article
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="x", falsifiable=True)
CM = AtomicClaim(id="c1", statement="alpha not saturated", type="mechanistic")


# ---------------------------------------------------------------------------
# Grounder — base tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grounder_downgrades_without_independent_source(cfg):
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: thin"
    rec = await ground_claim(CM, FC, None, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "uncertain" and rec.independent_sources == 0   # D8


@pytest.mark.asyncio
async def test_grounder_supported_with_independent(cfg):
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1(Smith),A2(Lee) | BASIS: ok"
    rec = await ground_claim(CM, FC, None, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "pass" and rec.independent_sources == 2 and rec.lens == "grounder"


# ---------------------------------------------------------------------------
# Prover — base tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prover_definitional_wellformed(cfg):
    body = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    rec = await prove_claim(AtomicClaim(id="d1", statement="define X", type="definitional"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "pass" and rec.lens == "prover"


@pytest.mark.asyncio
async def test_prover_fatal_gap_fails(cfg):
    body = "DERIVATION: gapped | GAPS: d1 | FATAL_GAP: yes"
    rec = await prove_claim(AtomicClaim(id="d1", statement="x", type="mathematical"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "fail"


# ---------------------------------------------------------------------------
# Task 10 addendum — metadata capture for citations
# ---------------------------------------------------------------------------

class FakeBackend:
    """No-network backend returning canned Articles."""
    async def search(self, query: str, max_results: int = 5) -> list[Article]:
        return [
            Article(title="Alpha Saturation in Proteins", summary="...",
                    url="https://arxiv.org/abs/1234.5678", published="2022-03-15"),
            Article(title="Saturation Mechanisms", summary="...",
                    url="https://arxiv.org/abs/2345.6789", published="2021-07-01"),
        ]


@pytest.mark.asyncio
async def test_grounder_sources_carry_metadata(cfg):
    """Addendum: sources in the CheckRecord must carry title/url from retrieved Articles."""
    # LLM cites [A1] in the SOURCES tail
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | SOURCES: A1(Alpha) | BASIS: direct"
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "pass"
    assert len(rec.sources) == 1
    src = rec.sources[0]
    assert src.title == "Alpha Saturation in Proteins"
    assert src.url == "https://arxiv.org/abs/1234.5678"
    assert src.year == "2022"


@pytest.mark.asyncio
async def test_grounder_unmatched_label_bare_source(cfg):
    """A label that doesn't match any article keeps a bare Source with locator=label."""
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | SOURCES: A9(Nobody) | BASIS: ok"
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: body), cfg)
    # A9 doesn't exist in the 2-article result — should get a bare Source
    assert len(rec.sources) == 1
    assert rec.sources[0].title is None
    assert rec.sources[0].url is None
    assert rec.sources[0].locator == "A9(Nobody)"
