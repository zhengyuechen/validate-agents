"""Tests for the Grounder and Prover per-claim lenses (Task 10)."""
import pytest
from valagents.agents.grounder import ground_claim, ground_novelty
from valagents.agents.prover import prove_claim, build_derivation
from valagents.artifact import AtomicClaim, CheckRecord, FormalClaim
from valagents.web_search import Article
from tests.fake_llm import FakeLLM

FC = FormalClaim(statement="x", falsifiable=True)
CM = AtomicClaim(id="c1", statement="alpha not saturated", type="mechanistic")


class FakeBackend:
    """No-network backend returning canned Articles."""
    async def search(self, query: str, max_results: int = 5) -> list[Article]:
        return [
            Article(title="Alpha Saturation in Proteins", summary="...",
                    url="https://arxiv.org/abs/1234.5678", published="2022-03-15"),
            Article(title="Saturation Mechanisms", summary="...",
                    url="https://arxiv.org/abs/2345.6789", published="2021-07-01"),
        ]


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
    """Fix 1: backend returns 2 articles; LLM cites A1,A2 → matched_independent=2 → pass."""
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: [A1],[A2] | BASIS: ok"
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "pass" and rec.independent_sources == 2 and rec.lens == "grounder"
    assert rec.sources[0].title == "Alpha Saturation in Proteins"
    assert rec.sources[0].url == "https://arxiv.org/abs/1234.5678"


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
    """Fix 1: unmatched label → bare source (url=None) → matched_independent=0 → min(1,0)=0 → uncertain."""
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | SOURCES: A9(Nobody) | BASIS: ok"
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: body), cfg)
    # A9 doesn't exist in the 2-article result — should get a bare Source
    assert len(rec.sources) == 1
    assert rec.sources[0].title is None
    assert rec.sources[0].url is None
    assert rec.sources[0].locator == "A9(Nobody)"
    assert rec.verdict == "uncertain"   # Fix 1: LLM claimed 1 but no matched source → cap to 0


# ---------------------------------------------------------------------------
# Fix 2 — fatal gap with empty basis must still be recorded in Derivation.gaps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_derivation_records_empty_basis_fatal_gap(cfg):
    """Fix 2: a fatal-gap CheckRecord with basis='' must not be silently dropped."""
    # Directly construct a CheckRecord with empty basis (the falsy case the old filter dropped)
    fatal_rec = CheckRecord(lens="prover", verdict="fail", basis="", independent_sources=0)
    claim = AtomicClaim(id="d1", statement="x", type="mathematical")
    claim.checks.append(fatal_rec)
    body = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"  # LLM body (unused by build_derivation)
    derivation = await build_derivation(FC, [claim], FakeLLM(lambda a, m: body), cfg)
    assert len(derivation.gaps) == 1   # must not be empty despite basis==""


# ---------------------------------------------------------------------------
# Fix 3 — lenient gapped parsing: "gapped with caveats" → uncertain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_prover_gapped_with_caveats_is_uncertain(cfg):
    """Fix 3: DERIVATION starting with 'gapped' (not exact match) → uncertain."""
    body = "DERIVATION: gapped with caveats | GAPS: x | FATAL_GAP: no"
    rec = await prove_claim(AtomicClaim(id="d1", statement="x", type="mathematical"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "uncertain"
