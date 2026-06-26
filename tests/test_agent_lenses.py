"""Tests for the Grounder and Prover per-claim lenses (Task 10)."""
import json as _json
import pytest
from valagents.agents.grounder import ground_claim, ground_novelty
from valagents.agents.prover import prove_claim, build_derivation
from valagents.artifact import AtomicClaim, CheckRecord, FormalClaim
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


def _grounder_body(tail: str, payload: dict) -> str:
    return tail + "\n```json\n" + _json.dumps(payload) + "\n```"

FC = FormalClaim(statement="x", falsifiable=True)
CM = AtomicClaim(id="c1", statement="alpha not saturated", type="mechanistic")


class FakeBackend:
    """No-network backend returning canned Articles with real abstracts (Tier-2: quotes must be code-checkable)."""
    async def search(self, query: str, max_results: int = 10) -> list[Article]:
        return [
            Article(title="Alpha Saturation in Proteins",
                    summary="We report that alpha is not saturated under physiological conditions in this work.",
                    url="https://arxiv.org/abs/1234.5678", published="2022-03-15"),
            Article(title="Saturation Mechanisms",
                    summary="A second independent group finds that alpha is not saturated below threshold here.",
                    url="https://arxiv.org/abs/2345.6789", published="2021-07-01"),
            Article(title="Saturation Observed",
                    summary="In contrast, alpha reaches clear saturation at high concentration in our samples.",
                    url="https://arxiv.org/abs/3456.7890", published="2023-01-01"),
            Article(title="Alpha Kinetics",
                    summary="The alpha pathway kinetics were characterized in detail in this study.",
                    url="https://arxiv.org/abs/4567.8901", published="2023-02-01"),
            Article(title="Alpha Review",
                    summary="A broad review of the alpha regulatory system is presented here today.",
                    url="https://arxiv.org/abs/5678.9012", published="2023-03-01"),
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
    """Two retrieved works each carry a code-checked on-property supporting quote → pass, 2 independent."""
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | BASIS: ok"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A1", "direction": "supports",
         "quote": "We report that alpha is not saturated under physiological conditions in this work."},
        {"label": "A2", "direction": "supports",
         "quote": "A second independent group finds that alpha is not saturated below threshold here."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "pass" and rec.independent_sources == 2 and rec.lens == "grounder"
    assert rec.sources[0].title == "Alpha Saturation in Proteins"
    assert rec.sources[0].url == "https://arxiv.org/abs/1234.5678"


@pytest.mark.asyncio
async def test_grounder_contradiction_is_recorded_not_refuting(cfg):
    """A code-admissible contradicting quote forces uncertain (not fail) and is surfaced loud in basis."""
    tail = "CLAIM: c1 | SUPPORT: unsupported | INDEPENDENT_SOURCES: 1 | BASIS: retrieved work disagrees"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A3", "direction": "contradicts",
         "quote": "In contrast, alpha reaches clear saturation at high concentration in our samples."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "uncertain"          # not refuting — the grounder never auto-fails a novel claim
    assert rec.basis.startswith("CONTRADICTION:")


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
async def test_prover_pass_does_not_self_credit(cfg):
    # PC-1a: 'DERIVATION: complete' is model say-so — the prover verdict may pass but must NEVER earn
    # an independent source (was indep=1 for definitional/mathematical pre-PC-1a). Credit now requires
    # a code-witness (grounder quote or PC-1b symbolic executor); the prover keeps only refutation power.
    body = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    for t in ("definitional", "mathematical"):
        rec = await prove_claim(AtomicClaim(id="d1", statement="x", type=t),
                                FC, FakeLLM(lambda a, m: body), cfg)
        assert rec.verdict == "pass" and rec.independent_sources == 0


@pytest.mark.asyncio
async def test_prover_fatal_gap_is_uncertain_and_repairable(cfg):
    body = "DERIVATION: gapped | GAPS: d1 | FATAL_GAP: yes"
    rec = await prove_claim(AtomicClaim(id="d1", statement="x", type="mathematical"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "uncertain"
    assert rec.basis.startswith("FATAL_DERIVATION_GAP:")


@pytest.mark.asyncio
async def test_prover_explicit_contradiction_fails(cfg):
    body = "DERIVATION: gapped | GAPS: CONTRADICTION: violates premise p | FATAL_GAP: yes"
    rec = await prove_claim(AtomicClaim(id="d1", statement="x", type="mathematical"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "fail"


@pytest.mark.asyncio
async def test_prover_refutes_prefix_fails(cfg):
    """Fix 2: GAPS starting with REFUTES: must escalate to verdict='fail', same as CONTRADICTION."""
    body = "DERIVATION: gapped | GAPS: REFUTES: prior result contradicts this | FATAL_GAP: yes"
    rec = await prove_claim(AtomicClaim(id="d1", statement="x", type="mathematical"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "fail"


# ---------------------------------------------------------------------------
# Task 10 addendum — metadata capture for citations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grounder_sources_carry_metadata(cfg):
    """Sources in the CheckRecord carry title/url/year from the retrieved Articles whose quotes passed."""
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: direct"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A1", "direction": "supports",
         "quote": "We report that alpha is not saturated under physiological conditions in this work."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "pass"
    assert len(rec.sources) == 1
    src = rec.sources[0]
    assert src.title == "Alpha Saturation in Proteins"
    assert src.url == "https://arxiv.org/abs/1234.5678"
    assert src.year == "2022"


@pytest.mark.asyncio
async def test_grounder_unmatched_label_dropped(cfg):
    """A citation whose label was not retrieved is dropped — it cannot manufacture credit."""
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: ok"
    payload = {"asserted_property": "not saturated", "subject_phrase": "alpha", "citations": [
        {"label": "A9", "direction": "supports",
         "quote": "Some sentence about an article that was never retrieved at all here."}]}
    rec = await ground_claim(CM, FC, FakeBackend(), FakeLLM(lambda a, m: _grounder_body(tail, payload)), cfg)
    assert rec.verdict == "uncertain" and rec.independent_sources == 0 and rec.sources == []


# ---------------------------------------------------------------------------
# Fix 2 — fatal gap with empty basis must still be recorded in Derivation.gaps
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_derivation_records_empty_basis_fatal_gap(cfg):
    """Fix 2: a fatal-gap CheckRecord with basis='' must not be silently dropped."""
    # Directly construct a CheckRecord with empty basis (the falsy case the old filter dropped)
    fatal_rec = CheckRecord(
        lens="prover",
        verdict="uncertain",
        basis="FATAL_DERIVATION_GAP: ",
        independent_sources=0,
    )
    claim = AtomicClaim(id="d1", statement="x", type="mathematical")
    claim.checks.append(fatal_rec)
    body = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"  # LLM body (unused by build_derivation)
    derivation = await build_derivation(FC, [claim], FakeLLM(lambda a, m: body), cfg)
    assert len(derivation.gaps) == 1   # must not be empty despite basis==""
    assert derivation.gaps[0].fatal is True


@pytest.mark.asyncio
async def test_mechanistic_prover_pass_is_not_external_support(cfg):
    body = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    rec = await prove_claim(AtomicClaim(id="m1", statement="mechanism", type="mechanistic"),
                            FC, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "pass"
    assert rec.independent_sources == 0


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


# ---------------------------------------------------------------------------
# Cap-bypass regression guard — no-backend path must NOT trust LLM source count
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grounder_no_backend_trusts_no_llm_count(cfg):
    """Regression: backend=None → matched_independent=0 → min(3,0)=0 → verdict is 'uncertain',
    NOT 'pass'. Catches anyone re-introducing a 'no backend → trust LLM count' bypass."""
    body = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 3 | SOURCES: [A1] | BASIS: ok"
    rec = await ground_claim(CM, FC, None, FakeLLM(lambda a, m: body), cfg)
    assert rec.verdict == "uncertain"  # matched_independent=0 → min(3,0)=0 → not pass
