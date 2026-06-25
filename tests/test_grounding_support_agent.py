import json
from valagents.config import Config
from valagents.agents.grounder import ground_claim
from valagents.artifact import AtomicClaim, FormalClaim
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


def _cfg():
    return Config(default_model="fake")


CLAIM = AtomicClaim(id="c1", statement="the noise PSD of YbZn2GaO5 is temperature-independent",
                    type="empirical")
FC = FormalClaim(statement="PSD(T) = const", falsifiable=True)

# Three real abstracts: A1 carries an on-property supporting sentence; A2 a synthesis-only sentence;
# A3 a contradicting sentence. URLs distinct arXiv ids.
A_SUPPORT = Article(
    title="Noise spectroscopy of YbZn2GaO5",
    summary=("We study YbZn2GaO5, a quantum-spin-liquid candidate. "
             "The measured noise PSD is temperature-independent below 1 K."),
    url="http://arxiv.org/abs/2501.00001v1", published="2025-01-01")
A_SYNTH = Article(
    title="Crystal growth of YbZn2GaO5",
    summary="Single crystals of YbZn2GaO5 were grown by the floating-zone method in this study.",
    url="http://arxiv.org/abs/2501.00002v1", published="2025-01-02")
A_CONTRA = Article(
    title="Strong T-dependence in YbZn2GaO5",
    summary="In YbZn2GaO5 the noise PSD shows a strong temperature dependence below 1 K.",
    url="http://arxiv.org/abs/2501.00003v1", published="2025-01-03")


class _Backend:
    def __init__(self, arts):
        self._arts = arts

    async def search(self, query, max_results=10):
        return list(self._arts)


def _body(tail, payload):
    return tail + "\n```json\n" + json.dumps(payload) + "\n```"


def _llm(tail, payload):
    return FakeLLM(lambda a, m: _body(tail, payload))


async def test_supports_with_on_property_quote_passes():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: direct on-property support"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "pass" and rec.independent_sources == 1


async def test_pass_basis_carries_honest_boundary():
    # §8: a credited pass must disclose IN THE BASIS that the credit is presence+topicality, not entailment/independence
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: direct on-property support"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "pass"
    assert "not code-witnessed" in rec.basis and "grounder credit" in rec.basis


async def test_fabricated_quote_uncertain():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: claimed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The PSD shows a clear 1/f temperature dependence nowhere in this abstract."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_off_property_synthesis_quote_uncertain():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: claimed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A2", "direction": "supports",
                              "quote": "Single crystals of YbZn2GaO5 were grown by the floating-zone method in this study."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_contradiction_guard_forces_uncertain_not_pass():
    # a valid supports AND a valid contradicts → the contradiction guard downgrades pass → uncertain (not fail)
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: mixed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [
                   {"label": "A1", "direction": "supports",
                    "quote": "The measured noise PSD is temperature-independent below 1 K."},
                   {"label": "A3", "direction": "contradicts",
                    "quote": "In YbZn2GaO5 the noise PSD shows a strong temperature dependence below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain"
    assert rec.basis.startswith("CONTRADICTION:")


async def test_dedup_preprint_and_published_count_once():
    # same arXiv id, two version URLs → one distinct work → code_witnessed capped to 1.
    # The backend carries 4 articles so 'independent' (in the 2 duplicates only, 2 of 4 < 0.6*4=2.4)
    # stays distinctive — otherwise the property co-saturates and nothing passes.
    a_v2 = Article(title=A_SUPPORT.title, summary=A_SUPPORT.summary,
                   url="http://arxiv.org/abs/2501.00001v2", published="2025-02-01")
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | BASIS: two cites same work"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [
                   {"label": "A1", "direction": "supports",
                    "quote": "The measured noise PSD is temperature-independent below 1 K."},
                   {"label": "A2", "direction": "supports",
                    "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, a_v2, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.independent_sources == 1


async def test_thin_corpus_formula_leak_uncertain():
    # §10 day-one probe: the model SMUGGLES the subject formula into the property
    # ('YbZn2GaO5 temperature-independent') and labels a formula-only synthesis sentence "supports".
    # On a thin corpus the formula is in only 1 of 3 abstracts (saturation misses it). The agent ALWAYS
    # forms the union (saturated ∪ subject_phrase), which subtracts the formula → off-property → uncertain.
    thin = [A_SYNTH,
            Article(title="x", summary="unrelated spin liquid candidate magnetization", url="http://arxiv.org/abs/2501.01001v1", published="2025"),
            Article(title="y", summary="another frustrated magnet heat capacity", url="http://arxiv.org/abs/2501.01002v1", published="2025")]
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: claimed"
    payload = {"asserted_property": "YbZn2GaO5 temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "Single crystals of YbZn2GaO5 were grown by the floating-zone method in this study."}]}
    rec = await ground_claim(CLAIM, FC, _Backend(thin), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_backend_off_uncertain():
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 3 | BASIS: claimed"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports", "quote": "anything"}]}
    rec = await ground_claim(CLAIM, FC, None, _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0


async def test_gate_purity_pass_requires_independent_at_least_one():
    # a VALID on-property supports quote (code_witnessed=1) but the LLM self-reports 0 independent →
    # min(0,1)=0 → the ≥1 bar still bites → uncertain. (3-article backend so 'independent' stays distinctive;
    # a 1-article corpus would co-saturate the whole property and fail for a different reason.)
    tail = "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 0 | BASIS: direct"
    payload = {"asserted_property": "temperature-independent", "subject_phrase": "YbZn2GaO5 noise PSD",
               "citations": [{"label": "A1", "direction": "supports",
                              "quote": "The measured noise PSD is temperature-independent below 1 K."}]}
    rec = await ground_claim(CLAIM, FC, _Backend([A_SUPPORT, A_SYNTH, A_CONTRA]), _llm(tail, payload), _cfg())
    assert rec.verdict == "uncertain" and rec.independent_sources == 0
