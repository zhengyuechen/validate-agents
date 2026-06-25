from __future__ import annotations

from valagents.grounding import _split_sentences, _sentence_bounded, _quote_admissible

ABS = ("We study YbZn2GaO5, a quantum-spin-liquid candidate. "
       "The measured noise PSD is temperature-independent below 1 K. "
       "Samples were grown by the floating-zone method, e.g. as in prior work.")


def test_split_does_not_break_on_abbreviation_or_decimal():
    # "e.g." (lowercase after the period) and "1 K." (digit/space, no capital after) must NOT split mid-sentence.
    sents = _split_sentences("The PSD saturates at 0.5 K. We grew samples, e.g. by floating zone.")
    assert sents == ["The PSD saturates at 0.5 K.", "We grew samples, e.g. by floating zone."]


def test_sentence_bounded_true_within_one_sentence():
    assert _sentence_bounded("noise PSD is temperature-independent below 1 K", ABS) is True


def test_sentence_bounded_false_across_boundary():
    # spans the '. ' boundary between sentence 2 and 3 — the splice-inversion attack
    assert _sentence_bounded("below 1 K. Samples were grown", ABS) is False


def test_quote_admissible_full():
    assert _quote_admissible("the measured noise PSD is temperature-independent below 1 K", ABS, 6) is True


def test_quote_admissible_fabricated_not_in_bytes():
    assert _quote_admissible("the PSD shows clear 1/f temperature dependence here", ABS, 6) is False


def test_quote_admissible_too_short():
    assert _quote_admissible("temperature-independent below", ABS, 6) is False


def test_quote_admissible_cross_sentence_rejected():
    assert _quote_admissible("temperature-independent below 1 K. Samples were grown by", ABS, 6) is False


from valagents.grounding import _retrieval_saturated_tokens, _support_quote_valid


class _Art:
    def __init__(self, summary):
        self.summary = summary


def test_retrieval_saturated_picks_common_tokens():
    arts = [_Art("ybzn2gao5 spin liquid noise"),
            _Art("ybzn2gao5 magnetization study"),
            _Art("ybzn2gao5 heat capacity")]
    sat = _retrieval_saturated_tokens(arts, 0.6)   # 'ybzn2gao5' in 3/3 ; others in 1/3
    assert "ybzn2gao5" in sat and "noise" not in sat


def test_retrieval_saturated_empty_corpus():
    assert _retrieval_saturated_tokens([], 0.6) == set()


# claim "the noise PSD of YbZn2GaO5 is temperature-independent"; the distinctive set is CODE-derived as
# claim_tokens − subject_tokens. With the saturated subject {ybzn2gao5, psd, noise} subtracted, the
# distinctive property is {temperature, independent}. The model never supplies the property (T2-D12).
CLAIM = "the noise PSD of YbZn2GaO5 is temperature-independent"
SUBJECT = {"ybzn2gao5", "psd", "noise"}   # code-derived retrieval-saturated subject (passed by the caller)
SRC = "We report that the noise PSD is temperature-independent below 1 K in this material."


def test_support_quote_valid_on_property():
    assert _support_quote_valid(
        "the noise PSD is temperature-independent below 1 K in this material",
        SRC, CLAIM, SUBJECT, 6) is True


def test_support_quote_off_property_rejected():
    # a synthesis sentence sharing only the SUBJECT formula, no distinctive property token → fails the floor
    off = "single crystals of YbZn2GaO5 were grown by the floating-zone method in this study"
    assert _support_quote_valid(off, off, CLAIM, SUBJECT, 6) is False


def test_support_quote_guard_claim_fully_saturated():
    # subject_tokens cover every claim content token → distinctive empty → fail-closed (uncertain)
    full = {"ybzn2gao5", "psd", "noise", "temperature", "independent"}
    assert _support_quote_valid(SRC, SRC, CLAIM, full, 6) is False


def test_support_quote_compound_fragment_rejected():
    # T2-D11/D12 regression: a compound property ('temperature-independent' → {temperature, independent})
    # must NOT be credited by a quote sharing only ONE fragment. require-ALL closes it; the distinctive set
    # is CLAIM-derived so the model cannot under-declare to dodge it.
    off_observable = "The magnetization shows strong temperature variation across the measured range here"   # only 'temperature'
    unrelated_sense = "The reported results were independent of the specific growth batch used here today"     # only 'independent'
    genuine = "We report the noise PSD is temperature-independent below 1 K in this material here"            # both
    assert _support_quote_valid(off_observable, off_observable, CLAIM, SUBJECT, 6) is False
    assert _support_quote_valid(unrelated_sense, unrelated_sense, CLAIM, SUBJECT, 6) is False
    assert _support_quote_valid(genuine, genuine, CLAIM, SUBJECT, 6) is True


def test_support_quote_under_declaration_closed():
    # T2-D12 (final-review CRITICAL): the distinctive set is CLAIM-derived, so a model cannot shrink the
    # required set by under-declaring a property. Compound-property claim; a quote sharing only a generic
    # fragment ('linear') fails, the genuine quote passes.
    claim = "the specific heat shows a linear temperature dependence"
    subject = {"specific", "heat"}   # code-derived saturated subject on a specific-heat corpus
    attack = "The fit to the data uses a linear background subtraction across the full measured range here"
    genuine = "the specific heat shows a clear linear temperature dependence below 1 K here today"
    assert _support_quote_valid(attack, attack, claim, subject, 6) is False
    assert _support_quote_valid(genuine, genuine, claim, subject, 6) is True


def test_support_quote_saturation_aids_recall():
    # Subtracting the saturated subject is a RECALL aid: a genuine quote stating the property but OMITTING the
    # subject formula passes once the subject is subtracted, and false-rejects without subtraction.
    genuine = "the noise PSD is temperature-independent below 1 K in this material"   # omits 'YbZn2GaO5'
    assert _support_quote_valid(genuine, genuine, CLAIM, set(), 6) is False         # nothing subtracted → must restate 'ybzn2gao5' → reject
    assert _support_quote_valid(genuine, genuine, CLAIM, {"ybzn2gao5"}, 6) is True  # subject subtracted → all distinctive tokens present → pass
