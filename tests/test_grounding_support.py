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


from valagents.grounding import _retrieval_saturated_tokens, _support_quote_valid, _content_tokens


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


# claim: "the noise PSD of YbZn2GaO5 is temperature-independent"
CLAIM = "the noise PSD of YbZn2GaO5 is temperature-independent"
PROP = "temperature-independent"
SUBJ_UNION = {"ybzn2gao5", "psd", "noise"}   # union = saturated ∪ subject_phrase tokens (caller-formed)
SRC = "We report that the noise PSD is temperature-independent below 1 K in this material."


def test_support_quote_valid_on_property():
    assert _support_quote_valid(
        "the noise PSD is temperature-independent below 1 K in this material",
        SRC, CLAIM, PROP, SUBJ_UNION, 6) is True


def test_support_quote_off_property_rejected():
    # a synthesis sentence sharing only the SUBJECT formula, no distinctive property token → fails the floor
    off = "single crystals of YbZn2GaO5 were grown by the floating-zone method in this study"
    assert _support_quote_valid(off, off, CLAIM, PROP, SUBJ_UNION, 6) is False


def test_support_quote_guard1_property_not_in_claim():
    # asserted_property carries a token absent from the claim → fail-closed
    assert _support_quote_valid(
        "the noise PSD is field-independent below 1 K in this material",
        "the noise PSD is field-independent below 1 K in this material",
        CLAIM, "field-independent", SUBJ_UNION, 6) is False


def test_support_quote_guard2_property_all_subject():
    # property is entirely subject tokens (property-as-subject) → prop_distinctive empty → fail
    assert _support_quote_valid(
        "the noise PSD of YbZn2GaO5 is reported in this material below 1 K",
        "the noise PSD of YbZn2GaO5 is reported in this material below 1 K",
        "the noise PSD of YbZn2GaO5", "noise PSD", {"ybzn2gao5", "psd", "noise"}, 6) is False


def test_support_quote_compound_fragment_rejected():
    # T2-D11 CRITICAL regression: a compound property ('temperature-independent' → {temperature, independent})
    # must NOT be credited by a quote sharing only ONE fragment. require-ALL closes this; any-overlap leaked.
    off_observable = "The magnetization shows strong temperature variation across the measured range here"   # only 'temperature'
    unrelated_sense = "The reported results were independent of the specific growth batch used here today"     # only 'independent'
    genuine = "We report the noise PSD is temperature-independent below 1 K in this material here"            # both
    assert _support_quote_valid(off_observable, off_observable, CLAIM, PROP, SUBJ_UNION, 6) is False
    assert _support_quote_valid(unrelated_sense, unrelated_sense, CLAIM, PROP, SUBJ_UNION, 6) is False
    assert _support_quote_valid(genuine, genuine, CLAIM, PROP, SUBJ_UNION, 6) is True


def test_support_quote_subject_subtraction_aids_recall():
    # Under require-ALL the subject subtraction is a RECALL aid: a genuine quote stating the full distinctive
    # property but OMITTING the subject formula still passes once the subject is subtracted; without
    # subtraction the same quote false-rejects (it would be forced to restate 'ybzn2gao5').
    leaky_prop = "YbZn2GaO5 temperature-independent"                       # property names the subject too (⊆ CLAIM)
    genuine = "the noise PSD is temperature-independent below 1 K in this material"   # omits 'YbZn2GaO5'
    assert _support_quote_valid(genuine, genuine, CLAIM, leaky_prop, set(), 6) is False          # no subtraction → forced to restate subject → reject
    assert _support_quote_valid(genuine, genuine, CLAIM, leaky_prop, {"ybzn2gao5"}, 6) is True   # subject subtracted → both present → pass
