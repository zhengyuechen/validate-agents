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


def test_support_quote_thin_corpus_union_closes_leak():
    # THE THIN-CORPUS PROBE (§10). The model emits a property that SMUGGLES the subject formula
    # ('YbZn2GaO5 temperature-independent'). On a thin corpus the formula is in only 1 of 3 abstracts,
    # so saturation-alone does NOT subtract it → it survives in prop_distinctive → a formula-only quote
    # FALSELY passes (the leak). The UNION (subject_phrase carries 'ybzn2gao5') subtracts it → fails.
    arts = [_Art("ybzn2gao5 noise psd temperature-independent below 1 k"),
            _Art("unrelated spin liquid candidate magnetization"),
            _Art("another frustrated magnet heat capacity")]
    saturated = _retrieval_saturated_tokens(arts, 0.6)        # 'ybzn2gao5' in 1/3 → NOT saturated → {} here
    assert "ybzn2gao5" not in saturated
    leaky_prop = "YbZn2GaO5 temperature-independent"          # property smuggles the subject formula (⊆ CLAIM)
    formula_only = "single crystals of YbZn2GaO5 were grown by floating zone in this study here"
    # saturation-alone: the formula survives in prop_distinctive → formula-only quote PASSES (the leak)
    assert _support_quote_valid(formula_only, formula_only, CLAIM, leaky_prop, saturated, 6) is True
    # UNION: subject_phrase subtracts 'ybzn2gao5' → quote is off the distinctive property → FAILS (leak closed)
    union = saturated | _content_tokens("YbZn2GaO5 noise PSD")
    assert _support_quote_valid(formula_only, formula_only, CLAIM, leaky_prop, union, 6) is False
