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
