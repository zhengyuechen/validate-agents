from valagents.grounding import _quote_valid, _quantity_overlap, _parse_floats

TEXT = "We find the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K below the transition."

def test_parse_floats():
    got = _parse_floats("value 1.2e-3 µB and 0.4 K")
    assert any(abs(x - 1.2e-3) < 1e-9 for x in got) and any(abs(x - 0.4) < 1e-9 for x in got)

def test_quote_valid_full():
    q = "the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K"
    assert _quote_valid(q, TEXT, 1.2e-3, "µB", "ordered moment per Yb ion", 6) is True

def test_quote_not_in_bytes():
    assert _quote_valid("a fabricated 1.2e-3 µB ordered moment never written here", TEXT, 1.2e-3, "µB", "moment", 6) is False

def test_quote_degenerate_bare_number():
    assert _quote_valid("1.2e-3", TEXT, 1.2e-3, "µB", "moment", 6) is False   # no unit, no referent, too short

def test_quote_missing_unit():
    q = "the ordered moment per Yb ion saturates at 1.2e-3 at T = 0.4 K"      # fabricated (no µB) -> not in bytes anyway
    assert _quote_valid(q, TEXT, 1.2e-3, "µB", "moment", 6) is False

def test_quantity_overlap():
    assert _quantity_overlap("ordered moment per Yb ion", "Yb³⁺ effective magnetic moment") is True
    assert _quantity_overlap("applied magnetic field", "Yb³⁺ magnetic moment") is False   # field vs moment
    assert _quantity_overlap("µ_eff", "effective moment") is False                        # symbol↔prose miss (safe)


# --- R1b: whole-token unit check for single-char units (§6 / final-review R1b) ---

# Text containing 'T' only inside words (strength / transition / the / at) — no standalone Tesla token.
_HOLLOW_T_TEXT = "the field strength is 5.0 at the transition point here"

def test_hollow_T_rejected():
    """Single-char unit 'T' present only inside words must NOT satisfy the unit gate."""
    assert _quote_valid(_HOLLOW_T_TEXT, _HOLLOW_T_TEXT, 5.0, "T", "field strength", 6) is False

_HONEST_T_TEXT = "the applied field is 5.0 T at the sample stage here today"

def test_honest_T_spaced_accepted():
    """Standalone 'T' with a space separator must pass the unit gate."""
    assert _quote_valid(_HONEST_T_TEXT, _HONEST_T_TEXT, 5.0, "T", "applied field", 6) is True

_ATTACHED_T_TEXT = "the applied field is 5.0T at the sample stage here today now"

def test_honest_T_attached_accepted():
    """'T' directly attached to a numeral (5.0T) must pass the unit gate."""
    assert _quote_valid(_ATTACHED_T_TEXT, _ATTACHED_T_TEXT, 5.0, "T", "applied field", 6) is True

def test_multi_char_unit_unaffected():
    """Multi-char units (µB, meV) must still pass — regression guard for existing Task-3 cases."""
    assert _quote_valid(
        "the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K",
        TEXT,
        1.2e-3, "µB", "ordered moment per Yb ion", 6,
    ) is True
