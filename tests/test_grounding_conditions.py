from valagents.grounding import _conditions_compatible as cc

def test_temperature_within():
    assert cc("T < 1 K", "T = 0.4 K") is True
    assert cc("T < 1 K", "T = 300 K") is False

def test_sub_kelvin_ladder():
    assert cc("T < 1 K", "400 mK") is True          # mK ladder, NOT SCALE_TABLE's energy-K
    assert cc("T < 1 K", "T = 400 mK") is True

def test_field_axis():
    assert cc("B < 0.1 T", "B = 5 mT") is True
    assert cc("B < 0.1 T", "B = 5 T") is False

def test_g_d5c_claim_silent_field_nonzero_source():
    # claim silent on field, source pins B=5 T -> NOT confirmed (the F1 hole)
    assert cc("T < 1 K", "T = 0.4 K, B = 5 T") is False
    # source B=0 (baseline) is permissive
    assert cc("T < 1 K", "T = 0.4 K, B = 0 T") is True
    assert cc("T < 1 K", "T = 0.4 K, B = 0") is True

def test_equals_clause_exact_only():
    assert cc("T = 0.3 K", "T = 0.3 K") is True
    assert cc("T = 0.3 K", "T = 0.4 K") is False

def test_absent_or_unparseable_not_confirmed():
    assert cc("T < 1 K", "measured at low temperature") is False   # no parseable source clause
    assert cc("", "T = 0.4 K") is False                            # empty claim -> not confirmed
