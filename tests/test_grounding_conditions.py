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


# C1: axis-identified but unit-uncanonicalizable source clause must block confirmation
def test_c1_uncanonicalizable_field_unit_blocks():
    assert cc("T < 1 K", "T = 0.4 K, B = 5 tesla") is False      # spelled-out unit, axis identified
    assert cc("T < 1 K", "T = 0.4 K, H = 50 kOe") is False       # prefixed unit; 50 kOe = 5 T
    assert cc("T < 1 K", "T = 0.4 K, B = 5 millitesla") is False  # 5 mT non-zero, unconstrained axis


# M1: multiple clauses on one axis — ALL points must satisfy the claim
def test_m1_multi_point_axis_all_must_satisfy():
    assert cc("T < 1 K", "measured from T = 300 K down to T = 0.4 K") is False  # 300 K fails
    assert cc("T < 1 K", "measured from T = 0.4 K up to T = 300 K") is False    # order-independent


# N1: µK (MICRO SIGN U+00B5) must round-trip through NFKC → matches after key uses GREEK MU form
def test_n1_micro_kelvin_nfkc():
    assert cc("T < 1 K", "T = 400 µK") is True   # 400 µK << 1 K, should confirm
