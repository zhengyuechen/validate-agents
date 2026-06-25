import math
from valagents.grounding import SCALE_TABLE, convert

def test_convert_same_dimension():
    assert math.isclose(convert(1.0, "meV", "K"), 11.6045, rel_tol=1e-3)   # energy via k_B
    assert math.isclose(convert(12.0, "meV", "K"), 139.25, rel_tol=1e-3)
    assert convert(1.0, "Oe", "T") == 1e-4

def test_convert_cross_dimension_is_none():
    assert convert(1.0, "µB", "T") is None        # moment vs field

def test_convert_out_of_table_is_none():
    assert convert(1.0, "emu/g", "µB") is None     # whole-token: emu/g is NOT a key
    assert convert(1.0, "emu", "µB") == 1.078e20    # bare emu IS a moment

def test_scale_table_factor_references():
    # G-D9: every entry passes a both-directions reference test within reference_rel_tol (1e-3).
    refs = [("meV", "K", 11.6045), ("cm^-1", "K", 1.4388), ("Oe", "T", 1e-4), ("emu", "µB", 1.078e20)]
    for a, b, expected in refs:
        got = convert(1.0, a, b)
        assert got is not None
        assert max(got / expected, expected / got) - 1 < 1e-3

def test_factor_reference_catches_a_wrong_factor():
    # a deliberately-wrong meV factor must fail the reference check (the guard)
    bad = 1.5e-22 / SCALE_TABLE["K"][1]            # ~10.9, off from 11.6045 by ~6%
    assert max(bad / 11.6045, 11.6045 / bad) - 1 >= 1e-3
