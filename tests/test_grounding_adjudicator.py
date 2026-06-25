from valagents.config import GroundCfg

def test_groundcfg_knobs():
    g = GroundCfg()
    assert g.supports_factor == 2.0 and g.contradict_factor == 10.0
    assert g.quote_min_tokens == 6 and g.reference_rel_tol == 1e-3
