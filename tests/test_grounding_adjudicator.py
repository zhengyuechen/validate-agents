from valagents.config import GroundCfg

def test_groundcfg_knobs():
    g = GroundCfg()
    assert g.supports_factor == 2.0 and g.contradict_factor == 10.0
    assert g.quote_min_tokens == 6 and g.reference_rel_tol == 1e-3

from valagents.config import Config
from valagents.grounding import ground_value, GroundingResult

def _cfg():
    return Config(default_model="fake")

TEXT = "We find the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K below 1 K."

def _ext(value="1.2e-3", unit="µB", referent="ordered moment per Yb ion",
         cond="T = 0.4 K", quote="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 0.4 K"):
    return {"extracted_value": value, "source_unit_token": unit, "referent": referent,
            "source_conditions": cond, "verbatim_quote": quote}

def _g(**kw):
    a = dict(asserted_value="1.2e-3", source_unit="µB", source_quantity="Yb³⁺ effective magnetic moment",
             claim_conditions="T < 1 K", extraction=_ext(), fetched_text=TEXT)
    a.update(kw)
    return ground_value(a["asserted_value"], a["source_unit"], a["source_quantity"],
                        a["claim_conditions"], a["extraction"], a["fetched_text"], _cfg())

def test_supports():
    r = _g()
    assert r.status == "supports" and r.quote and r.converted_value is not None

def test_not_found_unconfirmed():
    assert _g(extraction=None).status == "unconfirmed"

def test_fabricated_quote_unconfirmed():
    assert _g(extraction=_ext(quote="a 1.2e-3 µB moment not in the source text at all here")).status == "unconfirmed"

def test_out_of_table_unit_unconfirmed():
    assert _g(extraction=_ext(unit="emu/g",
              quote="the magnetization per gram is 1.2e-3 emu/g at T = 0.4 K low temperature"),
              source_unit="µB").status == "unconfirmed"

def test_quantity_mismatch_unconfirmed():
    r = _g(source_quantity="applied magnetic field strength")   # referent 'moment' vs 'field' -> no overlap
    assert r.status == "unconfirmed"

def test_wrong_conditions_inconclusive():
    r = _g(extraction=_ext(cond="T = 300 K", quote="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K"),
           fetched_text="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K room temperature")
    assert r.status == "inconclusive" and r.reason == "conditions_unconfirmed"

def test_numeric_inconclusive():
    r = _g(asserted_value="4e-4")     # ratio 1.2e-3/4e-4 = 3.0 in [2,10) -> inconclusive
    assert r.status == "inconclusive" and r.reason == "numeric_inconclusive"

def test_contradicts_same_regime():
    r = _g(asserted_value="1e-4")     # ratio 12 >= 10, conditions compatible -> contradicts
    assert r.status == "contradicts"

def test_no_false_contradict_wrong_regime():
    r = _g(asserted_value="1e-4",
           extraction=_ext(cond="T = 300 K", quote="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K"),
           fetched_text="the ordered moment per Yb ion saturates at 1.2e-3 µB at T = 300 K room temperature")
    assert r.status == "inconclusive"   # ratio gross BUT conditions incompatible -> not a contradiction

def test_code_owns_conversion():
    r = _g(asserted_value="139", source_unit="K",
           extraction=_ext(value="12", unit="meV", referent="exchange energy J",
                           cond="T = 0.4 K", quote="the exchange energy J is 12 meV at T = 0.4 K low temperature"),
           source_quantity="exchange energy", claim_conditions="T < 1 K",
           fetched_text="the exchange energy J is 12 meV at T = 0.4 K low temperature")
    assert r.status == "supports"   # 12 meV -> 139.25 K, ratio ~ 1
