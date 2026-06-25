import json
from valagents.config import Config
from valagents.agents.value_grounder import ground_plan
from valagents.computation import ComputationPlan
from tests.fake_llm import FakeLLM

def _cfg():
    return Config(default_model="fake")

ABSTRACT = "We report the ordered moment per Yb ion of 1.2e-3 µB at T = 0.4 K in the candidate QSL."

def _resolver(text=ABSTRACT):
    class R:
        async def fetch(self, locator):    # the injected fetch contract used by ground_plan
            return (text, {"locator": locator, "title": "T", "url": "u", "year": "2024"})
    return R()

EXTRACTION = {"extracted_value": "1.2e-3", "source_unit_token": "µB",
              "referent": "ordered moment per Yb ion", "source_conditions": "T = 0.4 K",
              "verbatim_quote": "the ordered moment per Yb ion of 1.2e-3 µB at T = 0.4 K"}

def _llm(ext=EXTRACTION):
    body = "```json\n" + json.dumps(ext) + "\n```"
    return FakeLLM(lambda a, m: body if a == "value_grounder" else "")

def _bplan():
    return ComputationPlan(kind="magnitude", comparison_kind="bound_check",
        predicted_effect="1e-3", bound="1.2e-3", bound_source="arXiv:2104.01234",
        source_quantity="Yb³⁺ effective magnetic moment", claim_conditions="T < 1 K", source_unit="µB")

async def test_ground_plan_off_when_no_resolver():
    assert await ground_plan(_bplan(), None, _llm(), _cfg()) is None

async def test_ground_plan_supports():
    r = await ground_plan(_bplan(), _resolver(), _llm(), _cfg())
    assert r is not None and r.status == "supports"

async def test_ground_plan_fabricated_quote_unconfirmed():
    bad = {**EXTRACTION, "verbatim_quote": "a moment of 1.2e-3 µB nowhere in the abstract text"}
    r = await ground_plan(_bplan(), _resolver(), _llm(bad), _cfg())
    assert r is not None and r.status == "unconfirmed"

async def test_ground_plan_unresolvable_unconfirmed():
    class Dead:
        async def fetch(self, locator):
            return None
    r = await ground_plan(_bplan(), Dead(), _llm(), _cfg())
    assert r is not None and r.status == "unconfirmed"
