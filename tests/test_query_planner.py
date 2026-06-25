"""arXiv query planner: pure renderer, the planner LLM call, and the planned_search ladder."""
from valagents.agents.query_planner import PlannedQuery, VALID_ARCHIVES, render_query
from valagents.web_search import ArxivBackend


def test_render_scoped_arxiv_two_archives():
    p = PlannedQuery(archives=["cond-mat", "quant-ph"], terms=["hole", "Hall coefficient"])
    assert render_query(p, ArxivBackend()) == '(cat:cond-mat* OR cat:quant-ph*) AND (hole AND "Hall coefficient")'


def test_render_widen_or_keeps_cat_scope_identical():
    p = PlannedQuery(archives=["cond-mat"], terms=["hole", "Hall coefficient"])
    tight = render_query(p, ArxivBackend(), widen=False)
    wide = render_query(p, ArxivBackend(), widen=True)
    assert tight == '(cat:cond-mat*) AND (hole AND "Hall coefficient")'
    assert wide == '(cat:cond-mat*) AND (hole OR "Hall coefficient")'
    assert tight.split(" AND (", 1)[0] == wide.split(" AND (", 1)[0]   # cat: clause byte-identical


def test_render_normalizes_prequoted_term():
    p = PlannedQuery(archives=["cond-mat"], terms=['"Hall coefficient"'])
    assert render_query(p, ArxivBackend()) == '(cat:cond-mat*) AND ("Hall coefficient")'   # not ""Hall coefficient""


def test_render_terms_only_when_no_archives():
    p = PlannedQuery(archives=[], terms=["hole", "Hall coefficient"])
    assert render_query(p, ArxivBackend()) == '(hole AND "Hall coefficient")'              # no cat:


def test_render_nonarxiv_space_join_no_operators():
    p = PlannedQuery(archives=["cond-mat"], terms=["hole", "Hall coefficient"])
    assert render_query(p, None) == "hole Hall coefficient"                                 # backend_label(None)=="none"


def test_valid_archives_complete():
    for a in ("cond-mat", "quant-ph", "eess", "nlin", "q-bio", "q-fin", "stat", "econ"):
        assert a in VALID_ARCHIVES


# --- Task 2: plan_query ---
from valagents.config import Config
from valagents.agents.query_planner import plan_query
from tests.fake_llm import FakeLLM


def _cfg():
    return Config(default_model="fake")


async def test_plan_query_parses_archives_and_terms():
    llm = FakeLLM(lambda a, m: 'ARCHIVES: cond-mat, quant-ph | TERMS: hole, "Hall coefficient", superconductor')
    p = await plan_query("a metal superconducts only if its carriers are holes", llm, _cfg())
    assert p.archives == ["cond-mat", "quant-ph"]
    assert p.terms == ["hole", '"Hall coefficient"', "superconductor"]


async def test_plan_query_truncates_leaf_and_drops_hallucinated_archive():
    llm = FakeLLM(lambda a, m: "ARCHIVES: cond-mat.supr-con, frobnicate | TERMS: hole, gap")
    p = await plan_query("x", llm, _cfg())
    assert p.archives == ["cond-mat"]                    # leaf -> archive; 'frobnicate' dropped


async def test_plan_query_caps_two_archives_and_four_terms():
    llm = FakeLLM(lambda a, m: "ARCHIVES: cond-mat, quant-ph, hep-th | TERMS: a, b, c, d, e, f")
    p = await plan_query("x", llm, _cfg())
    assert p.archives == ["cond-mat", "quant-ph"]
    assert p.terms == ["a", "b", "c", "d"]


async def test_plan_query_forwards_context_into_prompt():
    seen = {}
    def router(agent, messages):
        seen["user"] = messages[-1]["content"]
        return "ARCHIVES: cond-mat | TERMS: moment, anisotropy"
    p = await plan_query("the effective moment is 1.2 muB", FakeLLM(router), _cfg(),
                         context="a frustrated magnet realizes a quantum spin liquid")
    assert "frustrated magnet" in seen["user"]           # context reached the planner prompt
    assert p.archives == ["cond-mat"]


async def test_plan_query_failsoft_empty_on_unparseable():
    p = await plan_query("x", FakeLLM(lambda a, m: "I cannot help."), _cfg())
    assert p == PlannedQuery() and p.archives == [] and p.terms == []
