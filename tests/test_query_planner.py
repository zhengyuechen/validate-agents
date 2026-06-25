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


# --- Task 3: planned_search ladder ---
from valagents.agents.query_planner import planned_search
from valagents.web_search import Article


def _pool(n):
    return [Article(title=f"T{i}", summary="s", url=f"http://arxiv.org/abs/x{i}v1", published="2025") for i in range(n)]


async def test_planned_search_scoped_then_widens_keywords_not_scope(monkeypatch):
    pool, queries, calls = _pool(5), [], {"n": 0}
    async def fake_search(self, query, max_results=10):
        queries.append(query); calls["n"] += 1
        return pool[:1] if calls["n"] == 1 else pool        # thin first hit -> widen fires
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: 'ARCHIVES: cond-mat | TERMS: hole, "Hall coefficient"')
    fmt, arts, block = await planned_search(ArxivBackend(), "claim text", llm, _cfg())
    assert queries == ['(cat:cond-mat*) AND (hole AND "Hall coefficient")',
                       '(cat:cond-mat*) AND (hole OR "Hall coefficient")']
    assert block["rung"] == "scoped" and block["widened"] is True and block["n_hits"] == 5
    assert block["archives"] == ["cond-mat"] and block["rendered"] == queries[1]


async def test_planned_search_terms_only_when_no_valid_archive(monkeypatch):
    queries = []
    async def fake_search(self, query, max_results=10):
        queries.append(query); return _pool(5)
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: "ARCHIVES: frobnicate | TERMS: hole, gap")
    fmt, arts, block = await planned_search(ArxivBackend(), "the full claim sentence", llm, _cfg())
    assert block["rung"] == "terms_only"
    assert queries[0] == "(hole AND gap)"                   # no cat:, and NEVER the raw sentence
    assert "claim" not in queries[0]


async def test_planned_search_raw_when_planner_disabled(monkeypatch):
    cfg = Config(default_model="fake"); cfg.grounding.query_planner = False
    queries = []
    async def fake_search(self, query, max_results=10):
        queries.append(query); return _pool(1)
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: "ARCHIVES: cond-mat | TERMS: hole")     # ignored: planner off
    fmt, arts, block = await planned_search(ArxivBackend(), "the full claim sentence", llm, cfg)
    assert block["rung"] == "raw" and queries == ["the full claim sentence"]   # single call, current behavior


async def test_planned_search_raw_on_planner_collapse(monkeypatch):
    queries = []
    async def fake_search(self, query, max_results=10):
        queries.append(query); return _pool(1)
    monkeypatch.setattr(ArxivBackend, "search", fake_search)
    llm = FakeLLM(lambda a, m: "no machine-readable tail here")        # plan_query -> empty
    fmt, arts, block = await planned_search(ArxivBackend(), "raw claim", llm, _cfg())
    assert block["rung"] == "raw" and queries == ["raw claim"]
