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
