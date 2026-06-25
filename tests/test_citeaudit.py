from valagents.citeaudit import _title_match, _crossref_candidates, _Candidate


def test_title_match_requires_all_tokens():
    # name content-tokens (4) all present in the candidate title -> match
    assert _title_match("phase space quantum mechanics", "Quantum Mechanics in Phase Space", 3) is True


def test_title_match_min_token_gate():
    # short names (2 content tokens) are not title-like -> never attach, regardless of candidate
    assert _title_match("Born reciprocity", "Born Reciprocity and Reciprocal Relativity", 3) is False
    assert _title_match("prior model", "A Prior Model of Everything", 3) is False


def test_title_match_false_attach_rejected():
    # a name token absent from the candidate title -> no match (the harm we prevent)
    assert _title_match("spin liquid noise spectroscopy", "Spin Liquid Magnetization Study", 3) is False


def test_title_match_generic_overspecification_is_accepted_residual():
    # CA-D5: a generic 3-token name DOES match an arbitrary same-token title (real paper, human-checkable)
    assert _title_match("spin liquid model", "A New Spin Liquid Model Hamiltonian", 3) is True


def test_title_match_knob_4_kills_generic():
    # raising min_name_tokens to 4 demotes the 3-token generic name to unverified
    assert _title_match("spin liquid model", "A New Spin Liquid Model Hamiltonian", 4) is False


def test_crossref_candidates_parse():
    data = {"message": {"items": [
        {"title": ["Quantum Mechanics in Phase Space"],
         "author": [{"given": "C.", "family": "Zachos"}, {"given": "D.", "family": "Fairlie"}],
         "published-print": {"date-parts": [[2005]]}, "DOI": "10.1142/5287"},
        {"title": [], "author": []},  # titleless item -> skipped
    ]}}
    cands = _crossref_candidates(data)
    assert len(cands) == 1
    assert cands[0].title == "Quantum Mechanics in Phase Space"
    assert cands[0].authors == ["C. Zachos", "D. Fairlie"]
    assert cands[0].year == "2005"
    assert cands[0].url == "https://doi.org/10.1142/5287"


def test_crossref_candidates_empty():
    assert _crossref_candidates({}) == []
    assert _crossref_candidates({"message": {"items": []}}) == []


import pytest
from valagents.citeaudit import CiteResult, CiteAuditor, audit_narrative_refs
from valagents.config import Config
from valagents.artifact import IdeaArtifact, PriorArtPositioning


def _cfg():
    return Config(default_model="fake")


class _Art:  # minimal arxiv Article stand-in (title/url/published)
    def __init__(self, title, url="http://arxiv.org/abs/2501.00001v1", published="2025-01-01"):
        self.title = title
        self.url = url
        self.published = published
        self.summary = ""


class _FakeArxiv:
    def __init__(self, arts, raises=False):
        self._arts = arts
        self._raises = raises

    async def search(self, query, max_results=5):
        if self._raises:
            raise RuntimeError("network down")
        return list(self._arts)


async def _crossref_none(name, rows):
    return []


def _crossref_with(cands):
    async def _search(name, rows):
        return list(cands)
    return _search


async def test_audit_resolves_via_arxiv():
    arx = _FakeArxiv([_Art("Quantum Mechanics in Phase Space", url="http://arxiv.org/abs/2501.05287")])
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "resolved"
    assert r.reference.origin == "asserted"
    assert r.reference.title == "Quantum Mechanics in Phase Space"
    assert r.reference.locator == "arxiv:2501.05287"


async def test_audit_falls_back_to_crossref():
    arx = _FakeArxiv([])  # arXiv finds nothing
    cross = _crossref_with([_Candidate("Quantum Mechanics in Phase Space", ["C. Zachos"], "2005",
                                       "https://doi.org/10.1142/5287")])
    auditor = CiteAuditor(arx, cross, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "resolved" and r.reference.locator == "10.1142/5287"
    assert r.reference.authors == ["C. Zachos"]


async def test_audit_unverified_when_no_match():
    arx = _FakeArxiv([_Art("An Unrelated Paper About Something Else Entirely")])
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "unverified" and r.reference is None


async def test_audit_short_name_skips_search():
    arx = _FakeArxiv([_Art("Born Reciprocity and Reciprocal Relativity")], raises=True)  # would raise if searched
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    r = await auditor.audit("Born reciprocity")  # 2 content tokens < 3 -> no search, unverified
    assert r.status == "unverified"


async def test_audit_fail_soft_on_arxiv_error():
    arx = _FakeArxiv([], raises=True)  # arXiv raises -> swallowed
    cross = _crossref_with([])
    auditor = CiteAuditor(arx, cross, _cfg())
    r = await auditor.audit("phase space quantum mechanics")
    assert r.status == "unverified"  # no crash


async def test_audit_narrative_refs_scope_and_dedup():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(
        closest_prior="phase space quantum mechanics",
        must_cite=["phase space quantum mechanics", "spin liquid noise spectroscopy"],  # 1st dup of closest_prior
    ))
    arx = _FakeArxiv([_Art("Quantum Mechanics in Phase Space")])
    auditor = CiteAuditor(arx, _crossref_none, _cfg())
    out = await audit_narrative_refs(art, auditor)
    assert set(out) == {"phase space quantum mechanics", "spin liquid noise spectroscopy"}
    assert out["phase space quantum mechanics"].status == "resolved"


async def test_audit_narrative_refs_off_returns_empty():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(closest_prior="x y z"))
    assert await audit_narrative_refs(art, None) == {}
