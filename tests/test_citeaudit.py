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
