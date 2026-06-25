import pytest
from valagents.references import Reference, build_references, normalize_id
from valagents.artifact import IdeaArtifact, AtomicClaim, CheckRecord, Source, PriorArtPositioning
from valagents.citeaudit import CiteResult
from valagents.cli import render_report


def _resolved_ref(locator, title, url):
    return Reference(locator=normalize_id(locator), title=title, url=url, year="2005",
                     authors=["A. Author"], origin="asserted")


async def test_build_references_merges_asserted_fresh_locator():
    art = IdeaArtifact(raw_idea="s")
    asserted = [_resolved_ref("https://doi.org/10.1142/5287", "Quantum Mechanics in Phase Space",
                              "https://doi.org/10.1142/5287")]
    refs = await build_references(art, asserted_refs=asserted)
    assert len(refs) == 1 and refs[0].origin == "asserted" and refs[0].number == 1


async def test_build_references_collision_existing_wins():
    # a claim-cited (retrieved) ref AND an asserted ref share a locator -> ONE entry, retrieved kept (CA-D8)
    art = IdeaArtifact(raw_idea="s", claim_graph=[AtomicClaim(
        id="c1", statement="s", type="empirical",
        checks=[CheckRecord(lens="grounder", verdict="pass", independent_sources=1,
                            sources=[Source(locator="arxiv:2501.00001", title="Retrieved Title",
                                            url="http://arxiv.org/abs/2501.00001", relation="independent")])])])
    asserted = [_resolved_ref("http://arxiv.org/abs/2501.00001v2", "Asserted Title",
                              "http://arxiv.org/abs/2501.00001v2")]
    refs = await build_references(art, asserted_refs=asserted)
    assert len(refs) == 1
    r = refs[0]
    assert r.origin == "retrieved"          # existing wins (CA-D8)
    assert r.cited_by == ["c1"]             # preserved, not blanked
    assert r.title == "Retrieved Title"     # not clobbered by the asserted ref


def test_render_annotates_resolved_and_unverified_with_gloss():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(
        closest_prior="phase space quantum mechanics", must_cite=["made up nonexistent theorem paper"]))
    ref = _resolved_ref("https://doi.org/10.1142/5287", "Quantum Mechanics in Phase Space",
                        "https://doi.org/10.1142/5287")
    ref.number = 1
    audit_map = {
        "phase space quantum mechanics": CiteResult("phase space quantum mechanics", "resolved", ref),
        "made up nonexistent theorem paper": CiteResult("made up nonexistent theorem paper", "unverified"),
    }
    report = render_report(art, [ref], audit_map)
    assert "Quantum Mechanics in Phase Space" in report and "[1]" in report   # resolved -> loud title + [n]
    assert "[unverified]" in report
    assert "not resolved to a catalogued record" in report                    # the gloss


def test_render_off_no_markers():
    art = IdeaArtifact(raw_idea="s", prior_art_positioning=PriorArtPositioning(closest_prior="phase space qm"))
    report = render_report(art, [], None)   # audit_map None -> off
    assert "[unverified]" not in report and "not resolved to a catalogued record" not in report
