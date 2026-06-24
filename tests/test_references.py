from valagents.references import (
    Reference,
    build_references,
    collect_retrieved,
    detect_kind,
    markers_for_claim,
    normalize_id,
    to_bibtex,
)
from valagents.artifact import IdeaArtifact, AtomicClaim, CheckRecord, Source


class FakeResolver:
    def __init__(self, table):
        self.table = table

    async def resolve(self, identifier):
        return self.table.get(normalize_id(identifier))


def test_detect_kind():
    assert detect_kind("https://arxiv.org/abs/2401.12345") == "arxiv"
    assert detect_kind("arXiv:2401.12345") == "arxiv"
    assert detect_kind("10.1103/PhysRevLett.1.1") == "doi"
    assert detect_kind("https://doi.org/10.1/x") == "doi"


def _art(src):
    c = AtomicClaim(
        id="c1",
        statement="s",
        type="empirical",
        checks=[
            CheckRecord(
                lens="grounder",
                verdict="pass",
                independent_sources=1,
                sources=[src],
            )
        ],
    )
    return IdeaArtifact(raw_idea="s", claim_graph=[c])


def test_collect_retrieved_carries_metadata_and_cited_by():
    art = _art(
        Source(
            locator="https://arxiv.org/abs/2401.12345",
            title="Curl",
            url="https://arxiv.org/abs/2401.12345",
            relation="independent",
        )
    )
    refs = collect_retrieved(art)
    assert refs[0].title == "Curl"
    assert refs[0].cited_by == ["c1"]
    assert refs[0].origin == "retrieved"


async def test_build_dedups_provided_and_retrieved(tmp_path):
    art = _art(Source(locator="arxiv:2401.12345", title="Curl", relation="independent"))
    p = tmp_path / "refs.txt"
    p.write_text("https://arxiv.org/abs/2401.12345\n")
    resolver = FakeResolver(
        {
            "arxiv:2401.12345": Reference(
                locator="arxiv:2401.12345",
                title="Curl Descent",
                authors=["Smith"],
                year="2024",
                url="u",
                origin="provided",
            )
        }
    )
    refs = await build_references(art, str(p), resolver)
    assert len(refs) == 1
    assert refs[0].origin == "provided"
    assert refs[0].title == "Curl Descent"
    assert refs[0].cited_by == ["c1"]
    assert refs[0].number == 1
    assert refs[0].key


async def test_unresolved_kept(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("10.9999/nope\n")
    refs = await build_references(IdeaArtifact(raw_idea="s"), str(p), FakeResolver({}))
    assert refs[0].unresolved is True


def test_to_bibtex():
    r = Reference(
        locator="arxiv:1",
        key="smith2024",
        title="T",
        authors=["Smith"],
        year="2024",
        url="u",
        number=1,
    )
    bib = to_bibtex([r])
    assert "@article{smith2024," in bib
    assert "title = {T}" in bib


def test_markers_for_claim():
    refs = [
        Reference(locator="a", number=1, cited_by=["c1"]),
        Reference(locator="b", number=2, cited_by=["c2"]),
    ]
    assert markers_for_claim(refs, "c1") == [1]
