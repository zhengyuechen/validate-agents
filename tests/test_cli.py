import json

from valagents.artifact import IdeaArtifact
from valagents.cli import render_report, run_cli
from valagents.references import Reference
from tests.test_scheduler_repair import BASE, FakeBackend, scripted


class FakeResolver:
    async def resolve(self, identifier):
        return Reference(
            locator="arxiv:2401.12345",
            title="Provided Curl",
            authors=["Smith"],
            year="2024",
            url="https://arxiv.org/abs/2401.12345",
            origin="provided",
        )


def test_report_carries_limit_sentence_and_status():
    art = IdeaArtifact(raw_idea="seed")
    md = render_report(art)
    assert "never 'true'" in md
    assert "**Status:**" in md
    assert "raw_idea" not in md


async def test_run_cli_writes_json_and_report(tmp_path, cfg):
    script = dict(BASE)
    script["grounder"] = (
        "CLAIM: c1 | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: thin\n"
        "CLOSEST_PRIOR: p | DELTA: d | POSITION: new"
    )
    script["prover"] = "DERIVATION: gapped | GAPS: c1 | FATAL_GAP: no"
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: magnitude | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: ok"
    )
    script["arbiter"] = "STATUS: needs_experiment | LOAD_BEARING: c1 | DECISIVE_TEST: t"

    out = await run_cli("seed", scripted(script), cfg, out_dir=str(tmp_path))

    assert out["artifact"].status in ("needs_experiment", "internally_validated", "refuted")
    data = json.loads(open(out["json_path"]).read())
    assert data["raw_idea"] == "seed"
    assert "status" in data
    assert "never 'true'" in open(out["report_path"]).read()


async def test_run_cli_accepts_explicit_run_id(tmp_path, cfg):
    script = dict(BASE)
    script["prover"] = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: magnitude | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: ok"
    )

    out = await run_cli("seed with id", scripted(script), cfg, out_dir=str(tmp_path), run_id="web-run")

    assert out["json_path"].endswith("web-run.json")
    assert out["report_path"].endswith("web-run.md")
    assert (tmp_path / ".logs" / "web-run.jsonl").parent.exists()
    outputs_path = tmp_path / ".agent_outputs" / "web-run.jsonl"
    assert outputs_path.exists()
    outputs = [json.loads(line) for line in outputs_path.read_text().splitlines()]
    assert {row["agent"] for row in outputs} >= {"formalizer", "faithfulness", "decomposer"}
    assert all("body" in row and "parse_status" in row for row in outputs)


async def test_run_cli_writes_references_bib_and_markers(tmp_path, cfg):
    script = dict(BASE)
    script["grounder"] = (
        "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | SOURCES: A1 | BASIS: ok\n"
        "CLOSEST_PRIOR: p | DELTA: d | POSITION: new"
    )
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: magnitude | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: ok"
    )
    refs_path = tmp_path / "refs.txt"
    refs_path.write_text("https://arxiv.org/abs/2401.12345\n")

    out = await run_cli(
        "seed refs",
        scripted(script),
        cfg,
        backend=FakeBackend(),
        out_dir=str(tmp_path),
        references_path=str(refs_path),
        resolver=FakeResolver(),
    )

    report = open(out["report_path"]).read()
    bib = open(out["bib_path"]).read()
    assert "[1]" in report
    assert "## References" in report
    assert "Provided Curl" in report
    assert "@article" in bib
