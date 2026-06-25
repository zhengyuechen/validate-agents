import json

from valagents.artifact import (
    AtomicClaim,
    ConvincingCase,
    FormalClaim,
    IdeaArtifact,
    IdeaCompletion,
    KnownLimit,
    Prediction,
    PriorArtPositioning,
    SteelmanObjection,
    TheoryBridge,
    ValidationPlan,
)
from valagents.cli import render_report, run_cli
from valagents.references import Reference
from tests.test_scheduler_repair import BASE, FakeBackend, scripted, _grounder_body


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
    report = open(out["report_path"]).read()
    assert "never 'true'" in report
    assert "## Completed Candidate" in report
    assert "## Theory Bridge" in report
    assert "## Known Limits" in report


async def test_run_cli_accepts_explicit_run_id(tmp_path, cfg):
    script = dict(BASE)
    script["prover"] = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    script["redteam"] = (
        "ATTEMPTED: counterexample, magnitude\n"
        "ATTACK: magnitude | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: ok"
    )

    out = await run_cli("seed with id", scripted(script), cfg, out_dir=str(tmp_path), run_id="web-run")

    assert out["json_path"].endswith("web-run/artifact.json")
    assert out["report_path"].endswith("web-run/report.md")
    assert (tmp_path / "web-run" / "logs.jsonl").exists()
    outputs_path = tmp_path / "web-run" / "agent_outputs.jsonl"
    assert outputs_path.exists()
    outputs = [json.loads(line) for line in outputs_path.read_text().splitlines()]
    assert {row["agent"] for row in outputs} >= {"formalizer", "faithfulness", "decomposer"}
    assert all("body" in row and "parse_status" in row for row in outputs)


async def test_run_cli_writes_references_bib_and_markers(tmp_path, cfg):
    script = dict(BASE)
    script["grounder"] = _grounder_body(
        "CLAIM: c1 | SUPPORT: supported | INDEPENDENT_SOURCES: 1 | BASIS: ok\n"
        "CLOSEST_PRIOR: p | DELTA: d | POSITION: new",
        {"asserted_property": "exists", "subject_phrase": "effect",
         "citations": [{"label": "A1", "direction": "supports",
                        "quote": "The effect exists and has been experimentally confirmed here."}]}
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


def _populated_artifact() -> IdeaArtifact:
    """Return an artifact with all major sections populated."""
    claim = AtomicClaim(
        id="c1",
        statement="test claim",
        type="empirical",
        role="novel_core",
        load_bearing=True,
    )
    return IdeaArtifact(
        raw_idea="test idea",
        formal_claim=FormalClaim(statement="formal test claim", falsifiable=True),
        claim_graph=[claim],
        completion=IdeaCompletion(
            status="completed_candidate",
            completed_idea="a complete idea",
            mechanism="some mechanism",
            weakest_link="the weak link",
        ),
        theory_bridge=TheoryBridge(theory_family="QFT"),
        prior_art_positioning=PriorArtPositioning(
            closest_prior="prior work",
            similarity="similar in X",
            difference="different in Y",
            what_is_new="Z is new",
        ),
        known_limits=[KnownLimit(limit="finite size", recovered="no")],
        convincing_case=ConvincingCase(
            elevator_version="short pitch",
            technical_version="long pitch",
            why_existing_theory_leaves_room="gap",
            why_plausible="plausible",
        ),
        steelman_objection=SteelmanObjection(
            strongest_objection="strong objection text",
            mechanism_of_failure="fails via X",
            threatening_result="result R",
            what_would_kill_it="observation O",
            fair_summary="skeptic concludes: weak",
        ),
        predictions=[
            Prediction(observable="obs1", measurable=True, detectable="yes")
        ],
        validation_plan=ValidationPlan(
            decisive_test="run experiment E",
            confirm_if="result > threshold",
            refute_if="result < threshold",
            cost="low",
        ),
    )


def test_report_order_verdict_before_objection_before_case_for():
    """Strongest objection + decisive test must appear BEFORE the case-for section."""
    art = _populated_artifact()
    report = render_report(art)
    assert "Strongest objection" in report
    assert "Decisive test" in report
    assert "Case for" in report
    assert report.index("Strongest objection") < report.index("Case for")
    assert report.index("Decisive test") < report.index("Case for")


def test_report_full_order_invariant():
    """Populated artifact: verdict → strongest objection → decisive test →
    case for → case against → claims → completion → references."""
    art = _populated_artifact()
    refs = [
        Reference(
            locator="arxiv:0000.00000",
            title="Test Ref",
            authors=["Author"],
            year="2024",
            url="https://example.com",
            origin="provided",
        )
    ]
    report = render_report(art, refs)

    # Collect positions of key markers
    idx_verdict = report.index("**Verdict:**")
    idx_objection = report.index("Strongest objection")
    idx_decisive = report.index("Decisive test")
    idx_case_for = report.index("Case for")
    idx_case_against = report.index("Case against")
    idx_claims = report.index("## Claim Graph")
    idx_completion = report.index("## Completed Candidate")
    idx_references = report.index("## References")

    assert idx_verdict < idx_objection
    assert idx_objection < idx_decisive
    assert idx_decisive < idx_case_for
    assert idx_case_for < idx_case_against
    assert idx_case_against < idx_claims
    assert idx_claims < idx_completion
    assert idx_completion < idx_references


def test_report_no_section_dropped():
    """All sections present in the populated artifact render."""
    art = _populated_artifact()
    report = render_report(art)
    assert "**Verdict:**" in report
    assert "Strongest objection" in report
    assert "Decisive test" in report
    assert "Case for" in report
    assert "Case against" in report
    assert "## Claim Graph" in report
    assert "## Completed Candidate" in report
    assert "## Theory Bridge" in report
    assert "## Prior-Art Positioning" in report
    assert "## Known Limits" in report
    assert "## Predictions" in report
    assert "never 'true'" in report


def test_report_ill_posed_reframe_not_experiment():
    """ill_posed verdict must include the 'reframe, not experiment' phrase."""
    from valagents.artifact import FormalClaim, Faithfulness

    art = IdeaArtifact(
        raw_idea="unfalsifiable idea",
        formal_claim=FormalClaim(statement="not falsifiable", falsifiable=False),
        faithfulness=Faithfulness(verdict="yes"),
        finalized=True,
    )
    report = render_report(art)
    assert art.verdict_class == "ill_posed"
    assert "reframing, not an experiment" in report
