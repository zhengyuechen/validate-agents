from valagents.cli import run_cli
from valagents.references import Reference
from valagents.scheduler import run
from valagents.web_search import Article
from tests.fake_llm import FakeLLM


class FakeBackend:
    async def search(self, query, max_results=5):
        return [
            Article(title="Curl Descent", summary="curl", url="https://arxiv.org/abs/2401.12345", published="2024"),
            Article(title="Momentum", summary="momentum", url="https://arxiv.org/abs/2402.12345", published="2024"),
            Article(title="Saddles", summary="saddles", url="https://arxiv.org/abs/2403.12345", published="2024"),
            Article(title="Convergence", summary="convergence", url="https://arxiv.org/abs/2404.12345", published="2024"),
        ]


class FakeResolver:
    async def resolve(self, identifier):
        return Reference(
            locator="arxiv:2401.12345",
            title="Curl Descent",
            authors=["Smith"],
            year="2024",
            url="https://arxiv.org/abs/2401.12345",
            origin="provided",
        )


def router(agent, messages):
    content = messages[-1]["content"]
    if agent == "formalizer":
        return (
            "CLAIM: a curl term escapes strict saddles faster than GD | VARIABLES: theta, alpha "
            "| REGIME: strict saddles | FALSIFIABLE: yes"
        )
    if agent == "faithfulness":
        return "FAITHFUL: yes | BACK_TRANSLATION: rotation speeds saddle escape"
    if agent == "decomposer":
        return (
            "CLAIM: A | TYPE: mathematical | DEPENDS_ON: none | STATEMENT: curl projects on negative-curvature dir\n"
            "CLAIM: B | TYPE: mechanistic | DEPENDS_ON: none | STATEMENT: alpha does not saturate at the saddle\n"
            "CLAIM: C | TYPE: empirical | DEPENDS_ON: none | STATEMENT: rotation does not disrupt convergence"
        )
    if agent == "entailment":
        return "COVERS: complete | MISSING: none"
    if agent == "grounder":
        if "SUB-CLAIM" in content and "alpha" in content:
            return "CLAIM: B | SUPPORT: uncertain | INDEPENDENT_SOURCES: 0 | SOURCES: none | BASIS: saturation unclear"
        if "SUB-CLAIM" in content and "projects" in content:
            return "CLAIM: A | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A1,A2 | BASIS: standard"
        if "SUB-CLAIM" in content and "convergence" in content:
            return "CLAIM: C | SUPPORT: supported | INDEPENDENT_SOURCES: 2 | SOURCES: A3,A4 | BASIS: ok"
        return "CLOSEST_PRIOR: Curl-Descent | DELTA: alpha schedule | POSITION: new"
    if agent == "prover":
        if "alpha" in content:
            return "DERIVATION: gapped | GAPS: B | FATAL_GAP: no"
        return "DERIVATION: complete | GAPS: none | FATAL_GAP: no"
    if agent == "computation_designer":
        # PC-1b: mathematical claim A earns credit from a code-witnessed symbolic check (a real passing
        # plan: lim GM/r^2 as r->oo = 0), not the stripped prover say-so. B (mechanistic) stays the blocker.
        return ("EXPRESSION: G*M/r**2 | VARIABLES: G,M,r | LIMIT_VARIABLE: r | LIMIT_POINT: oo "
                "| EXPECTED: 0 | EXPECTED_SOURCE: textbook | CONFIRM_IF: limit is 0 | REFUTE_IF: differs")
    if agent == "predictor":
        return (
            "OBSERVABLE: mean escape time | EFFECT_SIZE: separates from GD | "
            "DISCRIMINATES_FROM: GD/momentum | MEASURABLE: yes | DETECTABLE: yes"
        )
    if agent == "redteam":
        return (
            "ATTEMPTED: counterexample, failure_regime, magnitude\n"
            "ATTACK: magnitude | SEVERITY: major | STATUS: survived | TARGET: B | BASIS: saturation bounded\n"
            "ATTACK: counterexample | SEVERITY: minor | STATUS: survived | TARGET: none | BASIS: none found"
        )
    if agent == "validation_designer":
        return (
            "TEST: synthetic-saddle escape-time vs GD/momentum/Curl-Descent | "
            "CONFIRM_IF: scaling separates | REFUTE_IF: no separation | "
            "DISCRIMINATES_FROM: Curl-Descent | INFERENTIAL_STANDARD: p=0.05 n=100 | COST: low"
        )
    if agent == "arbiter":
        return "STATUS: needs_experiment | LOAD_BEARING: B | DECISIVE_TEST: escape-time benchmark"
    return ""


async def test_escape_saddle_needs_experiment(cfg):
    art = await run(
        "adding an antisymmetric curl term to GD helps escape saddle points",
        FakeLLM(router),
        cfg,
        backend=FakeBackend(),
    )
    assert art.status == "needs_experiment"
    assert art.load_bearing == "B"
    assert art.blocker["reason"] == "inconclusive"
    claim_b = next(claim for claim in art.claim_graph if claim.id == "B")
    assert claim_b.status == "uncertain"
    assert len(claim_b.checks) >= cfg.gate.fanout_N
    assert art.validation_plan.cost == "low"


async def test_escape_saddle_cli_references(tmp_path, cfg):
    refs_path = tmp_path / "refs.txt"
    refs_path.write_text("https://arxiv.org/abs/2401.12345\n")

    out = await run_cli(
        "adding an antisymmetric curl term to GD helps escape saddle points",
        FakeLLM(router),
        cfg,
        backend=FakeBackend(),
        out_dir=str(tmp_path),
        references_path=str(refs_path),
        resolver=FakeResolver(),
    )

    report = open(out["report_path"]).read()
    bib = open(out["bib_path"]).read()
    assert out["artifact"].status == "needs_experiment"
    assert "[1]" in report
    assert "## References" in report
    assert "Curl Descent" in report
    assert "@article" in bib
