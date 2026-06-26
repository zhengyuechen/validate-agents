import inspect
import pytest
from valagents.agents.computation_designer import design_computation
from valagents.computation import verdict_to_check, ComputationPlan, ComputationResult, ComputationVerdict
from valagents.artifact import FormalClaim, AtomicClaim, IdeaArtifact, KnownLimit
from valagents.config import Config
from valagents.store import ArtifactStore
from valagents.scheduler import inject_limit_checks
from tests.fake_llm import FakeLLM

def cfg():
    return Config(default_model="fake")

async def test_designer_emits_structured_plan_only():
    body = ("EXPRESSION: G*M/r**2*(1+a/c**2) | VARIABLES: G,M,r,a,c | LIMIT_VARIABLE: c "
            "| LIMIT_POINT: oo | EXPECTED: G*M/r**2 | EXPECTED_SOURCE: Newtonian gravity "
            "| CONFIRM_IF: limit equals GM/r^2 | REFUTE_IF: limit differs")
    claim = AtomicClaim(id="L1", statement="recovers Newtonian gravity", type="mathematical")
    plan = await design_computation(claim, FormalClaim(statement="x", falsifiable=True),
                                    FakeLLM(lambda a, m: body), cfg())
    assert isinstance(plan, ComputationPlan)
    assert plan.limit_variable == "c" and plan.expected == "G*M/r**2" and plan.variables == ["G","M","r","a","c"]

async def test_designer_returns_none_on_unparseable():
    claim = AtomicClaim(id="L1", statement="s", type="mathematical")
    plan = await design_computation(claim, FormalClaim(statement="x", falsifiable=True),
                                    FakeLLM(lambda a, m: "no tail"), cfg())
    assert plan is None

def test_designer_returns_no_verdict():                 # F1: it designs, it does not judge
    src = inspect.getsource(design_computation)
    assert "ComputationVerdict" not in src and "verdict" not in src.replace("# ", "")

def test_verdict_to_check_pass_is_independent():
    p = ComputationPlan(expression="x", variables=["x"], limit_variable="x", limit_point="0",
                        expected="0", expected_source="src")
    v = ComputationVerdict(verdict="pass", measured="0", plan=p,
                           result=ComputationResult(ok=True, computed="0", matched="confirm"))
    rec = verdict_to_check(v, tick=0)
    assert rec.lens == "executor" and rec.verdict == "pass" and rec.independent_sources == 1
    assert "expected = 0" in rec.basis and "src" in rec.basis      # caveat surfaced in basis

def test_verdict_to_check_takes_no_llm():               # F3
    assert "llm" not in inspect.signature(verdict_to_check).parameters


# ---------------------------------------------------------------------------
# F2/§5: executor uncertain → fall back to Prover verdict (claim stays pass)
# ---------------------------------------------------------------------------

_DESIGNER_BODY = (
    "EXPRESSION: G*M/r**2 | VARIABLES: G,M,r | LIMIT_VARIABLE: r "
    "| LIMIT_POINT: oo | EXPECTED: 0 | EXPECTED_SOURCE: textbook "
    "| CONFIRM_IF: limit is 0 | REFUTE_IF: limit differs"
)
_PROVER_PASS = "DERIVATION: complete | GAPS: none | FATAL_GAP: no"


def _make_uncertain_plan() -> ComputationPlan:
    return ComputationPlan(
        expression="G*M/r**2",
        variables=["G", "M", "r"],
        limit_variable="r",
        limit_point="oo",
        expected="0",
        expected_source="textbook",
    )


@pytest.mark.asyncio
async def test_executor_uncertain_prover_cannot_self_validate(monkeypatch, cfg):
    """F2/§5 + PC-1a: when run_plan returns uncertain, no executor CheckRecord is added (the fallback
    mechanic is intact) — but the Prover pass NO LONGER self-validates (PC-1a strip), so a math claim
    with only an uncertain executor + prover pass does NOT reach 'pass' (status 'pending')."""

    # Build store with one known limit so inject_limit_checks creates a limit_recovery claim
    art = IdeaArtifact(
        raw_idea="seed",
        formal_claim=FormalClaim(statement="formal", falsifiable=True),
        known_limits=[KnownLimit(limit="recovers Newtonian gravity at low speed")],
        claim_graph=[],
    )
    store = ArtifactStore(art)

    # FakeLLM: prover returns a clean pass; computation_designer returns a valid plan body
    def route(agent, messages):
        if agent == "computation_designer":
            return _DESIGNER_BODY
        return _PROVER_PASS  # prover (and any other agent)

    llm = FakeLLM(route)

    # Monkeypatch run_plan to return an uncertain verdict (SymPy couldn't decide)
    uncertain_plan = _make_uncertain_plan()

    def fake_run_plan(plan, cfg_, artifacts_dir=None):
        return ComputationVerdict(
            verdict="uncertain",
            measured="",
            plan=plan,
            result=ComputationResult(ok=False, error="undecidable"),
        )

    monkeypatch.setattr("valagents.sandbox.executor.run_plan", fake_run_plan)

    await inject_limit_checks(store, llm, cfg, tick=0)

    limit_claims = [c for c in store.current.claim_graph if c.origin == "limit_recovery"]
    assert len(limit_claims) == 1
    claim = limit_claims[0]

    # Prover check is present
    prover_checks = [ch for ch in claim.checks if ch.lens == "prover"]
    assert prover_checks, "Expected a prover CheckRecord"

    # Executor check is NOT present (uncertain was skipped per F2/§5)
    executor_checks = [ch for ch in claim.checks if ch.lens == "executor"]
    assert not executor_checks, "Uncertain executor check must not be added (F2/§5)"

    # PC-1a: the prover can no longer self-credit, so an uncertain executor leaves the math claim
    # without a code-witnessed check → it does NOT validate (status 'pending', not 'pass').
    assert claim.status == "pending", f"Expected 'pending', got '{claim.status}'"
    assert all(ch.independent_sources == 0 for ch in prover_checks)   # prover earns no independent credit

    # Audit event for the execution attempt IS recorded
    exec_events = [e for e in store.events if e.get("event") == "limit_executed"]
    assert exec_events, "limit_executed audit event should still be recorded"
    assert exec_events[0]["verdict"] == "uncertain"
