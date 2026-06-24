import inspect
from valagents.agents.computation_designer import design_computation
from valagents.computation import verdict_to_check, ComputationPlan, ComputationResult, ComputationVerdict
from valagents.artifact import FormalClaim, AtomicClaim
from valagents.config import Config
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
