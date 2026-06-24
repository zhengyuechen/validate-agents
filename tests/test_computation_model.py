from valagents.computation import ComputationPlan, ComputationResult, ComputationVerdict

def test_plan_minimal():
    p = ComputationPlan(expression="G*M/r**2*(1+a/c**2)", variables=["G","M","r","a","c"],
                        limit_variable="c", limit_point="oo", expected="G*M/r**2")
    assert p.kind == "symbolic" and p.criterion == "symbolic_equality"

def test_verdict_wraps_plan_and_result():
    p = ComputationPlan(expression="x", variables=["x"], limit_variable="x", limit_point="0", expected="0")
    r = ComputationResult(ok=True, computed="0", matched="confirm")
    v = ComputationVerdict(verdict="pass", measured="0", plan=p, result=r)
    assert v.verdict == "pass" and v.plan.expected == "0"
