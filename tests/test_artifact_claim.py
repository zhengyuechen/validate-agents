from valagents.artifact import AtomicClaim, CheckRecord, Source

def mk(checks):
    return AtomicClaim(id="c1", statement="s", type="empirical", checks=checks)

def test_pending_when_no_checks():
    assert mk([]).status == "pending"

def test_pass_requires_independent_source():
    weak = CheckRecord(lens="grounder", verdict="pass", independent_sources=0)
    assert mk([weak]).status == "pending"           # I2: pending, never pass
    strong = CheckRecord(lens="grounder", verdict="pass", independent_sources=1)
    assert mk([strong]).status == "pass"

def test_fail_dominates():
    a = CheckRecord(lens="grounder", verdict="pass", independent_sources=2)
    b = CheckRecord(lens="redteam", verdict="fail")
    assert mk([a, b]).status == "fail"

def test_uncertain_over_pass():
    a = CheckRecord(lens="grounder", verdict="pass", independent_sources=2)
    b = CheckRecord(lens="prover", verdict="uncertain")
    assert mk([a, b]).status == "uncertain"
