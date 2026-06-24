import inspect
from valagents.artifact import AtomicClaim, CheckRecord, IdeaArtifact

def _claim(checks, ctype="mathematical"):
    return AtomicClaim(id="c1", statement="s", type=ctype, checks=checks)

def test_executor_pass_makes_math_claim_pass_over_grounder_uncertainty():
    execpass = CheckRecord(lens="executor", verdict="pass", independent_sources=1)
    gunc = CheckRecord(lens="grounder", verdict="uncertain")
    assert _claim([execpass, gunc]).status == "pass"   # computed equality is a strongest-pass

def test_executor_fail_makes_claim_fail():
    assert _claim([CheckRecord(lens="executor", verdict="fail")]).status == "fail"

def test_executor_pass_needs_independent_source():
    weak = CheckRecord(lens="executor", verdict="pass", independent_sources=0)
    assert _claim([weak]).status == "pending"          # pass requires independent_sources>=1

def test_evaluate_does_not_reference_executor():
    assert "executor" not in inspect.getsource(IdeaArtifact._evaluate)

def test_executor_pass_overrides_prover_uncertain():
    p = CheckRecord(lens="prover", verdict="uncertain", basis="gapped derivation")
    e = CheckRecord(lens="executor", verdict="pass", independent_sources=1)
    assert _claim([p, e]).status == "pass"     # executed proof dominates reasoned doubt

def test_prover_uncertain_alone_stays_uncertain():
    p = CheckRecord(lens="prover", verdict="uncertain", basis="gapped")
    assert _claim([p]).status == "uncertain"   # no proof pass -> still uncertain

def test_contradiction_uncertain_still_blocks_even_with_executor_pass():
    p = CheckRecord(lens="prover", verdict="uncertain", basis="CONTRADICTION: violates bound")
    e = CheckRecord(lens="executor", verdict="pass", independent_sources=1)
    assert _claim([p, e]).status == "uncertain"  # a contradiction blocks regardless
