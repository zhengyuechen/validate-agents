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

def test_mathematical_proof_overrides_noncontradictory_grounder_uncertainty():
    proof = CheckRecord(lens="prover", verdict="pass", independent_sources=1)
    no_literature = CheckRecord(lens="grounder", verdict="uncertain", basis="no direct literature support")
    claim = AtomicClaim(id="m1", statement="lemma", type="mathematical",
                        checks=[no_literature, proof])
    assert claim.status == "pass"

def test_definitional_claim_accepted_as_premise_without_independent_source():
    # PC-D6: a definitional claim is a premise/convention — a non-refuting pass (indep=0) accepts it,
    # without an independent code-witness. The exemption is type-gated: the SAME weak pass on a
    # non-definitional claim still does not pass (no say-so leak to other claim types).
    weak = CheckRecord(lens="prover", verdict="pass", independent_sources=0)
    d = AtomicClaim(id="d1", statement="define X", type="definitional", checks=[weak])
    assert d.status == "pass"
    assert AtomicClaim(id="e1", statement="s", type="empirical", checks=[weak]).status == "pending"

def test_definitional_claim_refuted_still_fails():
    # PC-D6 never overrides a refutation — a failing check dominates regardless of the premise exemption.
    weak = CheckRecord(lens="prover", verdict="pass", independent_sources=0)
    refute = CheckRecord(lens="redteam", verdict="fail")
    d = AtomicClaim(id="d1", statement="define X", type="definitional", checks=[weak, refute])
    assert d.status == "fail"

def test_definitional_claim_only_uncertain_does_not_pass():
    # the exemption requires a non-refuting PASS; an uncertain-only definitional claim stays uncertain.
    unc = CheckRecord(lens="prover", verdict="uncertain")
    d = AtomicClaim(id="d1", statement="define X", type="definitional", checks=[unc])
    assert d.status == "uncertain"

def test_mathematical_grounder_contradiction_still_blocks_pass():
    proof = CheckRecord(lens="prover", verdict="pass", independent_sources=1)
    contradiction = CheckRecord(lens="grounder", verdict="uncertain",
                                basis="CONTRADICTION: retrieved source gives a counterexample")
    claim = AtomicClaim(id="m1", statement="lemma", type="mathematical",
                        checks=[contradiction, proof])
    assert claim.status == "uncertain"
