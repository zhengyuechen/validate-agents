import inspect
from tests.test_artifact_gate import art, claim
from valagents.artifact import IdeaArtifact, CheckRecord

def test_maturity_is_a_float_in_unit_interval():
    m = art().maturity
    assert isinstance(m, float) and 0.0 <= m <= 1.0

def test_status_does_not_depend_on_maturity():
    # maturity ⊥ status: status source must not reference `maturity`
    src = inspect.getsource(IdeaArtifact._evaluate)
    assert "maturity" not in src

def test_passing_claims_mature_higher_than_pending():
    high = art()  # all pass
    low = art(claim_graph=[claim("c1", checks=[])])  # pending
    assert high.maturity > low.maturity
