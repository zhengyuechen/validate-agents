from tests.test_artifact_gate import art, claim
from valagents.artifact import CheckRecord


def test_status_and_blocker_agree_with_evaluate():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="prover", verdict="uncertain")])])
    assert a.status == "needs_experiment"
    assert a.blocker["claim_id"] == "c1" and a.blocker["reason"] == "inconclusive"


def test_load_bearing_is_blocker_claim_when_blocked():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.load_bearing == "c1"


def test_load_bearing_is_most_depended_on_when_validated():
    # c2 depends on c1 -> c1 has more transitive dependents -> pivotal
    a = art(claim_graph=[claim("c1"), claim("c2", deps=["c1"])])
    assert a.load_bearing == "c1"
