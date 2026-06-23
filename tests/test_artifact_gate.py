from valagents.artifact import (IdeaArtifact, AtomicClaim, CheckRecord, FormalClaim,
                                Faithfulness, Coverage, AttackSurface, Attack)

PASS = CheckRecord(lens="grounder", verdict="pass", independent_sources=1)

def claim(cid, checks=(PASS,), lb=True, deps=()):
    return AtomicClaim(id=cid, statement="s", type="empirical",
                       checks=list(checks), load_bearing=lb, depends_on=list(deps), exhausted=True)

def art(**kw):
    base = dict(raw_idea="seed",
                formal_claim=FormalClaim(statement="x", falsifiable=True),
                faithfulness=Faithfulness(verdict="yes"),
                coverage=Coverage(verdict="complete"),
                attack_surface=AttackSurface(attempted=["magnitude", "confound"]),
                claim_graph=[claim("c1")], finalized=True)
    base.update(kw)
    return IdeaArtifact(**base)

# --- happy path ---
def test_internally_validated():
    assert art().status == "internally_validated"

# --- entry gates (I3) ---
def test_not_falsifiable():
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False))
    assert a.status == "refuted" and a.blocker["reason"] == "not_falsifiable"

def test_unfaithful_drift_after_retry():
    a = art(faithfulness=Faithfulness(verdict="no", retried=True))
    assert a.status == "refuted" and a.blocker["reason"] == "unfaithful_drift"

def test_unfaithful_narrowed_after_retry():
    a = art(faithfulness=Faithfulness(verdict="narrowed", retried=True))
    assert a.status == "refuted" and a.blocker["reason"] == "unfaithful_narrowed"

def test_faithfulness_none_cannot_validate():        # the SPOF-in-code test (rev 3)
    a = art(faithfulness=None)
    assert a.status != "internally_validated"

def test_empty_graph_ill_formed():
    a = art(claim_graph=[])
    assert a.status == "refuted" and a.blocker["reason"] == "ill_formed"

# --- refutation ---
def test_failed_claim():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.status == "refuted" and a.blocker["reason"] == "failed"

def test_fatal_attack_landed():
    a = art(attacks=[Attack(type="counterexample", severity="fatal", status="landed", target_claim_id="c1")])
    assert a.status == "refuted" and a.blocker["reason"] == "attacked"

# --- needs_experiment ---
def test_uncertain_claim():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="prover", verdict="uncertain")])])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "inconclusive"

def test_uncovered_pending_claim():
    a = art(claim_graph=[claim("c1", checks=[])])    # exhausted + pending
    assert a.status == "needs_experiment" and a.blocker["reason"] == "uncovered"

def test_coverage_gap():
    a = art(coverage=Coverage(verdict="gap", missing="load-bearing step"))
    assert a.status == "needs_experiment" and a.blocker["reason"] == "decomposition_gap"

def test_thin_attack_surface():
    a = art(attack_surface=AttackSurface(attempted=["counterexample"]))  # no magnitude, <2
    assert a.status == "needs_experiment" and a.blocker["reason"] == "thin_attack_surface"

def test_open_major_objection():
    a = art(attacks=[Attack(type="confound", severity="major", status="landed")])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "open_objection"

# --- D4 minor attack does not block ---
def test_minor_landed_still_validates():
    a = art(attacks=[Attack(type="confound", severity="minor", status="landed")])
    assert a.status == "internally_validated"

# --- repair-cap exhaustion (D5) ---
def test_repair_cap_exhaustion_refuted():
    a = art(repairs_spent=3, repair_cap=3,
            attacks=[Attack(type="counterexample", severity="fatal", status="landed")])
    assert a.status == "refuted"

# --- non-terminal draft (I3: scheduler keeps going) ---
def test_draft_when_unfinalized_pending():
    a = art(finalized=False, claim_graph=[claim("c1", checks=[], )])
    a.claim_graph[0].exhausted = False
    assert a.status == "draft"

# --- order independence (pre-validates Spec 4) ---
def test_order_independence():
    c = claim("c1", checks=[CheckRecord(lens="grounder", verdict="pass", independent_sources=1),
                            CheckRecord(lens="redteam", verdict="uncertain")])
    a1 = art(claim_graph=[c])
    c2 = claim("c1", checks=list(reversed(c.checks)))
    a2 = art(claim_graph=[c2])
    assert a1.status == a2.status == "needs_experiment"
