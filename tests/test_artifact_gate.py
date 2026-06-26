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

def test_definitional_premise_does_not_block_strict_validation():
    # PC-D6 (both gate sites): a load-bearing definitional PREMISE (non-refuting prover pass, indep=0)
    # is accepted without an independent check and does NOT block validation, while the substantive core
    # claim still carries a REAL code-witnessed independent check. Fails without the _has_independent_
    # external_check exemption (the strict gate's second site) — guards against the half-wired PC-D6.
    premise = AtomicClaim(id="d1", statement="define X", type="definitional",
                          checks=[CheckRecord(lens="prover", verdict="pass", independent_sources=0)],
                          load_bearing=True, exhausted=True)
    assert art(claim_graph=[premise, claim("c1")]).status == "internally_validated"

def test_refuted_definitional_premise_still_blocks():
    # the exemption never rescues a refuted premise — a failing check drives REFUTED.
    bad = AtomicClaim(id="d1", statement="define X", type="definitional",
                      checks=[CheckRecord(lens="redteam", verdict="fail")],
                      load_bearing=True, exhausted=True)
    assert art(claim_graph=[bad, claim("c1")]).status == "refuted"

def test_purely_definitional_roots_do_not_strict_validate():
    # PC-D6 guard: the definitional exemption keys on the decomposer's (LLM) `type` label, so an
    # all-definitional root set would satisfy the witness requirement on ZERO real checks. The guard
    # requires >=1 NON-definitional root with a real code-witnessed check, so a premises-only artifact
    # cannot reach validated even though each premise is individually accepted (status=pass via PC-D6).
    d = lambda cid: AtomicClaim(id=cid, statement="define", type="definitional",
                                checks=[CheckRecord(lens="prover", verdict="pass", independent_sources=0)],
                                load_bearing=True, exhausted=True)
    assert art(claim_graph=[d("d1"), d("d2")]).status == "draft"   # accepted-but-unwitnessed -> not validated

# --- FG-1: not_falsifiable is a code-witnessed LAST-RESORT, no longer an entry-gate on the LLM flag ---
def test_not_falsifiable_when_nothing_landed():
    # fires only when falsifiable=False AND no root received any landed check (demonstrably unassessable):
    # a root with no checks (pending, not validatable) → not_falsifiable, outranking "uncovered".
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[])])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "not_falsifiable"

def test_falsifiable_false_with_witness_validates():
    # FG-1: a falsifiable=False artifact whose root carries a real code-witnessed check is NO LONGER
    # blocked at entry — it flows to its real verdict (the flag is surfaced, not a gate).
    assert art(formal_claim=FormalClaim(statement="x", falsifiable=False)).status == "internally_validated"

def test_falsifiable_false_refuted_is_refuted():
    # a landed contradiction on a falsifiable=False root → REFUTED (refutation precedes the last-resort).
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.status == "refuted"

def test_unfaithful_drift_after_retry():
    a = art(faithfulness=Faithfulness(verdict="no", retried=True))
    assert a.status == "needs_experiment" and a.blocker["reason"] == "unfaithful_drift"

def test_unfaithful_narrowed_after_retry():
    a = art(faithfulness=Faithfulness(verdict="narrowed", retried=True))
    assert a.status == "needs_experiment" and a.blocker["reason"] == "unfaithful_narrowed"

def test_faithfulness_none_cannot_validate():        # the SPOF-in-code test (rev 3)
    a = art(faithfulness=None)
    assert a.status != "internally_validated"

def test_empty_graph_ill_formed():
    a = art(claim_graph=[])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "ill_formed"

# --- refutation ---
def test_failed_claim():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.status == "refuted" and a.blocker["reason"] == "failed"

def test_fatal_attack_landed():
    a = art(attacks=[Attack(type="counterexample", severity="fatal", status="landed", target_claim_id="c1")])
    assert a.status == "needs_experiment" and a.blocker["reason"] == "severe_objection"

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

# --- repair-cap exhaustion leaves an unresolved objection ---
def test_repair_cap_exhaustion_needs_experiment():
    a = art(repairs_spent=3, repair_cap=3,
            attacks=[Attack(type="counterexample", severity="fatal", status="landed")])
    assert a.status == "needs_experiment"

# --- non-terminal draft (I3: scheduler keeps going) ---
def test_draft_when_unfinalized_pending():
    a = art(finalized=False, claim_graph=[claim("c1", checks=[], )])
    a.claim_graph[0].exhausted = False
    assert a.status == "draft"

# --- unformalizable (I3 totality hole fix) ---
def test_unformalizable_when_finalized_without_formal_claim():
    a = IdeaArtifact(raw_idea="s", finalized=True)
    assert a.status == "needs_experiment" and a.blocker["reason"] == "unformalizable"

def test_draft_when_not_finalized_without_formal_claim():
    a = IdeaArtifact(raw_idea="s")
    assert a.status == "draft"

# --- order independence (pre-validates Spec 4) ---
def test_order_independence():
    c = claim("c1", checks=[CheckRecord(lens="grounder", verdict="pass", independent_sources=1),
                            CheckRecord(lens="redteam", verdict="uncertain")])
    a1 = art(claim_graph=[c])
    c2 = claim("c1", checks=list(reversed(c.checks)))
    a2 = art(claim_graph=[c2])
    assert a1.status == a2.status == "needs_experiment"
