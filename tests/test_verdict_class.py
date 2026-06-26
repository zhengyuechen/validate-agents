"""Tests for IdeaArtifact.verdict_class (R2.1) and render_report ill_posed honesty (R2.2)."""
from valagents.artifact import (
    IdeaArtifact, AtomicClaim, CheckRecord, FormalClaim,
    Faithfulness, Coverage, AttackSurface, Attack,
)
from valagents.cli import render_report

# --- reuse helpers from test_artifact_gate ---
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


# ---- verdict_class: one test per leaf ----

def test_verdict_class_validated():
    assert art().verdict_class == "validated"

def test_verdict_class_refuted():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="redteam", verdict="fail")])])
    assert a.status == "refuted"
    assert a.verdict_class == "refuted"

def test_verdict_class_draft():
    a = IdeaArtifact(raw_idea="s")
    assert a.status == "draft"
    assert a.verdict_class == "draft"

def test_verdict_class_ill_posed_not_falsifiable():
    # FG-1: not_falsifiable is now the LAST-RESORT (falsifiable=False AND nothing landed), still ill_posed.
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[])])          # nothing landed -> demonstrably unassessable
    assert a.status == "needs_experiment" and a.blocker["reason"] == "not_falsifiable"
    assert a.verdict_class == "ill_posed"

def test_verdict_class_ill_posed_unfaithful_drift():
    a = art(faithfulness=Faithfulness(verdict="no", retried=True))
    assert a.verdict_class == "ill_posed"

def test_verdict_class_ill_posed_unfaithful_narrowed():
    a = art(faithfulness=Faithfulness(verdict="narrowed", retried=True))
    assert a.verdict_class == "ill_posed"

def test_verdict_class_ill_posed_ill_formed():
    a = art(claim_graph=[])
    assert a.verdict_class == "ill_posed"

def test_verdict_class_ill_posed_unformalizable():
    a = IdeaArtifact(raw_idea="s", finalized=True)
    assert a.status == "needs_experiment" and a.blocker["reason"] == "unformalizable"
    assert a.verdict_class == "ill_posed"

def test_verdict_class_challenged_severe_objection():
    a = art(attacks=[Attack(type="counterexample", severity="fatal", status="landed", target_claim_id="c1")])
    assert a.blocker["reason"] == "severe_objection"
    assert a.verdict_class == "challenged"

def test_verdict_class_challenged_open_objection():
    a = art(attacks=[Attack(type="confound", severity="major", status="landed")])
    assert a.blocker["reason"] == "open_objection"
    assert a.verdict_class == "challenged"

def test_verdict_class_promising_inconclusive():
    a = art(claim_graph=[claim("c1", checks=[CheckRecord(lens="prover", verdict="uncertain")])])
    assert a.blocker["reason"] == "inconclusive"
    assert a.verdict_class == "promising"

def test_verdict_class_promising_uncovered():
    a = art(claim_graph=[claim("c1", checks=[])])
    assert a.blocker["reason"] == "uncovered"
    assert a.verdict_class == "promising"

def test_verdict_class_promising_decomposition_gap():
    a = art(coverage=Coverage(verdict="gap", missing="step"))
    assert a.blocker["reason"] == "decomposition_gap"
    assert a.verdict_class == "promising"

def test_verdict_class_promising_thin_attack_surface():
    a = art(attack_surface=AttackSurface(attempted=["counterexample"]))
    assert a.blocker["reason"] == "thin_attack_surface"
    assert a.verdict_class == "promising"


# ---- render_report: ill_posed honesty (R2.2) ----

def test_render_report_ill_posed_contains_reframe_not_experiment():
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[])])          # FG-1: nothing landed -> ill_posed last-resort
    report = render_report(a)
    assert "not yet a testable claim" in report

def test_render_report_ill_posed_does_not_say_needs_experiment_or_run_a():
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[])])          # FG-1: nothing landed -> ill_posed last-resort
    report = render_report(a)
    # The status line may contain "needs_experiment" but no actionable "run a test" phrasing
    # Strip the status line itself before checking prose
    prose_lines = [ln for ln in report.splitlines() if not ln.startswith("**Status:**")]
    prose = "\n".join(prose_lines)
    assert "needs experiment" not in prose.lower()
    assert "run a test" not in prose.lower()
    assert "run a " not in prose.lower()

def test_render_report_ill_posed_verdict_class_headline():
    a = art(formal_claim=FormalClaim(statement="x", falsifiable=False),
            claim_graph=[claim("c1", checks=[])])          # FG-1: nothing landed -> ill_posed last-resort
    report = render_report(a)
    assert "**Verdict:** ill_posed" in report

def test_render_report_validated_no_reframe_note():
    a = art()
    report = render_report(a)
    assert "not yet a testable claim" not in report
    assert "**Verdict:** validated" in report
