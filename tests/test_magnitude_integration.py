import inspect
from valagents.scheduler import run_magnitude_checks
from valagents.agents.magnitude_designer import design_magnitude
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, Prediction, AttackSurface
from valagents.config import Config
from tests.fake_llm import FakeLLM

def cfg():
    return Config(default_model="fake")

def store_with_prediction(discriminates=True):
    art = IdeaArtifact(raw_idea="seed", formal_claim=FormalClaim(statement="x", falsifiable=True),
                       predictions=[Prediction(observable="shift", effect_size="1e-9",
                                               discriminates_from=("GR" if discriminates else ""), measurable=True)],
                       attack_surface=AttackSurface(attempted=["counterexample"]))
    return ArtifactStore(art)

INERT = ("COMPARISON_KIND: sensitivity_ratio | PREDICTED_EFFECT: 1e-18 | BASELINE_OR_NULL: 0 "
         "| SENSITIVITY: 1e-12 | SENSITIVITY_SOURCE: arXiv:1234 | THRESHOLD: 3 "
         "| CONFIRM_IF: ratio>=3 | REFUTE_IF: ratio<3")
DETECT = INERT.replace("PREDICTED_EFFECT: 1e-18", "PREDICTED_EFFECT: 1e-9")

def router(body):
    return FakeLLM(lambda a, m: body if a == "magnitude_designer" else "")

async def test_designer_emits_plan_only():
    plan = await design_magnitude(store_with_prediction().current.predictions[0],
                                  store_with_prediction().current, router(INERT), cfg())
    assert plan is not None and plan.kind == "magnitude" and plan.comparison_kind == "sensitivity_ratio"
    assert plan.discriminating is True            # prediction discriminates_from set
    assert "ComputationVerdict" not in inspect.getsource(design_magnitude)   # F1: no verdict

async def test_inert_lands_fatal_magnitude_attack_and_marks_attempted():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(INERT), cfg())
    mags = [a for a in s.current.attacks if a.type == "magnitude"]
    assert mags and mags[0].status == "landed" and mags[0].severity == "fatal"
    assert "magnitude" in s.current.attack_surface.attempted

async def test_detectable_survives_and_marks_attempted():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(DETECT), cfg())
    mags = [a for a in s.current.attacks if a.type == "magnitude"]
    assert mags and mags[0].status == "survived"
    assert "magnitude" in s.current.attack_surface.attempted

async def test_uncertain_adds_no_attack_and_does_not_mark_attempted(monkeypatch):
    # L2-D9: a fail-closed magnitude run must NOT mark "magnitude" attempted
    import valagents.sandbox.executor as ex
    from valagents.computation import ComputationVerdict, ComputationResult, ComputationPlan
    def fake(plan, cfg, artifacts_dir=None):
        return ComputationVerdict(verdict="uncertain", measured="", plan=plan,
                                  result=ComputationResult(ok=False, error="missing source"))
    monkeypatch.setattr(ex, "run_plan", fake)
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(INERT), cfg())
    assert not [a for a in s.current.attacks if a.type == "magnitude"]   # no attack
    assert "magnitude" not in s.current.attack_surface.attempted          # NOT marked (teeth not laundered)

async def test_evaluate_ignores_magnitude_fields():
    assert "magnitude" not in inspect.getsource(IdeaArtifact._evaluate)
    assert "comparison_kind" not in inspect.getsource(IdeaArtifact._evaluate)


BOUND_OK = ("COMPARISON_KIND: bound_check | PREDICTED_EFFECT: 1e-3 | BOUND: 1e-2 "
            "| BOUND_SOURCE: PDG2024 | CONFIRM_IF: p<=bound | REFUTE_IF: p>bound")
BOUND_VIOLATE = BOUND_OK.replace("PREDICTED_EFFECT: 1e-3", "PREDICTED_EFFECT: 1e-1")
BOUND_NO_SOURCE = BOUND_OK.replace("| BOUND_SOURCE: PDG2024 ", "| BOUND_SOURCE:  ")

async def test_bound_violation_injects_failed_claim_and_refutes():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_VIOLATE), cfg())
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert bnd and bnd[0].status == "fail" and bnd[0].load_bearing
    assert s.current.status == "refuted"

async def test_bound_compliance_injects_passing_sourced_claim():
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_OK), cfg())
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert bnd and bnd[0].status == "pass"
    assert s.current._has_independent_external_check(bnd[0])   # bound_source counts as the external check

async def test_bound_check_does_not_mark_magnitude_attempted():    # L2-D9: claim path, not an attack
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_OK), cfg())
    assert "magnitude" not in s.current.attack_surface.attempted
    assert not [a for a in s.current.attacks if a.type == "magnitude"]

async def test_bound_missing_source_injects_no_claim():            # fail-closed: no source -> no claim
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_NO_SOURCE), cfg())
    assert not [c for c in s.current.claim_graph if c.origin == "bound_check"]

async def test_bound_idempotent_across_reruns():                   # L2-D11
    s = store_with_prediction(discriminates=True)
    await run_magnitude_checks(s, router(BOUND_OK), cfg())
    await run_magnitude_checks(s, router(BOUND_OK), cfg())          # rerun = a repair iteration
    bnd = [c for c in s.current.claim_graph if c.origin == "bound_check"]
    assert len(bnd) == 1                                           # cleared and re-injected, not duplicated
