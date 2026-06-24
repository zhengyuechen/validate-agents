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
