import inspect
import json
from valagents.scheduler import run_simulation_checks
from valagents.agents.simulation_designer import design_simulation
from valagents.store import ArtifactStore
from valagents.artifact import IdeaArtifact, FormalClaim, AtomicClaim, AttackSurface
from valagents.config import Config
from tests.fake_llm import FakeLLM

def cfg():
    return Config(default_model="fake")

PLAN = {
    "primitive": "ode_integrate", "state_vars": ["x"], "rhs": {"x": "-a*x"},
    "params": {"a": "1.0"}, "init": {"x": "1.0"}, "t_span": ["0", "5"], "dt": "0.01",
    "param_sweep": {"a": ["0.8", "1.2", "5"]},
    "observable": {"name": "final_value", "var": "x", "window_frac": "0.1"},
    "sim_criterion": {"op": "le", "threshold": ["0.2"]}, "robust_frac": "0.8",
    "max_steps": 2000, "max_grid_points": 50, "max_state_vars": 4, "max_expr_nodes": 50,
}
PASS_BODY = "Here is the plan.\n```json\n" + json.dumps(PLAN) + "\n```"
FAIL_PLAN = {**PLAN, "sim_criterion": {"op": "le", "threshold": ["1e-9"]}}   # never met -> robust fail
FAIL_BODY = "```json\n" + json.dumps(FAIL_PLAN) + "\n```"

def _store(role="novel_core", load_bearing=True, mechanistic=True):
    claim = AtomicClaim(id="m1", statement="mechanism M produces oscillation",
                        type=("mechanistic" if mechanistic else "empirical"),
                        role=role, load_bearing=load_bearing)
    art = IdeaArtifact(raw_idea="seed", formal_claim=FormalClaim(statement="x", falsifiable=True),
                       claim_graph=[claim], attack_surface=AttackSurface(attempted=["counterexample"]))
    return ArtifactStore(art)

def router(body):
    return FakeLLM(lambda a, m: body if a == "simulation_designer" else "")

async def test_designer_emits_plan_only():
    s = _store()
    p = await design_simulation(s.current.claim_graph[0], s.current, router(PASS_BODY), cfg())
    assert p is not None and p.kind == "simulation" and p.primitive == "ode_integrate"
    assert p.target_claim_id == "m1"
    assert "ComputationVerdict" not in inspect.getsource(design_simulation)   # F1: no verdict

async def test_designer_malformed_json_returns_none():
    s = _store()
    assert await design_simulation(s.current.claim_graph[0], s.current, router("no json here at all"), cfg()) is None
    bad = router("```json\n{not valid json,,}\n```")
    assert await design_simulation(s.current.claim_graph[0], s.current, bad, cfg()) is None

async def test_robust_fail_lands_fatal_and_challenges():
    s = _store(role="novel_core", load_bearing=True)
    await run_simulation_checks(s, router(FAIL_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "landed" and sims[0].severity == "fatal"
    assert "simulation" in s.current.attack_surface.attempted
    assert s.current.verdict_class == "challenged"

async def test_fail_non_novelcore_is_major():
    s = _store(role="background", load_bearing=True)
    await run_simulation_checks(s, router(FAIL_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].severity == "major"

async def test_robust_pass_is_discounted_survived():
    s = _store()
    await run_simulation_checks(s, router(PASS_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived"
    # discounted: PASS creates NO CheckRecord and sets NO independent_sources on the claim
    assert s.current.claim_graph[0].checks == []
    assert "simulation" in s.current.attack_surface.attempted

async def test_uncertain_no_attack_not_marked(monkeypatch):
    import valagents.sandbox.executor as ex
    from valagents.computation import ComputationVerdict, ComputationResult
    def fake(plan, cfg, artifacts_dir=None):
        return ComputationVerdict(verdict="uncertain", measured="", plan=plan,
                                  result=ComputationResult(ok=False, error="x"))
    monkeypatch.setattr(ex, "run_plan", fake)
    s = _store()
    await run_simulation_checks(s, router(PASS_BODY), cfg())
    assert not [a for a in s.current.attacks if a.type == "simulation"]
    assert "simulation" not in s.current.attack_surface.attempted

async def test_no_mechanistic_claim_is_noop():
    s = _store(mechanistic=False)
    await run_simulation_checks(s, router(PASS_BODY), cfg())
    assert not [a for a in s.current.attacks if a.type == "simulation"]
    assert "simulation" not in s.current.attack_surface.attempted

def test_simulation_does_not_satisfy_magnitude_teeth():
    art = IdeaArtifact(raw_idea="x", attack_surface=AttackSurface(attempted=["simulation", "counterexample"]))
    assert art._thin_attack_surface() is True        # "simulation" != mandatory "magnitude"

def test_evaluate_ignores_simulation_fields():
    assert "simulation" not in inspect.getsource(IdeaArtifact._evaluate)
    assert "primitive" not in inspect.getsource(IdeaArtifact._evaluate)
