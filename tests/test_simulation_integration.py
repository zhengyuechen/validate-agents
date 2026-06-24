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

DISC_PLAN = {**PLAN, "null_overrides": {"a": "0"}}
DISC_BODY = "```json\n" + json.dumps(DISC_PLAN) + "\n```"
NOTNEC_PLAN = {**PLAN, "null_overrides": {"a": "0"}, "sim_criterion": {"op": "le", "threshold": ["2.0"]}}
NOTNEC_BODY = "```json\n" + json.dumps(NOTNEC_PLAN) + "\n```"

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

async def test_designer_emits_null_overrides():
    s = _store()
    p = await design_simulation(s.current.claim_graph[0], s.current, router(DISC_BODY), cfg())
    assert p is not None and p.null_overrides == {"a": "0"}

async def test_discriminating_pass_is_discounted_survived():
    s = _store()
    await run_simulation_checks(s, router(DISC_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived"            # discounted positive
    assert s.current.claim_graph[0].checks == []            # no CheckRecord injected
    assert "simulation" in s.current.attack_surface.attempted
    assert "discriminating" in sims[0].basis   # proves the TWO-ARM discrimination path ran (not single-arm "robust:")

async def test_behavior_without_mechanism_challenges():
    s = _store(role="novel_core", load_bearing=True)
    await run_simulation_checks(s, router(NOTNEC_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "landed" and sims[0].severity == "fatal"
    assert s.current.verdict_class == "challenged"

NUMERIC_PLAN = {
    "primitive": "ode_integrate", "state_vars": ["x"], "rhs": {"x": "-a*x"},
    "params": {"a": 1.0}, "init": {"x": 1.0}, "t_span": [0, 5], "dt": 0.01,
    "param_sweep": {"a": [0.8, 1.2, 5]},
    "observable": {"name": "final_value", "var": "x", "window_frac": 0.1},
    "sim_criterion": {"op": "le", "threshold": [0.2]}, "robust_frac": 0.8,
    "max_steps": 2000, "max_grid_points": 50, "max_state_vars": 4, "max_expr_nodes": 50,
    "null_overrides": {"a": 0},
}
NUMERIC_BODY = "```json\n" + json.dumps(NUMERIC_PLAN) + "\n```"

async def test_designer_coerces_numeric_json_values():
    s = _store()
    p = await design_simulation(s.current.claim_graph[0], s.current, router(NUMERIC_BODY), cfg())
    assert p is not None                                   # was None before the fix (ValidationError swallowed)
    assert p.dt == "0.01"                                  # float scalar -> str
    assert p.params == {"a": "1.0"} and p.init == {"x": "1.0"}   # dict[str,str] values coerced
    assert p.null_overrides == {"a": "0"}                  # null_overrides numeric value coerced
    assert p.param_sweep == {"a": ["0.8", "1.2", "5"]}     # nested sweep values coerced
    assert p.t_span == ["0", "5"]                          # list[str] elements coerced
    assert p.sim_criterion == {"op": "le", "threshold": ["0.2"]}   # nested threshold coerced
    assert p.robust_frac == "0.8"
    assert p.max_steps == 2000 and p.max_grid_points == 50 and p.max_state_vars == 4 and p.max_expr_nodes == 50  # int caps stay ints (NOT stringified)

async def test_numeric_json_plan_runs_end_to_end():
    # the coerced plan must actually execute (discriminating, since null_overrides a=0) -> a real attack, not a silent skip
    s = _store()
    await run_simulation_checks(s, router(NUMERIC_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived" and "discriminating" in sims[0].basis  # discriminating two-arm path ran

async def test_numeric_state_vars_rejected():
    # numeric state-var names must NOT be coerced into valid names -> Pydantic list[str] rejects -> None
    bad = {**NUMERIC_PLAN, "state_vars": [0, 1], "rhs": {"0": "-a*0", "1": "a*1"}}
    body = "```json\n" + json.dumps(bad) + "\n```"
    s = _store()
    assert await design_simulation(s.current.claim_graph[0], s.current, router(body), cfg()) is None

async def test_bool_value_rejected():
    # a bool in a dict[str,str] field stays a bool -> Pydantic rejects -> None (not coerced to "True")
    bad = {**NUMERIC_PLAN, "params": {"a": True}}
    body = "```json\n" + json.dumps(bad) + "\n```"
    s = _store()
    assert await design_simulation(s.current.claim_graph[0], s.current, router(body), cfg()) is None

LS_PLAN = {
    "primitive": "linear_stability", "state_vars": ["x"], "rhs": {"x": "-a*x"},
    "params": {"a": "1.0"}, "fixed_point": {"x": "0"},
    "param_sweep": {"a": ["0.5", "2.0", "6"]},
    "sim_criterion": {"op": "lt", "threshold": ["0"]}, "robust_frac": "1",
    "max_grid_points": 50, "max_state_vars": 4, "max_expr_nodes": 50,
}
LS_BODY = "```json\n" + json.dumps(LS_PLAN) + "\n```"
LS_UNSTABLE = {**LS_PLAN, "rhs": {"x": "a*x"}}   # alpha=+a>0, criterion lt 0 -> NOT stable -> refute -> challenged
LS_UNSTABLE_BODY = "```json\n" + json.dumps(LS_UNSTABLE) + "\n```"

async def test_ls_designer_emits_fixed_point():
    s = _store()
    p = await design_simulation(s.current.claim_graph[0], s.current, router(LS_BODY), cfg())
    assert p is not None and p.primitive == "linear_stability" and p.fixed_point == {"x": "0"}

async def test_ls_stable_is_discounted_survived():
    s = _store()
    await run_simulation_checks(s, router(LS_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "survived"
    assert s.current.claim_graph[0].checks == []                  # discounted: no CheckRecord
    assert "linear_stability" in sims[0].basis                    # the basis branch (not observable=?(?))

async def test_ls_unstable_challenges():
    s = _store(role="novel_core", load_bearing=True)
    await run_simulation_checks(s, router(LS_UNSTABLE_BODY), cfg())
    sims = [a for a in s.current.attacks if a.type == "simulation"]
    assert sims and sims[0].status == "landed" and sims[0].severity == "fatal"
    assert s.current.verdict_class == "challenged"
