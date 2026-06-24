import inspect
from valagents.computation import (ComputationPlan, ComputationResult,
                                   ComputationVerdict, verdict_to_sim_attack)

def splan(**kw):
    base = dict(kind="simulation", primitive="ode_integrate", state_vars=["x"],
                rhs={"x": "-x"}, init={"x": "1.0"}, t_span=["0", "5"], dt="0.01",
                param_sweep={"a": ["0.3", "0.7", "3"]},
                observable={"name": "final_value", "var": "x", "window_frac": "0.2"},
                sim_criterion={"op": "le", "threshold": ["0.1"]}, robust_frac="0.8",
                max_steps=1000, max_grid_points=50, max_state_vars=4, max_expr_nodes=50)
    base.update(kw)
    return ComputationPlan(**base)

def test_simulation_plan_constructs():
    p = splan()
    assert p.kind == "simulation" and p.primitive == "ode_integrate"
    assert p.rhs == {"x": "-x"} and p.param_sweep == {"a": ["0.3", "0.7", "3"]}
    assert p.expression == ""        # symbolic fields stay defaulted

def _sv(matched):
    p = splan()
    r = ComputationResult(ok=True, computed="robust: 3/3 pass (1.00 >= 0.80)", matched=matched)
    v = ComputationVerdict(verdict=("pass" if matched == "confirm" else "fail"),
                           measured=r.computed, plan=p, result=r)
    return v

def test_confirm_is_survived_minor():
    a = verdict_to_sim_attack(_sv("confirm"), target_claim_id="c1", fatal_eligible=True)
    assert a.type == "simulation" and a.status == "survived" and a.severity == "minor"

def test_refute_eligible_is_landed_fatal():
    a = verdict_to_sim_attack(_sv("refute"), target_claim_id="c1", fatal_eligible=True)
    assert a.status == "landed" and a.severity == "fatal" and a.target_claim_id == "c1"
    assert "simulation" in a.basis and "final_value" in a.basis
    assert "['" not in a.basis  # no Python list repr leaked into the basis

def test_refute_not_eligible_is_landed_major():
    a = verdict_to_sim_attack(_sv("refute"), target_claim_id="c1", fatal_eligible=False)
    assert a.status == "landed" and a.severity == "major"

def test_verdict_to_sim_attack_takes_no_llm():
    assert "llm" not in inspect.signature(verdict_to_sim_attack).parameters

from valagents.config import SimCfg

def test_fixed_point_field_and_simcfg_knobs():
    p = ComputationPlan(kind="simulation", primitive="linear_stability", fixed_point={"x": "sqrt(a/b)"})
    assert p.fixed_point == {"x": "sqrt(a/b)"}
    c = SimCfg()
    assert c.fixed_point_tol == 1e-6 and c.min_points_per_axis == 5

def test_sim_attack_basis_linear_stability_branch():
    p = ComputationPlan(kind="simulation", primitive="linear_stability",
                        fixed_point={"x": "0"}, sim_criterion={"op": "lt", "threshold": ["0"]}, robust_frac="1")
    r = ComputationResult(ok=True, computed="linear_stability: 5/5 points satisfy criterion; alpha in [-0.5, -0.2]",
                          matched="confirm")
    v = ComputationVerdict(verdict="pass", measured=r.computed, plan=p, result=r)
    a = verdict_to_sim_attack(v, target_claim_id="m1", fatal_eligible=True)
    assert a.type == "simulation" and a.status == "survived"
    assert "linear_stability" in a.basis and "alpha" in a.basis
    assert "fixed_point" in a.basis and "?(?)" not in a.basis    # NOT the ode observable rendering
