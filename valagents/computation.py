"""Spec 2 execution models. A ComputationPlan is FROZEN before execution; the verdict
is produced in code (no LLM) — see the design doc F1/F3."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

class ComputationPlan(BaseModel):
    kind: Literal["symbolic", "magnitude"] = "symbolic"
    # --- symbolic (now defaulted so magnitude plans omit them) ---
    expression: str = ""
    variables: list[str] = []
    limit_variable: str = ""
    limit_point: str = ""
    expected: str = ""
    expected_source: str = ""
    # --- magnitude ---
    comparison_kind: Literal["sensitivity_ratio", "bound_check", "discriminating_margin"] | None = None
    predicted_effect: str = ""
    baseline_or_null: str = ""
    sensitivity: str = ""
    sensitivity_source: str = ""
    bound: str = ""
    bound_source: str = ""
    closest_prior_effect: str = ""
    uncertainty: str = ""
    threshold: str = ""
    target_claim_id: str | None = None
    discriminating: bool = False
    # --- criterion / glosses (shared) ---
    criterion: Literal["symbolic_equality", "magnitude"] = "symbolic_equality"
    confirm_if: str = ""
    refute_if: str = ""

class ComputationResult(BaseModel):
    ok: bool
    computed: str = ""
    matched: Literal["confirm", "refute", "neither"] = "neither"
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    resource_use: dict = {}
    artifacts_path: str = ""

class ComputationVerdict(BaseModel):
    verdict: Literal["pass", "fail", "uncertain"]
    measured: str = ""
    plan: ComputationPlan
    result: ComputationResult


def verdict_to_check(v: "ComputationVerdict", tick: int = 0):
    """Map an executed ComputationVerdict to a CheckRecord(lens='executor'). No LLM (F3).
    Kind-aware: a bound_check (magnitude) surfaces bound/bound_source; a symbolic limit
    surfaces expected/expected_source."""
    from valagents.artifact import CheckRecord, Source
    indep = 1 if v.verdict == "pass" else 0
    if v.plan.kind == "magnitude" and v.plan.comparison_kind == "bound_check":
        basis = (f"computed {v.measured or '?'}; bound = {v.plan.bound} "
                 f"(source: {v.plan.bound_source or 'n/a'}); matched = {v.result.matched}")
        src = v.plan.bound_source
    else:
        basis = (f"computed limit = {v.measured or '?'}; expected = {v.plan.expected} "
                 f"(source: {v.plan.expected_source or 'n/a'}); matched = {v.result.matched}")
        src = v.plan.expected_source
    sources = ([Source(locator=src, relation="independent")] if src else [])
    return CheckRecord(lens="executor", verdict=v.verdict, basis=basis,
                       independent_sources=indep, sources=sources, tick=tick)


def verdict_to_attack(v: "ComputationVerdict", target_claim_id, discriminating: bool, tick: int = 0):
    """Map an executed magnitude ComputationVerdict to a Red-team Attack. No LLM (F3).
    Call ONLY on a decisive verdict (matched in {'confirm','refute'})."""
    from valagents.artifact import Attack
    if v.result.matched == "confirm":
        status, severity = "survived", "minor"
    else:  # "refute" — inert / non-discriminating
        status, severity = "landed", ("fatal" if discriminating else "major")
    basis = (f"{v.plan.comparison_kind}: computed = {v.measured or '?'}; "
             f"sensitivity = {v.plan.sensitivity or 'n/a'} "
             f"(source: {v.plan.sensitivity_source or 'n/a'}); threshold = {v.plan.threshold or 'n/a'}")
    return Attack(type="magnitude", severity=severity, status=status,
                  target_claim_id=target_claim_id, basis=basis)
