"""Spec 2 execution models. A ComputationPlan is FROZEN before execution; the verdict
is produced in code (no LLM) — see the design doc F1/F3."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel

class ComputationPlan(BaseModel):
    kind: Literal["symbolic"] = "symbolic"
    expression: str
    variables: list[str] = []
    limit_variable: str
    limit_point: str
    expected: str
    expected_source: str = ""
    criterion: Literal["symbolic_equality"] = "symbolic_equality"
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
    """Map an executed ComputationVerdict to a CheckRecord(lens='executor'). No LLM (F3)."""
    from valagents.artifact import CheckRecord, Source
    indep = 1 if v.verdict == "pass" else 0
    basis = (f"computed limit = {v.measured or '?'}; expected = {v.plan.expected} "
             f"(source: {v.plan.expected_source or 'n/a'}); matched = {v.result.matched}")
    sources = ([Source(locator=v.plan.expected_source, relation="independent")]
               if v.plan.expected_source else [])
    return CheckRecord(lens="executor", verdict=v.verdict, basis=basis,
                       independent_sources=indep, sources=sources, tick=tick)
