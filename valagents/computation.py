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
