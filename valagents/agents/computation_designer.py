"""Computation-Designer: emits a structured ComputationPlan for a known-limit-recovery claim.
It DESIGNS the check only — it returns no plan execution and never sees the execution result (F1/F3)."""
from __future__ import annotations
from valagents.computation import ComputationPlan
from valagents.parse import checked
from valagents.prompts import COMPUTATION_DESIGNER
from valagents.agents.base import build_messages

_KEYS = ["EXPRESSION", "VARIABLES", "LIMIT_VARIABLE", "LIMIT_POINT",
         "EXPECTED", "EXPECTED_SOURCE", "CONFIRM_IF", "REFUTE_IF"]

async def design_computation(claim, formal_claim, llm, cfg) -> ComputationPlan | None:
    user = COMPUTATION_DESIGNER.format(
        formal=formal_claim.statement if formal_claim else "", statement=claim.statement)
    tail = await checked("computation_designer", build_messages("You design symbolic checks.", user),
                         _KEYS, llm=llm)
    if tail is None:
        return None
    variables = [v.strip() for v in tail["variables"].split(",") if v.strip()]
    try:
        return ComputationPlan(
            expression=tail["expression"], variables=variables,
            limit_variable=tail["limit_variable"], limit_point=tail["limit_point"],
            expected=tail["expected"], expected_source=tail["expected_source"],
            confirm_if=tail["confirm_if"], refute_if=tail["refute_if"])
    except Exception:
        return None
