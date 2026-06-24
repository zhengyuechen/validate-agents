"""Simulation-Designer: emits a structured SimulationPlan (kind='simulation') for a mechanistic claim.
It DESIGNS the toy model only — returns no verdict and never sees the execution result (F1/F3)."""
from __future__ import annotations
import json
import re
from valagents.computation import ComputationPlan
from valagents.prompts import SIMULATION_DESIGNER
from valagents.agents.base import build_messages

_FIELDS = ("primitive", "state_vars", "rhs", "params", "init", "param_sweep", "init_sweep",
           "t_span", "dt", "observable", "sim_criterion", "robust_frac",
           "max_steps", "max_grid_points", "max_state_vars", "max_expr_nodes")

def _extract_json(text: str):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"(\{.*\})", text, re.DOTALL)   # fall back to the last bare object
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None

async def design_simulation(claim, art, llm, cfg) -> ComputationPlan | None:
    user = SIMULATION_DESIGNER.format(
        formal=art.formal_claim.statement if art.formal_claim else "",
        statement=claim.statement)
    body = await llm.complete("simulation_designer", build_messages("You design toy-model simulations.", user))
    data = _extract_json(body)
    if not isinstance(data, dict):
        return None
    fields = {k: data[k] for k in _FIELDS if k in data}     # accept only known keys (ignore extras)
    try:
        return ComputationPlan(kind="simulation", target_claim_id=claim.id, **fields)
    except Exception:
        return None
