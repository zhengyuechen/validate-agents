"""Magnitude-Designer: emits a structured sensitivity_ratio ComputationPlan for a measurable prediction.
It DESIGNS the check only — returns no verdict and never sees the execution result (F1/F3)."""
from __future__ import annotations
from valagents.computation import ComputationPlan
from valagents.parse import checked
from valagents.prompts import MAGNITUDE_DESIGNER
from valagents.agents.base import build_messages

_KEYS = ["COMPARISON_KIND", "PREDICTED_EFFECT", "BASELINE_OR_NULL", "SENSITIVITY",
         "SENSITIVITY_SOURCE", "THRESHOLD", "CONFIRM_IF", "REFUTE_IF"]

async def design_magnitude(prediction, art, llm, cfg) -> ComputationPlan | None:
    user = MAGNITUDE_DESIGNER.format(
        formal=art.formal_claim.statement if art.formal_claim else "",
        observable=prediction.observable, effect_size=prediction.effect_size,
        discriminates_from=prediction.discriminates_from or "(none)")
    tail = await checked("magnitude_designer", build_messages("You design detectability checks.", user),
                         _KEYS, llm=llm)
    if tail is None or tail["comparison_kind"].strip().lower() != "sensitivity_ratio":
        return None
    try:
        return ComputationPlan(
            kind="magnitude", comparison_kind="sensitivity_ratio",
            predicted_effect=tail["predicted_effect"], baseline_or_null=tail["baseline_or_null"],
            sensitivity=tail["sensitivity"], sensitivity_source=tail["sensitivity_source"],
            threshold=tail["threshold"], confirm_if=tail["confirm_if"], refute_if=tail["refute_if"],
            target_claim_id=art.load_bearing, discriminating=bool(prediction.discriminates_from),
            criterion="magnitude")
    except Exception:
        return None
