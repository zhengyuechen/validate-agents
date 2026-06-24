"""Magnitude-Designer: emits ONE structured magnitude ComputationPlan for a measurable prediction,
routed to the comparison kind the LLM chose. It DESIGNS the check only — returns no verdict and never
sees the execution result (F1/F3)."""
from __future__ import annotations
from valagents.computation import ComputationPlan
from valagents.parse import checked_body, parse_tail, StrictTailError
from valagents.prompts import MAGNITUDE_DESIGNER
from valagents.agents.base import build_messages

# All kinds carry COMPARISON_KIND + PREDICTED_EFFECT + CONFIRM_IF + REFUTE_IF; the rest are kind-specific.
_COMMON = ["COMPARISON_KIND", "PREDICTED_EFFECT", "CONFIRM_IF", "REFUTE_IF"]
_KIND_KEYS = {
    "sensitivity_ratio": ["BASELINE_OR_NULL", "SENSITIVITY", "SENSITIVITY_SOURCE", "THRESHOLD"],
    "bound_check": ["BOUND", "BOUND_SOURCE"],
}

async def design_magnitude(prediction, art, llm, cfg) -> ComputationPlan | None:
    user = MAGNITUDE_DESIGNER.format(
        formal=art.formal_claim.statement if art.formal_claim else "",
        observable=prediction.observable, effect_size=prediction.effect_size,
        discriminates_from=prediction.discriminates_from or "(none)")
    head, body = await checked_body(
        "magnitude_designer", build_messages("You design detectability checks.", user),
        _COMMON, llm=llm)
    if head is None:
        return None
    ck = head["comparison_kind"].strip().lower()
    extra = _KIND_KEYS.get(ck)
    if extra is None:
        return None
    try:
        t = parse_tail(body, _COMMON + extra)        # same body, full key set for this kind
    except StrictTailError:
        return None                                  # missing a kind-specific quantity/source -> no plan (fallback)
    common = dict(kind="magnitude", confirm_if=t["confirm_if"], refute_if=t["refute_if"],
                  target_claim_id=art.load_bearing, discriminating=bool(prediction.discriminates_from),
                  criterion="magnitude")
    try:
        if ck == "sensitivity_ratio":
            return ComputationPlan(comparison_kind="sensitivity_ratio",
                predicted_effect=t["predicted_effect"], baseline_or_null=t["baseline_or_null"],
                sensitivity=t["sensitivity"], sensitivity_source=t["sensitivity_source"],
                threshold=t["threshold"], **common)
        if ck == "bound_check":
            bs = t["bound_source"].strip()
            if not bs or "|" in bs:                  # fail-closed: empty or spilled value -> no plan
                return None
            return ComputationPlan(comparison_kind="bound_check",
                predicted_effect=t["predicted_effect"], bound=t["bound"], bound_source=bs,
                **common)
    except Exception:
        return None
    return None
