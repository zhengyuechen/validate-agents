"""Value-Grounder: READS a source and extracts a value + verbatim quote + conditions (F1 — reading only,
never judging). ground_plan orchestrates fetch -> extract -> code adjudication (ground_value)."""
from __future__ import annotations
import json
import re
from valagents.prompts import VALUE_GROUNDER
from valagents.agents.base import build_messages
from valagents.grounding import ground_value, GroundingResult
from valagents.grounding_fetch import fetch_source_text

# kind -> (value_attr, source_attr) for the single sourced value of each comparison_kind
_KIND_FIELDS = {
    "bound_check": ("bound", "bound_source"),
    "sensitivity_ratio": ("sensitivity", "sensitivity_source"),
    "discriminating_margin": ("closest_prior_effect", "closest_prior_source"),
}


class LiveFetcher:
    """The injected resolver the CLI builds when grounding is on. Wraps the real network fetch so the
    on/off is the *presence of this object*, not the backend string (tests inject a fake fetcher instead)."""
    async def fetch(self, locator: str):
        return await fetch_source_text(locator)


def _extract_json(text: str):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL) or re.findall(r"(\{.*\})", text, re.DOTALL)
    for block in reversed(blocks):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None


async def extract_value(text: str, source_quantity: str, source_unit: str, llm) -> dict | None:
    user = VALUE_GROUNDER.format(text=text[:8000], source_quantity=source_quantity, source_unit=source_unit)
    body = await llm.complete("value_grounder", build_messages("You read sources and report values verbatim.", user))
    data = _extract_json(body)
    if not isinstance(data, dict) or data.get("not_found"):
        return None
    keys = ("extracted_value", "source_unit_token", "referent", "source_conditions", "verbatim_quote")
    return {k: str(data.get(k, "")) for k in keys}


async def ground_plan(plan, resolver, llm, cfg) -> GroundingResult | None:
    """Ground the plan's single sourced value. `resolver` is a fetcher with `async fetch(locator)` (the CLI's
    LiveFetcher, or a fake in tests). **`resolver is None` → grounding OFF → return None**, regardless of the
    backend string (the on/off is the injected dependency). When ON, a fetch/extraction failure yields a
    GroundingResult('unconfirmed'), never None."""
    if resolver is None:
        return None
    fields = _KIND_FIELDS.get(plan.comparison_kind)
    if fields is None:
        return None
    value = getattr(plan, fields[0], "")
    locator = getattr(plan, fields[1], "")
    if not value or not locator:
        return GroundingResult("unconfirmed", reason="missing_value_or_locator")
    fetched = await resolver.fetch(locator)
    if not fetched:
        return GroundingResult("unconfirmed", reason="unresolvable")
    text, _meta = fetched
    extraction = await extract_value(text, plan.source_quantity, plan.source_unit, llm)
    return ground_value(value, plan.source_unit, plan.source_quantity, plan.claim_conditions,
                        extraction, text, cfg)
