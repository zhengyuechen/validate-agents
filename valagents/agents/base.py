"""Shared agent helpers."""
from __future__ import annotations

import re


def build_messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def as_int(s: str, default: int = 0) -> int:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else default


def choice(value: str, allowed: set[str]) -> str | None:
    normalized = (value or "").strip().lower()
    return normalized if normalized in allowed else None


def map_support_to_verdict(support: str, independent_sources: int) -> str:
    support = (support or "").strip().lower()
    if support == "supported":
        return "pass" if independent_sources >= 1 else "uncertain"   # D8 downgrade
    if support == "unsupported":
        return "fail"
    return "uncertain"
