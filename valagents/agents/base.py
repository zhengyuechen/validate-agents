"""Shared agent helpers."""
from __future__ import annotations


def build_messages(system: str, user: str) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def as_int(s: str, default: int = 0) -> int:
    try:
        return int("".join(ch for ch in s if ch.isdigit() or ch == "-") or default)
    except ValueError:
        return default


def map_support_to_verdict(support: str, independent_sources: int) -> str:
    support = (support or "").strip().lower()
    if support == "supported":
        return "pass" if independent_sources >= 1 else "uncertain"   # D8 downgrade
    if support == "unsupported":
        return "fail"
    return "uncertain"
