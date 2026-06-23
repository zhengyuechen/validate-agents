"""Deterministic LLM stand-in for unit tests (no network)."""
from __future__ import annotations
from typing import Callable

class FakeLLM:
    def __init__(self, router: Callable[[str, list[dict]], str]) -> None:
        self._router = router
        self.calls: list[dict] = []

    async def complete(self, agent: str, messages: list[dict],
                       temperature: float | None = None,
                       max_tokens: int | None = None) -> str:
        self.calls.append({"agent": agent, "messages": messages, "temperature": temperature})
        return self._router(agent, messages)
