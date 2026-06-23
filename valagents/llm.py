"""LLM access. Only module that talks to OpenRouter. Swappable with FakeLLM in tests."""
from __future__ import annotations
import json
import re
from typing import Protocol
from tenacity import retry, wait_exponential, stop_after_attempt
from valagents.config import Config, require_openrouter_key

def extract_json(text: str) -> object:
    """Parse JSON from an LLM response, tolerating ```json / ``` code fences."""
    s = text.strip()
    # strip a leading code fence with optional language tag (```, ```json, ```JSON ...)
    s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
    # strip a trailing code fence
    s = re.sub(r"\n?```$", "", s)
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse JSON: {e}") from e

class LLMClient(Protocol):
    async def complete(self, agent: str, messages: list[dict],
                       temperature: float | None = None,
                       max_tokens: int | None = None) -> str: ...

class OpenRouterClient:
    def __init__(self, config: Config) -> None:
        from openai import AsyncOpenAI
        self._cfg = config
        self._client = AsyncOpenAI(
            api_key=require_openrouter_key(),
            base_url="https://openrouter.ai/api/v1",
        )

    @retry(wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4), reraise=True)
    async def complete(self, agent: str, messages: list[dict],
                       temperature: float | None = None,
                       max_tokens: int | None = None) -> str:
        if temperature is None:
            temperature = self._cfg.temperature.get(agent)
        kwargs: dict = {"model": self._cfg.model_for(agent), "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
