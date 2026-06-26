"""Spec-4: bounded concurrency for the scheduler's independent fan-out.
Order-preserving; a single semaphore caps in-flight LLM calls so parallel lenses
don't draw provider 429s (valagents/llm.py has no rate limiting of its own)."""
from __future__ import annotations
import asyncio


async def bounded_gather(coros: list, limit: int) -> list:
    """Run ``coros`` with at most ``limit`` in flight, returning results in input order.
    ``limit <= 0`` (or falsy) → unbounded ``asyncio.gather``."""
    if not limit or limit <= 0:
        return await asyncio.gather(*coros)
    sem = asyncio.Semaphore(limit)

    async def _run(c):
        async with sem:
            return await c

    return await asyncio.gather(*[_run(c) for c in coros])
