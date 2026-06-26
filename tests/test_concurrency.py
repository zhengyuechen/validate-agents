"""Spec-4: bounded_gather — order-preserving, concurrency-capped, unbounded on non-positive limit."""
import asyncio
import pytest

from valagents.concurrency import bounded_gather


async def test_bounded_gather_preserves_order():
    async def f(x):
        await asyncio.sleep(0)
        return x * 2
    assert await bounded_gather([f(i) for i in range(5)], limit=2) == [0, 2, 4, 6, 8]


async def test_bounded_gather_respects_limit():
    cur = 0
    peak = 0
    async def f():
        nonlocal cur, peak
        cur += 1
        peak = max(peak, cur)
        await asyncio.sleep(0.01)
        cur -= 1
        return peak
    await bounded_gather([f() for _ in range(10)], limit=3)
    assert peak <= 3


async def test_bounded_gather_unbounded_when_limit_nonpositive():
    async def f(x):
        return x
    assert await bounded_gather([f(i) for i in range(4)], limit=0) == [0, 1, 2, 3]
