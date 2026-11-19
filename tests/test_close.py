import asyncio
from collections.abc import Callable

import pytest

from async_lru import alru_cache


async def test_cache_close(check_lru: Callable[..., None]) -> None:
    @alru_cache()
    async def coro(val: int) -> int:
        await asyncio.sleep(0.2)

        return val

    assert not coro.cache_parameters()["closed"]

    inputs = [1, 2, 3, 4, 5]

    coros = [coro(v) for v in inputs]

    gather = asyncio.gather(*coros)

    await asyncio.sleep(0.1)

    check_lru(coro, hits=0, misses=5, cache=5, tasks=5)

    close = coro.cache_close()

    check_lru(coro, hits=0, misses=5, cache=5, tasks=5)

    await close

    check_lru(coro, hits=0, misses=5, cache=0, tasks=0)
    assert coro.cache_parameters()["closed"]

    with pytest.raises(asyncio.CancelledError):
        await gather

    check_lru(coro, hits=0, misses=5, cache=0, tasks=0)
    assert coro.cache_parameters()["closed"]

    # double call is no-op
    await coro.cache_close()


async def test_cache_close_wait_bound_method(check_lru: Callable[..., None]) -> None:
    class Foo:
        @alru_cache()
        async def coro(self, val: int) -> int:
            await asyncio.sleep(0.02)
            return val

    foo = Foo()
    inputs = [1, 2, 3]

    coros = [foo.coro(v) for v in inputs]
    gather = asyncio.gather(*coros)

    # Yield to loop to start tasks
    await asyncio.sleep(0)

    # wait=True should allow tasks to finish (no cancellation)
    await foo.coro.cache_close(wait=True)

    results = await gather
    assert results == inputs

    check_lru(foo.coro, hits=0, misses=3, cache=3, tasks=0)
    assert foo.coro.cache_parameters()["closed"]


async def test_cache_close_waits_for_evicted_computation(
    check_lru: Callable[..., None]
) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        if val == 1:
            started.set()
            await finish.wait()
        return val

    result = asyncio.create_task(coro(1))
    await started.wait()
    assert await coro(2) == 2
    check_lru(coro, hits=0, misses=2, cache=1, tasks=1, maxsize=1)

    close = asyncio.create_task(coro.cache_close(wait=True))
    await asyncio.sleep(0)
    assert not close.done()
    finish.set()
    await close
    assert await result == 1
    check_lru(coro, hits=0, misses=2, cache=1, tasks=0, maxsize=1)


async def test_cache_close_cancels_evicted_computation(
    check_lru: Callable[..., None]
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()
    finish_cancellation = asyncio.Event()

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        if val == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await finish_cancellation.wait()
                raise
        return val

    result = asyncio.create_task(coro(1))
    await started.wait()
    assert await coro(2) == 2
    check_lru(coro, hits=0, misses=2, cache=1, tasks=1, maxsize=1)

    close = asyncio.create_task(coro.cache_close(wait=False))
    await cancelled.wait()
    assert not close.done()
    finish_cancellation.set()
    await close
    assert cancelled.is_set()
    with pytest.raises(asyncio.CancelledError):
        await result
    check_lru(coro, hits=0, misses=2, cache=1, tasks=0, maxsize=1)


async def test_cache_close_does_not_cancel_invalidated_work() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    cancelled = asyncio.Event()

    @alru_cache()
    async def coro() -> str:
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "done"

    result = asyncio.create_task(coro())
    await started.wait()
    assert coro.cache_invalidate()
    assert coro.cache_parameters()["tasks"] == 0

    await asyncio.wait_for(coro.cache_close(wait=False), timeout=0.1)
    assert not cancelled.is_set()
    finish.set()
    assert await result == "done"


async def test_cache_close_does_not_wait_for_cleared_evicted_work() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    cancelled = asyncio.Event()

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        if val == 1:
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return val

    result = asyncio.create_task(coro(1))
    await started.wait()
    assert await coro(2) == 2
    coro.cache_clear()
    assert coro.cache_parameters()["tasks"] == 0

    await asyncio.wait_for(coro.cache_close(wait=True), timeout=0.1)
    assert not cancelled.is_set()
    finish.set()
    assert await result == 1
