import asyncio
from collections.abc import Callable

import pytest

from async_lru import alru_cache


async def test_alru_cache_removing_lru_keys(check_lru: Callable[..., None]) -> None:
    @alru_cache(maxsize=3)
    async def coro(val: int) -> int:
        return val

    for i, v in enumerate([3, 4, 5]):
        await coro(v)
        check_lru(coro, hits=0, misses=i + 1, cache=i + 1, tasks=0, maxsize=3)

    check_lru(coro, hits=0, misses=3, cache=3, tasks=0, maxsize=3)
    assert list(coro._LRUCacheWrapper__cache) == [3, 4, 5]  # type: ignore[attr-defined]

    for v in [3, 2, 1]:
        await coro(v)
    check_lru(coro, hits=1, misses=5, cache=3, tasks=0, maxsize=3)
    assert list(coro._LRUCacheWrapper__cache) == [3, 2, 1]  # type: ignore[attr-defined]


async def test_alru_cache_removing_lru_keys_with_full_displacement(
    check_lru: Callable[..., None]
) -> None:
    @alru_cache(maxsize=3)
    async def coro(val: int) -> int:
        return val

    for i, v in enumerate([3, 4, 5]):
        await coro(v)
        check_lru(coro, hits=0, misses=i + 1, cache=i + 1, tasks=0, maxsize=3)

    check_lru(coro, hits=0, misses=3, cache=3, tasks=0, maxsize=3)
    assert list(coro._LRUCacheWrapper__cache) == [3, 4, 5]  # type: ignore[attr-defined]

    for v in [1, 2, 3]:
        await coro(v)
    check_lru(coro, hits=0, misses=6, cache=3, tasks=0, maxsize=3)
    assert list(coro._LRUCacheWrapper__cache) == [1, 2, 3]  # type: ignore[attr-defined]


async def test_alru_cache_none_max_size(check_lru: Callable[..., None]) -> None:
    @alru_cache(maxsize=None)
    async def coro(val: int) -> int:
        return val

    inputs = [1, 2, 3, 4] * 2
    coros = [coro(v) for v in inputs]

    ret = await asyncio.gather(*coros)

    check_lru(coro, hits=4, misses=4, cache=4, tasks=0, maxsize=None)
    assert ret == inputs


async def test_alru_cache_zero_max_size(check_lru: Callable[..., None]) -> None:
    @alru_cache(maxsize=0)
    async def coro(val: int) -> int:
        return val

    inputs = [1, 2, 3, 4] * 2
    coros = [coro(v) for v in inputs]

    ret = await asyncio.gather(*coros)

    check_lru(coro, hits=4, misses=4, cache=0, tasks=0, maxsize=0)
    assert ret == inputs


async def test_evicted_pending_computation_is_shared(check_lru: Callable[..., None]) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    calls: list[int] = []

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        calls.append(val)
        if val == 1:
            started.set()
            await finish.wait()
        return val

    original = asyncio.create_task(coro(1))
    await started.wait()
    assert await coro(2) == 2
    assert not coro.cache_invalidate(1)

    later = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    check_lru(coro, hits=1, misses=2, cache=1, tasks=1, maxsize=1)
    assert calls == [1, 2]

    finish.set()
    assert list(await asyncio.gather(original, later)) == [1, 1]
    assert calls == [1, 2]
    check_lru(coro, hits=1, misses=2, cache=1, tasks=0, maxsize=1)

    assert await coro(1) == 1
    assert calls == [1, 2, 1]
    check_lru(coro, hits=1, misses=3, cache=1, tasks=0, maxsize=1)


async def test_zero_size_does_not_reuse_finished_evicted_task(
    check_lru: Callable[..., None]
) -> None:
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("requires asyncio.eager_task_factory")

    calls = 0
    loop = asyncio.get_running_loop()
    task_factory = loop.get_task_factory()

    @alru_cache(maxsize=0)
    async def coro() -> int:
        nonlocal calls
        calls += 1
        return calls

    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        assert await coro() == 1
        assert await coro() == 2
    finally:
        loop.set_task_factory(task_factory)

    assert calls == 2
    check_lru(coro, hits=0, misses=2, cache=0, tasks=0, maxsize=0)


async def test_zero_size_does_not_reuse_finished_evicted_failure() -> None:
    if not hasattr(asyncio, "eager_task_factory"):
        pytest.skip("requires asyncio.eager_task_factory")

    calls = 0
    loop = asyncio.get_running_loop()
    task_factory = loop.get_task_factory()

    @alru_cache(maxsize=0)
    async def coro() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("first call")
        return calls

    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        with pytest.raises(ValueError, match="first call"):
            await coro()
        assert await coro() == 2
    finally:
        loop.set_task_factory(task_factory)

    assert calls == 2


async def test_new_caller_does_not_join_finished_evicted_task_before_callback() -> None:
    started = asyncio.Event()
    finish_first = asyncio.Event()
    start_new_caller = asyncio.Event()
    calls = 0

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        nonlocal calls
        calls += 1
        call_number = calls
        if call_number == 1:
            started.set()
            await finish_first.wait()
        return call_number

    first = asyncio.create_task(coro(1))
    await started.wait()
    assert await coro(2) == 2

    async def call_again() -> int:
        await start_new_caller.wait()
        return await coro(1)

    later = asyncio.create_task(call_again())
    await asyncio.sleep(0)
    finish_first.set()
    start_new_caller.set()

    assert await first == 1
    assert await later == 3
    assert calls == 3
