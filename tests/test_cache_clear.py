import asyncio
from collections.abc import Callable

from async_lru import alru_cache


async def test_cache_clear(check_lru: Callable[..., None]) -> None:
    @alru_cache()
    async def coro(val: int) -> int:
        return val

    inputs = [1, 2, 3]
    coros = [coro(v) for v in inputs]
    ret = await asyncio.gather(*coros)
    assert ret == inputs
    check_lru(coro, hits=0, misses=3, cache=3, tasks=0)

    coro.cache_clear()

    check_lru(coro, hits=0, misses=0, cache=0, tasks=0)


async def test_cache_clear_pending_task() -> None:
    calls = 0
    started = asyncio.Event()
    finish = asyncio.Event()

    @alru_cache()
    async def coro() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
        await finish.wait()
        return "foo"

    t = asyncio.create_task(coro())
    await started.wait()
    assert len(coro._LRUCacheWrapper__tasks) == 1  # type: ignore[attr-defined]
    inner_task = next(iter(coro._LRUCacheWrapper__tasks))  # type: ignore[attr-defined]
    assert not inner_task.done()

    coro.cache_clear()
    replacement = asyncio.create_task(coro())
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert calls == 2

    finish.set()
    assert list(await asyncio.gather(t, replacement)) == ["foo", "foo"]
    await inner_task
    assert await t == "foo"
    assert inner_task.done()
    assert await coro() == "foo"
    assert calls == 2


async def test_cache_clear_forgets_evicted_pending_computation() -> None:
    started = [asyncio.Event() for _ in range(3)]
    finish = [asyncio.Event() for _ in range(3)]
    calls = 0

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        nonlocal calls
        index = calls
        calls += 1
        if val == 1:
            started[index].set()
            await finish[index].wait()
        return index

    old = asyncio.create_task(coro(1))
    await started[0].wait()
    assert await coro(2) == 1

    coro.cache_clear()
    assert coro.cache_parameters()["tasks"] == 0
    replacement = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert calls == 3

    finish[2].set()
    finish[0].set()
    assert list(await asyncio.gather(old, replacement)) == [0, 2]


async def test_cache_clear_ttl_callback(check_lru: Callable[..., None]) -> None:
    @alru_cache(ttl=0.5)
    async def coro() -> str:
        return "foo"

    await coro()
    assert len(coro._LRUCacheWrapper__cache) == 1  # type: ignore[attr-defined]
    cache_item = next(iter(coro._LRUCacheWrapper__cache.values()))  # type: ignore[attr-defined]
    assert not cache_item.later_call.cancelled()

    coro.cache_clear()

    assert cache_item.later_call.cancelled()
    await asyncio.sleep(0.5)
