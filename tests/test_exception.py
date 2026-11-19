import asyncio
import gc
import sys
import weakref
from collections.abc import Callable
from typing import Any

import pytest

from async_lru import alru_cache


async def test_alru_exception(check_lru: Callable[..., None]) -> None:
    @alru_cache()
    async def coro(val: int) -> None:
        1 / 0

    inputs = [1, 1, 1]
    coros = [coro(v) for v in inputs]

    ret = await asyncio.gather(*coros, return_exceptions=True)

    check_lru(coro, hits=2, misses=1, cache=0, tasks=0)

    for item in ret:
        assert isinstance(item, ZeroDivisionError)

    with pytest.raises(ZeroDivisionError):
        await coro(1)

    check_lru(coro, hits=2, misses=2, cache=0, tasks=0)


async def test_evicted_exception_clears_inflight_entry(
    check_lru: Callable[..., None]
) -> None:
    started = asyncio.Event()
    fail = asyncio.Event()
    inner_task_done = asyncio.Event()
    calls = 0
    inner_task_ref: weakref.ReferenceType[asyncio.Task[Any]] | None = None

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        nonlocal calls, inner_task_ref
        calls += 1
        if val == 1 and calls == 1:
            inner_task = asyncio.current_task()
            assert inner_task is not None
            inner_task_ref = weakref.ref(inner_task)
            inner_task.add_done_callback(lambda _: inner_task_done.set())
            started.set()
            await fail.wait()
            raise ValueError("failed")
        return val

    pending = asyncio.create_task(coro(1))
    await started.wait()
    assert await coro(2) == 2
    check_lru(coro, hits=0, misses=2, cache=1, tasks=1, maxsize=1)

    fail.set()
    with pytest.raises(ValueError, match="failed") as error:
        await pending
    await inner_task_done.wait()
    check_lru(coro, hits=0, misses=2, cache=1, tasks=0, maxsize=1)
    error.value.__traceback__ = None
    del error
    del pending
    gc.collect()
    assert inner_task_ref is not None
    assert inner_task_ref() is None

    assert await coro(1) == 1
    assert calls == 3
    check_lru(coro, hits=0, misses=3, cache=1, tasks=0, maxsize=1)


@pytest.mark.xfail(
    reason="Memory leak is not fixed for PyPy",
    condition=sys.implementation.name == "pypy",
)
async def test_alru_exception_reference_cleanup(check_lru: Callable[..., None]) -> None:
    class CustomClass:
        ...

    @alru_cache()
    async def coro(val: int) -> None:
        _ = CustomClass()  # object we are verifying not to leak
        1 / 0

    coros = [coro(v) for v in range(1000)]

    await asyncio.gather(*coros, return_exceptions=True)

    check_lru(coro, hits=0, misses=1000, cache=0, tasks=0)

    await asyncio.sleep(0.00001)
    gc.collect()

    assert (
        len([obj for obj in gc.get_objects() if isinstance(obj, CustomClass)]) == 0
    ), "Only objects in the cache should be left in memory."
