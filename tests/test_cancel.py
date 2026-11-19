import asyncio
import gc
import weakref
from typing import Any

import pytest

from async_lru import alru_cache


@pytest.mark.parametrize("num_to_cancel", (0, 1, 2, 3))
async def test_cancel(num_to_cancel: int) -> None:
    @alru_cache
    async def coro(val: int) -> int:
        # I am a long running coro function
        await asyncio.sleep(0.1)
        return val

    # create 3 tasks for the cached function using the same key
    tasks = [asyncio.create_task(coro(i)) for i in range(3)]

    # force the event loop to run once so the tasks can begin
    await asyncio.sleep(0)

    # maybe cancel some tasks
    for i in range(num_to_cancel):
        tasks[i].cancel()

    # allow enough time for the non-cancelled tasks to complete
    await asyncio.sleep(0.2)

    # check tasks are properly cancelled
    for i in range(num_to_cancel):
        assert tasks[i].cancelled()

    # check non-cancelled tasks return expected outputs
    for i in range(num_to_cancel, 3):
        assert await tasks[i] == i


@pytest.mark.asyncio
async def test_cancel_single_waiter_triggers_handle_cancelled_error() -> None:
    # This test ensures the _handle_cancelled_error path (waiters == 1) is exercised.
    cache_item_task_finished = False

    @alru_cache
    async def coro(val: int) -> None:
        nonlocal cache_item_task_finished
        await asyncio.sleep(2)
        cache_item_task_finished = True  # pragma: no cover

    task = asyncio.create_task(coro(42))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # The underlying coroutine should be cancelled, so the flag should remain False
    assert cache_item_task_finished is False


@pytest.mark.parametrize("evict", (False, True))
async def test_cancel_one_waiter_preserves_shared_work(evict: bool) -> None:
    started = asyncio.Event()
    finish = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        nonlocal calls
        calls += 1
        if val == 1:
            started.set()
            try:
                await finish.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return val

    first = asyncio.create_task(coro(1))
    await started.wait()
    if evict:
        assert await coro(2) == 2
    second = asyncio.create_task(coro(1))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert not cancelled.is_set()
    assert coro.cache_parameters()["tasks"] == 1

    finish.set()
    assert await second == 1
    assert calls == (2 if evict else 1)
    assert coro.cache_parameters()["tasks"] == 0


@pytest.mark.parametrize("evict", (False, True))
async def test_cancel_last_waiter_cancels_shared_work(evict: bool) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        if val == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return val

    waiter = asyncio.create_task(coro(1))
    await started.wait()
    if evict:
        assert await coro(2) == 2
    assert coro.cache_parameters()["tasks"] == 1

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.sleep(0)
    assert cancelled.is_set()
    assert coro.cache_parameters()["tasks"] == 0


@pytest.mark.parametrize("evict", (False, True))
async def test_cancelled_computation_cannot_remove_replacement(evict: bool) -> None:
    started = asyncio.Event()
    cancellation_started = asyncio.Event()
    finish_cancellation = asyncio.Event()
    done_callback_ran = asyncio.Event()
    old_task_ref: weakref.ReferenceType[asyncio.Task[Any]] | None = None
    calls = 0

    @alru_cache(maxsize=1)
    async def coro(val: int) -> int:
        nonlocal calls, old_task_ref
        calls += 1
        if val == 1 and calls == 1:
            current_task = asyncio.current_task()
            assert current_task is not None
            old_task_ref = weakref.ref(current_task)
            current_task.add_done_callback(lambda _: done_callback_ran.set())
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_started.set()
                await finish_cancellation.wait()
                raise
        return val

    first = asyncio.create_task(coro(1))
    await started.wait()
    if evict:
        assert await coro(2) == 2
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await cancellation_started.wait()

    assert await coro(1) == 1
    assert coro.cache_parameters()["tasks"] == 1
    finish_cancellation.set()
    await done_callback_ran.wait()
    gc.collect()
    assert old_task_ref is not None
    assert old_task_ref() is None
    assert await coro(1) == 1
    assert calls == (3 if evict else 2)
    assert coro.cache_parameters()["tasks"] == 0
