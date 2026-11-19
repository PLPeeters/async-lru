import asyncio
import sys
from collections.abc import Callable

import pytest

from async_lru import alru_cache


async def test_key_excludes_argument(check_lru: Callable[..., None]) -> None:
    """Calls differing only in the excluded argument share one entry."""

    @alru_cache(key=lambda conn, query: query)
    async def query_db(conn: str, query: str) -> str:
        return f"{conn}={query}"

    assert await query_db("conn1", "SELECT 1") == "conn1=SELECT 1"
    check_lru(query_db, hits=0, misses=1, cache=1, tasks=0)

    assert await query_db("conn2", "SELECT 1") == "conn1=SELECT 1"
    check_lru(query_db, hits=1, misses=1, cache=1, tasks=0)

    assert await query_db("conn2", "SELECT 2") == "conn2=SELECT 2"
    check_lru(query_db, hits=1, misses=2, cache=2, tasks=0)


async def test_key_receives_kwargs(check_lru: Callable[..., None]) -> None:
    @alru_cache(key=lambda conn, *, query: query)
    async def query_db(conn: str, *, query: str) -> str:
        return f"{conn}={query}"

    assert await query_db("conn1", query="q") == "conn1=q"
    assert await query_db("conn2", query="q") == "conn1=q"
    check_lru(query_db, hits=1, misses=1, cache=1, tasks=0)


async def test_key_cache_invalidate(check_lru: Callable[..., None]) -> None:
    """cache_invalidate computes the same custom key as the call did."""

    @alru_cache(key=lambda conn, query: query)
    async def query_db(conn: str, query: str) -> str:
        return f"{conn}={query}"

    await query_db("conn1", "SELECT 1")
    check_lru(query_db, hits=0, misses=1, cache=1, tasks=0)

    # A different excluded argument still addresses the same entry.
    assert query_db.cache_invalidate("conn9", "SELECT 1")
    check_lru(query_db, hits=0, misses=1, cache=0, tasks=0)
    assert not query_db.cache_invalidate("conn9", "SELECT 1")


async def test_key_cache_contains(check_lru: Callable[..., None]) -> None:
    @alru_cache(key=lambda conn, query: query)
    async def query_db(conn: str, query: str) -> str:
        return f"{conn}={query}"

    await query_db("conn1", "SELECT 1")
    assert query_db.cache_contains("conn9", "SELECT 1")
    assert not query_db.cache_contains("conn9", "SELECT 2")


def test_key_with_typed_raises() -> None:
    with pytest.raises(ValueError, match="typed"):
        alru_cache(typed=True, key=lambda x: x)


def test_key_not_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        alru_cache(key="query")  # type: ignore[call-overload]


async def test_typed_without_key_still_works() -> None:
    @alru_cache(typed=True)
    async def coro(val: int) -> int:
        return val

    assert coro.cache_parameters()["typed"] is True
    assert await coro(1) == 1


async def test_default_key_preserves_argument_form_and_keyword_order() -> None:
    calls = 0

    @alru_cache
    async def coro(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return calls

    positional = await coro("value", 1)
    keyword = await coro(value=1)
    first_order = await coro(first=1, second=2)
    reverse_order = await coro(second=2, first=1)

    assert (positional, keyword, first_order, reverse_order) == (1, 2, 3, 4)
    assert coro.cache_contains("value", 1)
    assert coro.cache_contains(value=1)
    assert coro.cache_contains(first=1, second=2)
    assert coro.cache_contains(second=2, first=1)
    assert coro.cache_invalidate("value", 1)
    assert coro.cache_invalidate(value=1)
    assert coro.cache_invalidate(first=1, second=2)
    assert coro.cache_invalidate(second=2, first=1)


@pytest.mark.parametrize(
    ("typed", "expected_calls"),
    [(False, 2), (True, 3)],
)
async def test_default_key_preserves_typed_single_argument_behavior(
    typed: bool, expected_calls: int
) -> None:
    calls = 0

    @alru_cache(typed=typed)
    async def coro(value: object) -> int:
        nonlocal calls
        calls += 1
        return calls

    integer = await coro(1)
    float_value = await coro(1.0)
    bool_value = await coro(True)
    same_integer = await coro(1)

    assert integer == same_integer
    if typed:
        assert float_value != bool_value
    else:
        assert float_value == bool_value
    assert calls == expected_calls
    assert coro.cache_contains(1)
    assert coro.cache_contains(1.0)
    assert coro.cache_contains(True)
    assert coro.cache_invalidate(1)
    assert coro.cache_invalidate(1.0)
    assert coro.cache_invalidate(True) is typed


@pytest.mark.parametrize(
    ("typed", "expected_calls"),
    [(False, 1), (True, 2)],
)
async def test_default_key_preserves_typed_keyword_value_behavior(
    typed: bool, expected_calls: int
) -> None:
    calls = 0

    @alru_cache(typed=typed)
    async def coro(**kwargs: object) -> int:
        nonlocal calls
        calls += 1
        return calls

    integer = await coro(value=1)
    float_value = await coro(value=1.0)

    if typed:
        assert integer != float_value
    else:
        assert integer == float_value
    assert calls == expected_calls
    assert coro.cache_contains(value=1)
    assert coro.cache_contains(value=1.0)
    assert coro.cache_invalidate(value=1)
    assert coro.cache_invalidate(value=1.0) is typed


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="tuple hash caching compatibility"
)
@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
async def test_default_key_memoizes_hash_on_supported_legacy_runtimes(
    keyword: bool,
) -> None:
    class HashCounter:
        def __init__(self) -> None:
            self.calls = 0

        def __hash__(self) -> int:
            self.calls += 1
            return id(self)

    @alru_cache
    async def coro(value: HashCounter) -> None:
        pass

    value = HashCounter()
    if keyword:
        await coro(value=value)
    else:
        await coro(value)
    assert value.calls == 1

    if keyword:
        await coro(value=value)
    else:
        await coro(value)
    assert value.calls == 2


async def test_key_on_method(check_lru: Callable[..., None]) -> None:
    """The key callable receives the instance for bound methods."""

    class Api:
        @alru_cache(key=lambda self, query: query)
        async def fetch(self, query: str) -> str:
            return f"{id(self)}={query}"

    first = Api()
    second = Api()
    result = await first.fetch("SELECT 1")
    # Another instance shares the entry since self is excluded.
    assert await second.fetch("SELECT 1") == result
    check_lru(Api.fetch, hits=1, misses=1, cache=1, tasks=0)


async def test_key_unhashable_result(check_lru: Callable[..., None]) -> None:
    @alru_cache(key=lambda val: [val])  # type: ignore[arg-type,return-value]
    async def coro(val: int) -> int:
        return val  # pragma: no cover  # the key fails before the call

    with pytest.raises(TypeError, match="unhashable"):
        await coro(1)
    check_lru(coro, hits=0, misses=0, cache=0, tasks=0)


async def test_key_callable_raising(check_lru: Callable[..., None]) -> None:
    def broken_key(val: int) -> int:
        raise RuntimeError("broken key")

    @alru_cache(key=broken_key)
    async def coro(val: int) -> int:
        return val  # pragma: no cover  # the key fails before the call

    with pytest.raises(RuntimeError, match="broken key"):
        await coro(1)
    check_lru(coro, hits=0, misses=0, cache=0, tasks=0)


async def test_key_with_ttl(check_lru: Callable[..., None]) -> None:
    @alru_cache(ttl=0.05, key=lambda conn, query: query)
    async def query_db(conn: str, query: str) -> str:
        return f"{conn}={query}"

    await query_db("conn1", "SELECT 1")
    assert await query_db("conn2", "SELECT 1") == "conn1=SELECT 1"
    check_lru(query_db, hits=1, misses=1, cache=1, tasks=0)

    await asyncio.sleep(0.06)
    assert await query_db("conn2", "SELECT 1") == "conn2=SELECT 1"
    check_lru(query_db, hits=1, misses=2, cache=1, tasks=0)


async def test_key_with_maxsize(check_lru: Callable[..., None]) -> None:
    @alru_cache(maxsize=1, key=lambda conn, query: query)
    async def query_db(conn: str, query: str) -> str:
        return f"{conn}={query}"

    await query_db("conn1", "SELECT 1")
    await query_db("conn1", "SELECT 2")
    check_lru(query_db, hits=0, misses=2, cache=1, tasks=0, maxsize=1)

    # The first entry was evicted; same query recomputes.
    assert await query_db("conn2", "SELECT 1") == "conn2=SELECT 1"
    check_lru(query_db, hits=0, misses=3, cache=1, tasks=0, maxsize=1)
