"""Compare legacy quadratic keyword-key construction with the linear builder.

Run from the repository root with
``PYTHONPATH=. python benchmarks/benchmark_keyword_keys.py``.
Use CPython 3.11 to measure the affected standard-library implementation
against the local builder. The benchmark builds each key and hashes it six
times, matching repeated cache dictionary lookups. A second table covers
small positional-only keys, such as a bound-method ``(self, x)`` call. It has no
timing assertions and uses only the standard library.
"""

import functools
import sys
from collections.abc import Callable, Hashable
from statistics import median
from timeit import repeat
from typing import cast

from async_lru import _make_cache_key


def _measure(
    builder: Callable[..., Hashable],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    iterations: int,
) -> int:
    samples = repeat(
        lambda: _build_and_hash(builder, args, kwargs),
        number=iterations,
        repeat=5,
    )
    return round(median(samples) * 1_000_000_000 / iterations)


def _build_and_hash(
    builder: Callable[..., Hashable],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    key = builder(args, kwargs, False)
    for _ in range(6):
        hash(key)


def main() -> None:
    old_builder = cast(Callable[..., Hashable], functools._make_key)
    print(f"Python {sys.version.split()[0]}; six key hashes per call")
    print("keywords  iterations  stdlib ns/call  linear ns/call  speedup")
    for count in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        kwargs = {f"kw{index}": index for index in range(count)}
        iterations = max(100, 100_000 // max(count, 1))
        legacy_ns = _measure(old_builder, (), kwargs, iterations)
        linear_ns = _measure(_make_cache_key, (), kwargs, iterations)
        print(
            f"{count:8}  {iterations:10}  {legacy_ns:14}  "
            f"{linear_ns:14}  {legacy_ns / linear_ns:6.2f}x"
        )

    print()
    print("positional-only small keys")
    print("shape                  iterations  stdlib ns/call  linear ns/call  speedup")
    owner = object()
    shapes: dict[str, tuple[object, ...]] = {
        "(self, int)": (owner, 1),
        "(int, int)": (1, 2),
        "(self, int, str)": (owner, 1, "a"),
    }
    for label, args in shapes.items():
        legacy_ns = _measure(old_builder, args, {}, 200_000)
        linear_ns = _measure(_make_cache_key, args, {}, 200_000)
        print(
            f"{label:20}  {200_000:10}  {legacy_ns:14}  "
            f"{linear_ns:14}  {legacy_ns / linear_ns:6.2f}x"
        )


if __name__ == "__main__":
    main()
