"""What type checkers see: decorated functions and methods keep their own types.

Checked by pyright and mypy (CI's lint job), and also run by pytest, so every intended use here
also works at runtime. The calls under ``TYPE_CHECKING`` at the end are mistakes that must be
reported: each has a ``# type: ignore``, and both checkers are set to report an ignore that isn't
needed, so a mistake that stops being reported fails the check.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=true

import abc
import copy
import importlib.util
from typing import TYPE_CHECKING, Any, TypeVar, assert_type, overload

import pytest

from mcrotk_cache_utils import (
    CacheControls,
    cache_controls,
    cached_function,
    cached_method,
    clear_caches,
    copy_instance,
    rng_passthrough_cached_function,
    rng_passthrough_cached_method,
)

T = TypeVar("T")

needs_numpy = pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="needs numpy")


@cached_function
def area(width: float, height: float = 1.0) -> float:
    return width * height


@cached_function(omit="verbose", info=True)
def total(xs: list[int], verbose: bool = False) -> int:
    return sum(xs)


@cached_function
def first(xs: tuple[T, ...]) -> T:
    return xs[0]


@overload
def double(x: int) -> int: ...
@overload
def double(x: str) -> str: ...
@cached_function
def double(x: int | str) -> int | str:
    return x * 2


class Model:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    @cached_method
    def fit(self, n: int, *, scale: float = 1.0) -> list[float]:
        return [self.scale * scale] * n

    @cached_method(info=True, on_copy="copy")
    def label(self, x: str) -> str:
        return x.upper()

    def plain(self, n: int) -> int:
        return n

    @staticmethod
    @cached_function
    def kernel(n: int) -> int:
        return 2 * n

    @classmethod
    @cached_function(info=True)
    def defaults(cls, n: int) -> tuple[str, int]:
        return cls.__name__, n


class Sub(Model):
    def fit(self, n: int, *, scale: float = 1.0) -> list[float]:  # plain overriding cached
        return super().fit(n, scale=scale)

    @cached_method
    def plain(self, n: int) -> int:  # cached overriding plain
        return super().plain(n) + 1

    @staticmethod
    def kernel(n: int) -> int:  # plain static overriding cached
        return n

    @classmethod
    def defaults(cls, n: int) -> tuple[str, int]:  # plain class method overriding cached
        return "sub", n


class Shape(abc.ABC):
    @abc.abstractmethod
    @cached_method
    def size(self) -> int: ...


class Square(Shape):
    @cached_method
    def size(self) -> int:
        return 4


def test_functions():
    assert_type(area(2.0, 3.0), float)
    assert_type(area(2.0), float)
    assert_type(total([1, 2], verbose=True), int)
    assert_type(first((1, 2)), int)
    assert_type(first(("a",)), str)
    assert_type(double(2), int)
    assert_type(double("a"), str)
    assert area(2.0, 3.0) == 6.0 and total([1, 2]) == 3 and first(("a",)) == "a" and double("a") == "aa"


def test_methods():
    m = Model(2.0)
    assert_type(m.fit(2), list[float])
    assert_type(m.fit(2, scale=3.0), list[float])
    assert_type(Model.fit(m, 2), list[float])
    assert_type(m.label("a"), str)
    assert m.fit(2, scale=3.0) == [6.0, 6.0] and m.label("a") == "A"


def test_static_and_class_methods():
    assert_type(Model.kernel(2), int)
    assert_type(Model().kernel(2), int)
    assert_type(Model.defaults(2), tuple[str, int])
    assert_type(Model().defaults(2), tuple[str, int])
    assert Model.kernel(2) == 4 and Model.defaults(2) == ("Model", 2)


def test_overrides_in_either_direction():
    s = Sub()
    assert_type(s.fit(1), list[float])
    assert_type(s.plain(1), int)
    assert s.fit(1) == [1.0] and s.plain(1) == 2 and Sub.kernel(3) == 3 and Sub.defaults(1) == ("sub", 1)
    assert_type(Square().size(), int)
    assert Square().size() == 4


def test_cache_controls():
    controls = cache_controls(total)
    assert_type(controls, CacheControls)
    m = Model()
    m.label("a")
    m.label("a")
    Model.defaults(1)
    assert cache_controls(m.label).cache_info().hits == 1
    assert cache_controls(Model.defaults).cache_info().misses >= 1
    assert cache_controls(area).cache is not None
    cache_controls(Model.kernel).cache_clear()
    cache_controls(m.fit).cache_clear()
    clear_caches(m)


def test_copies_keep_the_type():
    m = Model()
    assert_type(copy_instance(m), Model)
    assert_type(copy_instance(m, "copy", deep=True), Model)
    assert_type(copy.copy(m), Model)
    assert_type(copy.deepcopy(m), Model)


@needs_numpy
def test_rng_decorators():
    import numpy as np

    @rng_passthrough_cached_function
    def draw(n: int, rng: Any = None) -> tuple[float, np.random.Generator]:
        rng = np.random.default_rng(rng)
        return float(rng.random(n).sum()), rng

    class Sim:
        @rng_passthrough_cached_method(info=True)
        def draw(self, n: int, rng: Any = None) -> tuple[float, np.random.Generator]:
            rng = np.random.default_rng(rng)
            return float(rng.random(n).sum()), rng

    assert_type(draw(3, rng=1), tuple[float, np.random.Generator])
    assert_type(Sim().draw(3, rng=1), tuple[float, np.random.Generator])
    assert draw(3, rng=1)[0] == Sim().draw(3, rng=1)[0]


if TYPE_CHECKING:
    # Mistakes that must be reported (see the module docstring).
    area("wide")  # type: ignore
    area(1.0, 2.0, 3.0)  # type: ignore
    total([1], verbose=True, extra=2)  # type: ignore
    wrong: str = area(1.0)  # type: ignore
    Model().fit("two")  # type: ignore
    Model().fit(2, scale="big")  # type: ignore
    Model.fit(2)  # type: ignore
    Model.kernel()  # type: ignore
    Model.defaults("two")  # type: ignore
    Model().label(1)  # type: ignore
    double(2.5)  # type: ignore
    cache_controls(area).cache_clear(1)  # type: ignore
    size = cache_controls(area).cache_size  # type: ignore
    # The documented limitation: the cache attributes aren't part of the decorated function's type.
    area.cache_clear()  # type: ignore
    Model().fit.cache_info()  # type: ignore
