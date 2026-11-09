"""Errors for decorators used in the wrong place: cached_function on a method, cached_method off one."""

import abc
import importlib.util

import pytest

from mcrotk_cache_utils import cached_function, cached_method, rng_cached_function, rng_cached_method

needs_numpy = pytest.mark.skipif(importlib.util.find_spec("numpy") is None, reason="needs numpy")


def error_text(e: BaseException) -> str:
    # Python 3.11 wraps errors raised in __set_name__ in a RuntimeError, with ours as the cause.
    return f"{e} {e.__cause__ or ''}"


@pytest.mark.parametrize(
    "decorator", [cached_function, cached_function(), pytest.param(rng_cached_function, marks=needs_numpy)]
)
def test_cached_function_directly_in_class_body(decorator):
    with pytest.raises((TypeError, RuntimeError)) as e:

        class C:
            @decorator
            def f(self, x, rng=None):
                return x, rng

    text = error_text(e.value)
    assert "C.f: @cached_function can't cache a method" in text and "Use @cached_method" in text


@pytest.mark.parametrize("wrapper", [staticmethod, classmethod])
def test_cached_function_above_static_or_class_method(wrapper):
    with pytest.raises(TypeError, match=f"@cached_function is above @{wrapper.__name__}"):

        class C:
            @cached_function
            @wrapper
            def f(x):
                return x


@pytest.mark.parametrize("decorator", [cached_method, pytest.param(rng_cached_method, marks=needs_numpy)])
@pytest.mark.parametrize("wrapper", [staticmethod, classmethod])
def test_cached_method_under_static_or_class_method(decorator, wrapper):
    class C:
        @wrapper
        @decorator
        def f(x, rng=None):
            return x, rng

    for call in (lambda: C.f(1), lambda: C().f(1)):  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match="C.f: @cached_method only works on ordinary methods.*@cached_function"):
            call()


@pytest.mark.parametrize("wrapper", [staticmethod, classmethod])
def test_cached_method_above_static_or_class_method(wrapper):
    with pytest.raises(TypeError, match=f"@cached_method is above @{wrapper.__name__}.*@cached_function"):

        class C:
            @cached_method
            @wrapper
            def f(x):
                return x


def test_cached_method_on_a_function():
    @cached_method
    def f(x):
        return x

    with pytest.raises(TypeError, match="f: @cached_method only works on ordinary methods"):
        f(1)


def test_cached_method_under_abstractmethod_works():
    class Base(abc.ABC):
        @abc.abstractmethod
        @cached_method
        def f(self, x): ...

    class Impl(Base):
        @cached_method
        def f(self, x):
            return x + 1

    with pytest.raises(TypeError, match="abstract"):
        Base()  # pyright: ignore[reportAbstractUsage]
    assert Impl().f(1) == 2
