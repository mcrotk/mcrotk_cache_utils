"""cache_controls: typed access to the cache attributes of cached functions and methods."""

import threading

import cachetools
import pytest

from mcrotk_cache_utils import cache_controls, cached_function, cached_method


@cached_function(info=True)
def square(x):
    return x * x


@cached_function(cache_factory=lambda: cachetools.LRUCache(maxsize=4), lock=threading.Lock())
def bounded(x):
    return x


class Model:
    @cached_method(info=True)
    def fit(self, x):
        return x

    @cached_method(passthrough={"t": -1}, cache_factory=lambda: cachetools.LRUCache(maxsize=3), info=True)
    def through(self, x, t):
        return x, t

    @cached_method
    def no_info(self, x):
        return x

    @staticmethod
    @cached_function
    def kernel(n):
        return n

    @classmethod
    @cached_function(info=True)
    def defaults(cls, n):
        return cls.__name__, n


def test_function():
    cache_controls(square).cache_clear()
    square(2)
    square(2)
    controls = cache_controls(square)
    assert controls is square  # the same object, typed
    assert controls.cache_info().hits == 1 and controls.cache == {(2,): 4}
    assert controls.cache_key(2) == (2,)
    controls.cache_clear()
    assert controls.cache == {} and controls.cache_info().hits == 0


def test_function_cache_and_lock():
    controls = cache_controls(bounded)
    assert isinstance(controls.cache, cachetools.LRUCache) and controls.cache.maxsize == 4
    assert controls.cache_lock is not None and controls.cache_condition is None


def test_method_on_an_instance():
    a, b = Model(), Model()
    a.fit(1)
    a.fit(1)
    b.fit(1)
    assert cache_controls(a.fit).cache_info().hits == 1
    assert cache_controls(b.fit).cache_info().hits == 0
    cache_controls(a.fit).cache_clear()
    assert len(cache_controls(a.fit).cache) == 0 and len(cache_controls(b.fit).cache) == 1


def test_passthrough_method():
    m = Model()
    m.through(1, "a")
    m.through(1, "b")
    info = cache_controls(m.through).cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 1, 3, 1)


def test_static_and_class_methods():
    Model.kernel(1)
    assert len(cache_controls(Model.kernel).cache) >= 1
    cache_controls(Model.kernel).cache_clear()
    assert len(cache_controls(Model().kernel).cache) == 0
    Model.defaults(1)
    Model().defaults(1)
    assert cache_controls(Model.defaults).cache_info().hits >= 1
    cache_controls(Model.defaults).cache_clear()
    assert cache_controls(Model.defaults).cache_info().currsize == 0


def test_cache_info_needs_info():
    with pytest.raises(AttributeError):
        cache_controls(Model().no_info).cache_info()


def test_method_accessed_on_the_class():
    with pytest.raises(
        TypeError, match=r"not a cached function or method \(pass a cached method accessed on an instance"
    ):
        cache_controls(Model.fit)


@pytest.mark.parametrize("value", [len, lambda x: x, 3, None])
def test_not_cached(value):
    with pytest.raises(TypeError, match="not a cached function or method$"):
        cache_controls(value)
