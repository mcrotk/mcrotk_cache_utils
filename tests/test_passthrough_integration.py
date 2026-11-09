"""cachetools' cachedmethod features and edge cases combined with passthrough (its key and cache view)."""

import copy
import pickle
import threading
import time
from typing import Any, NamedTuple

import cachetools
import pytest

from mcrotk_cache_utils import cached_method, clear_caches


class Result(NamedTuple):
    value: int
    token: str


class Tagged:
    def __init__(self):
        self.runs = []

    @cached_method(passthrough={"token": -1})
    def f(self, x, token):
        self.runs.append((x, token))
        return Result(x, token)

    @cached_method(passthrough={"token": -1}, info=True, cache_factory=lambda: cachetools.LRUCache(maxsize=2))
    def bounded(self, x, token):
        self.runs.append((x, token))
        return x, token


def test_super_override_with_passthrough():
    class Base:
        @cached_method(passthrough={"t": -1})
        def f(self, x, t):
            return x, t

    class Child(Base):
        @cached_method(passthrough={"t": -1})
        def f(self, x, t):
            value, t = super().f(x, t)
            return 2 * value, t + "!"

        def parent_f(self, x, t):
            return super().f(x, t)

    c = Child()
    assert c.parent_f(3, "a") == (3, "a")  # miss in Base's cache
    assert c.f(3, "b") == (6, "b!")  # miss in Child's cache; Base's is a hit, so its t is "b"
    assert c.f(3, "c") == (6, "c")  # hit in Child's cache
    assert c.parent_f(3, "d") == (3, "d")  # hit in Base's cache, unaffected by Child's
    assert {"_Base_f_cache", "_Child_f_cache"} <= set(c.__dict__)


def test_class_access_with_passthrough():
    t = Tagged()
    assert Tagged.f(t, 3, "a") == Result(3, "a")
    assert t.f(3, "b") == Result(3, "b")
    assert Tagged.f(t, 3, "c") == Result(3, "c")
    assert t.runs == [(3, "a")]


def test_pickling_carries_passthrough_cache():
    t = Tagged()
    t.f(3, "a")
    c = pickle.loads(pickle.dumps(t))
    hit = c.f(3, "new")
    assert hit == Result(3, "new") and type(hit) is Result  # the stored result type survives the copy
    assert c.runs == [(3, "a")]  # the copy's own runs list (copied), no new run
    assert c.__dict__["_Tagged_f_cache"] is not t.__dict__["_Tagged_f_cache"]


def test_locally_defined_named_tuple_result_cannot_be_pickled():
    # Documented limitation: the cache stores the result's type, and a class defined inside a
    # function can't be pickled. deepcopy still works (classes are copied by reference).
    class LocalResult(NamedTuple):
        value: int
        token: str

    class Local:
        @cached_method(passthrough={"token": -1})
        def f(self, x, token):
            return LocalResult(x, token)

    obj = Local()
    obj.f(1, "a")
    assert copy.deepcopy(obj).f(1, "b") == LocalResult(1, "b")
    with pytest.raises((pickle.PicklingError, AttributeError)):
        pickle.dumps(obj)


def test_clear_caches_with_passthrough_and_info():
    t = Tagged()
    t.bounded(1, "a")
    t.bounded(1, "b")
    assert t.bounded.cache_info().hits == 1
    clear_caches(t, "bounded")
    info = t.bounded.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)
    assert t.bounded(1, "c") == (1, "c") and t.runs == [(1, "a"), (1, "c")]  # a miss again


def test_info_with_bounded_cache_through_view():
    t = Tagged()
    for x in (1, 2, 3):
        t.bounded(x, "a")
    t.bounded(3, "b")
    info = t.bounded.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 3, 2, 2)


def test_condition_with_passthrough():
    entered, release = threading.Event(), threading.Event()

    class C:
        def __init__(self):
            self.cond = threading.Condition()
            self.runs = []

        @cached_method(condition=lambda self: self.cond, passthrough={"token": -1})
        def slow(self, x, token):
            self.runs.append(token)
            entered.set()
            release.wait(5)
            return x, token + "-out"

    c = C()
    results = {}
    first = threading.Thread(target=lambda: results.__setitem__("a", c.slow(1, "a")))
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=lambda: results.__setitem__("b", c.slow(1, "b")))
    second.start()
    time.sleep(0.05)  # give the second thread time to reach the condition
    release.set()
    first.join(5)
    second.join(5)
    assert c.runs == ["a"]  # computed once
    assert results == {"a": (1, "a-out"), "b": (1, "b")}  # a: its own outputs (miss); b: a hit with its own token


def test_reading_the_view_directly():
    t = Tagged()
    t.bounded(3, "a")
    view = t.bounded.cache  # the view cachetools sees, not the real cache
    assert list(view) == [(3,)] and (3,) in view and len(view) == 1
    stored = view[(3,)]  # outside a call: the internal stored form
    assert (stored.length, stored.values) == (2, (3,))
    assert view.maxsize == 2  # attributes of the real cache come through
    assert isinstance(t.__dict__["_Tagged_bounded_cache"], cachetools.LRUCache)
    t.bounded.cache_clear()
    assert len(t.__dict__["_Tagged_bounded_cache"]) == 0


def test_view_over_plain_dict_reports_no_maxsize():
    t = Tagged()
    t.f(1, "a")
    assert not isinstance(t.f.cache, cachetools.Cache)
    assert t.__dict__["_Tagged_f_cache"].__class__ is dict


def stored_size(stored: Any) -> int:
    return len(stored.values)  # the number of values stored (without the passed-through ones)


def test_view_over_cache_with_custom_getsizeof():
    class C:
        @cached_method(
            passthrough={"t": -1},
            info=True,
            cache_factory=lambda: cachetools.LRUCache(maxsize=10, getsizeof=stored_size),
        )
        def f(self, xs, t):
            return *xs, t

    c = C()
    c.f((1, 2, 3), "a")  # stored without t: 3 values
    c.f((4,), "a")  # 1 value
    assert c.f.cache_info().currsize == 4
    assert c.f.cache.getsizeof(c.f.cache[((1, 2, 3),)]) == 3
