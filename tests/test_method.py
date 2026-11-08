import copy
import dataclasses
import pickle
import threading
import time
from typing import NamedTuple

import cachetools
import pytest

from mcrotk_cache_utils import cached_method, clear_caches


class Calc:
    """Counts body runs per method, so tests can tell hits from misses."""

    def __init__(self, scale=1):
        self.scale = scale
        self.runs = []

    @cached_method
    def mul(self, x, y=2):
        self.runs.append(("mul", x, y))
        return self.scale * x * y

    @cached_method(omit="verbose")
    def add(self, x, verbose=False):
        self.runs.append(("add", x))
        return self.scale + x

    @cached_method(key_converters={"xs": tuple})
    def total(self, xs):
        self.runs.append(("total", tuple(xs)))
        return self.scale * sum(xs)

    @cached_method
    def kw(self, *args, **kwargs):
        self.runs.append(("kw", args, tuple(sorted(kwargs.items()))))
        return args, kwargs

    @cached_method(passthrough={"token": 0})
    def tagged(self, token, x):
        self.runs.append(("tagged", x))
        return token, self.scale * x


def test_repeated_call_runs_once():
    c = Calc()
    assert c.mul(3) == 6 and c.mul(3) == 6
    assert c.runs == [("mul", 3, 2)]


def test_bound_arguments_give_one_key():
    c = Calc()
    c.mul(3)
    c.mul(x=3)
    c.mul(3, 2)
    c.mul(3, y=2)
    assert c.runs == [("mul", 3, 2)]
    c.mul(3, 4)
    assert c.runs == [("mul", 3, 2), ("mul", 3, 4)]


def test_omit():
    c = Calc()
    c.add(1, verbose=False)
    c.add(1, verbose=True)
    assert c.runs == [("add", 1)]


def test_key_converter_makes_list_hashable():
    c = Calc()
    assert c.total([1, 2, 3]) == 6
    assert c.total([1, 2, 3]) == 6
    assert c.runs == [("total", (1, 2, 3))]


def test_var_positional_and_keyword():
    c = Calc()
    c.kw(1, 2, a=1, b=2)
    c.kw(1, 2, b=2, a=1)
    c.kw(1, 2, a=1)
    assert len(c.runs) == 2


def test_passthrough_without_transform():
    c = Calc()
    assert c.tagged("first", 5) == ("first", 5)
    assert c.tagged("second", 5) == ("second", 5)  # hit: caller's token
    assert c.runs == [("tagged", 5)]


def test_caches_are_per_instance():
    a, b = Calc(1), Calc(10)
    assert a.mul(3) == 6 and b.mul(3) == 60
    assert a.__dict__["_Calc_mul_cache"] is not b.__dict__["_Calc_mul_cache"]


def test_cache_attribute_name_and_type():
    c = Calc()
    c.mul(1)
    assert type(c.__dict__["_Calc_mul_cache"]) is dict


def test_cache_factory():
    class C:
        @cached_method(cache_factory=lambda: cachetools.LRUCache(maxsize=2))
        def f(self, x):
            return x

    c = C()
    for x in range(5):
        c.f(x)
    cache = c.__dict__["_C_f_cache"]
    assert isinstance(cache, cachetools.LRUCache) and len(cache) == 2


def test_local_class_attribute_name():
    class Local:
        @cached_method
        def f(self):
            return 1

    obj = Local()
    obj.f()
    assert "_Local_f_cache" in obj.__dict__


def test_super_override_keeps_caches_separate():
    class Base:
        @cached_method
        def score(self, x):
            return x

    class Child(Base):
        @cached_method
        def score(self, x):
            return 2 * super().score(x)

        def parent_score(self, x):
            return super().score(x)

    c = Child()
    assert c.parent_score(3) == 3
    assert c.score(3) == 6
    assert c.score(3) == 6
    assert {"_Base_score_cache", "_Child_score_cache"} <= set(c.__dict__)


def test_class_access():
    c = Calc()
    assert Calc.mul(c, 3) == 6
    c.mul(3)
    assert c.runs == [("mul", 3, 2)]


def test_dataclass_fields_unaffected():
    @dataclasses.dataclass(frozen=True)
    class Frozen:
        n: int

        @cached_method
        def double(self):
            return 2 * self.n

    a, b = Frozen(1), Frozen(1)
    assert a.double() == 2
    assert a == b and repr(a) == "test_dataclass_fields_unaffected.<locals>.Frozen(n=1)"
    assert dataclasses.replace(a, n=5).double() == 10  # starts with an empty cache


def test_deepcopy_and_pickle_get_own_caches():
    c = Calc(1)
    c.mul(3)
    d = copy.deepcopy(c)
    d.scale = 100
    assert d.mul(4) == 800  # binds to the copy
    assert d.mul(3) == 6  # cached result carried over (stale by design; see clear_caches)
    assert d.__dict__["_Calc_mul_cache"] is not c.__dict__["_Calc_mul_cache"]
    p = pickle.loads(pickle.dumps(c))
    assert p.mul(3) == 6 and p.runs == [("mul", 3, 2)]


def test_clear_caches_all_and_named():
    c = Calc()
    c.mul(1)
    c.add(1)
    clear_caches(c, "mul")
    assert c.__dict__["_Calc_mul_cache"] == {} and len(c.__dict__["_Calc_add_cache"]) == 1
    clear_caches(c)
    assert c.__dict__["_Calc_add_cache"] == {}
    with pytest.raises(ValueError, match="no cached method"):
        clear_caches(c, "nope")


def test_clear_caches_leaves_other_attributes_and_reaches_parent():
    class Base:
        @cached_method
        def f(self, x):
            return x

    class Child(Base):
        @cached_method
        def f(self, x):
            return super().f(x) + 1

    c = Child()
    c.mine = {"keep": 1}
    c.f(1)
    clear_caches(c)
    assert c.__dict__["_Base_f_cache"] == {} and c.__dict__["_Child_f_cache"] == {}
    assert c.mine == {"keep": 1}


def test_clear_caches_resets_info():
    class C:
        @cached_method(info=True)
        def f(self, x):
            return x

    c = C()
    c.f(1)
    c.f(1)
    assert c.f.cache_info().hits == 1
    clear_caches(c)
    assert c.f.cache_info().hits == 0 and c.f.cache_info().currsize == 0


def test_lock_and_condition_options():
    class C:
        def __init__(self):
            self.lock = threading.Lock()
            self.cond = threading.Condition()
            self.runs = 0

        @cached_method(lock=lambda self: self.lock)
        def locked(self, x):
            self.runs += 1
            return x

        @cached_method(condition=lambda self: self.cond)
        def conditioned(self, x):
            self.runs += 1
            return x

    c = C()
    threads = [threading.Thread(target=lambda: (c.locked(1), c.conditioned(2))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.locked(1) == 1 and c.conditioned(2) == 2
    assert c.__dict__["_C_locked_cache"] == {(1,): 1}


class Pair(NamedTuple):
    value: int
    token: object


class Shapes:
    @cached_method(passthrough={"token": -1})
    def named(self, x, token):
        return Pair(x, token)

    @cached_method(passthrough={-1: 0})
    def single(self, x, token):
        return (token,)

    @cached_method(passthrough={"token": 1})
    def middle(self, x, token):
        return x, token, x + 1


def test_passthrough_rebuilds_named_tuple():
    s = Shapes()
    s.named(1, "a")
    hit = s.named(1, "b")
    assert type(hit) is Pair and hit == Pair(1, "b")


def test_passthrough_one_tuple_stays_tuple():
    s = Shapes()
    assert s.single(1, "a") == ("a",)
    assert s.single(1, "b") == ("b",)


def test_passthrough_middle_position_and_stored_form():
    s = Shapes()
    s.middle(1, "a")
    assert s.middle(1, "b") == (1, "b", 2)
    (stored,) = s.__dict__["_Shapes_middle_cache"].values()
    assert stored.values == (1, 2) and stored.length == 3
    assert all(type(k) is tuple for k in s.__dict__["_Shapes_middle_cache"])


class Multi:
    def __init__(self):
        self.runs = 0

    @cached_method(passthrough={"a": 0, "b": (-2, str.upper)})
    def two(self, x, a, b):
        self.runs += 1
        return a, x, b, x + 1

    @cached_method(passthrough={2: (-1, str.upper)})
    def by_position(self, x, token):
        self.runs += 1
        return x, token


def test_two_passthrough_arguments_mixed_positions_and_transforms():
    m = Multi()
    assert m.two(1, "first", "low") == ("first", 1, "low", 2)  # miss: the function's own outputs
    assert m.two(1, "second", "new") == ("second", 1, "NEW", 2)  # hit: a as is at 0, b transformed at -2
    assert m.runs == 1


def test_transform_applies_only_on_hit():
    m = Multi()
    assert m.by_position(1, "abc") == (1, "abc")  # miss: not transformed
    assert m.by_position(1, "xyz") == (1, "XYZ")  # hit: transformed
    assert m.runs == 1


def test_passthrough_argument_positional_or_keyword():
    m = Multi()
    m.by_position(1, "abc")
    assert m.by_position(x=1, token="def") == (1, "DEF")
    assert m.runs == 1


def test_called_form_without_options():
    class C:
        @cached_method()
        def f(self, x):
            return [x]

    c = C()
    assert c.f(1) is c.f(1)


def test_condition_prevents_duplicate_computation():
    entered, release = threading.Event(), threading.Event()

    class C:
        def __init__(self):
            self.cond = threading.Condition()
            self.runs = []

        @cached_method(condition=lambda self: self.cond)
        def slow(self, x):
            self.runs.append(x)
            entered.set()
            release.wait(5)
            return x

    c = C()
    results = []
    first = threading.Thread(target=lambda: results.append(c.slow(1)))
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=lambda: results.append(c.slow(1)))
    second.start()
    time.sleep(0.05)  # give the second thread time to reach the condition
    assert c.runs == [1]  # the second thread waits instead of computing
    release.set()
    first.join(5)
    second.join(5)
    assert results == [1, 1] and c.runs == [1]


def test_lock_race_each_caller_keeps_its_own_passthrough_outputs():
    barrier = threading.Barrier(2, timeout=5)

    class C:
        def __init__(self):
            self.lock = threading.Lock()
            self.runs = []

        @cached_method(lock=lambda self: self.lock, passthrough={"token": -1})
        def f(self, x, token):
            self.runs.append(token)
            barrier.wait()  # both threads have missed before either stores
            return x, token + "-out"

    c = C()
    results = {}
    threads = [threading.Thread(target=lambda t=t: results.__setitem__(t, c.f(1, t))) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results == {"a": (1, "a-out"), "b": (1, "b-out")}
    assert sorted(c.runs) == ["a", "b"] and len(c.__dict__["_C_f_cache"]) == 1
    assert c.f(1, "c") == (1, "c")  # later calls hit


class Bounded:
    def __init__(self):
        self.runs = 0

    @cached_method(cache_factory=lambda: cachetools.LRUCache(maxsize=2), passthrough={"t": -1})
    def lru(self, x, t):
        self.runs += 1
        return x, t


def test_lru_cache_through_passthrough_view():
    e = Bounded()
    for x in (1, 2, 3):
        e.lru(x, "a")
    assert e.lru(3, "b") == (3, "b") and e.runs == 3  # kept
    assert e.lru(1, "b") == (1, "b") and e.runs == 4  # dropped by the LRU limit, recomputed


def test_ttl_cache_through_passthrough_view():
    now = [0]

    class C:
        def __init__(self):
            self.runs = 0

        @cached_method(cache_factory=lambda: cachetools.TTLCache(maxsize=10, ttl=10, timer=lambda: now[0]), passthrough={"t": -1})
        def f(self, x, t):
            self.runs += 1
            return x, t

    c = C()
    c.f(1, "a")
    assert c.f(1, "b") == (1, "b") and c.runs == 1
    now[0] = 11
    assert c.f(1, "c") == (1, "c") and c.runs == 2  # expired, recomputed


def test_key_converter_on_var_keyword():
    class C:
        def __init__(self):
            self.runs = 0

        @cached_method(key_converters={"kwargs": lambda d: tuple(sorted((k, tuple(v)) for k, v in d.items()))})
        def f(self, **kwargs):
            self.runs += 1
            return sum(map(sum, kwargs.values()))

    c = C()
    assert c.f(a=[1, 2], b=[3]) == 6
    assert c.f(b=[3], a=[1, 2]) == 6
    assert c.runs == 1


def test_transform_error_reaches_caller_and_leaves_cache_unchanged():
    def bad(value):
        raise RuntimeError("bad transform")

    class C:
        @cached_method(passthrough={"t": (-1, bad)})
        def f(self, x, t):
            return x, t

    c = C()
    assert c.f(1, "a") == (1, "a")  # miss: transform not used
    with pytest.raises(RuntimeError, match="bad transform"):
        c.f(1, "b")
    (stored,) = c.__dict__["_C_f_cache"].values()
    assert stored.values == (1,) and len(c.__dict__["_C_f_cache"]) == 1


# Documented caveats: these pin down current behavior, so a change in it gets noticed.


def test_caveat_returned_objects_are_the_cached_objects():
    class C:
        @cached_method
        def f(self, x):
            return [x]

    c = C()
    c.f(1).append(99)
    assert c.f(1) == [1, 99]


def test_caveat_stale_after_attribute_change_until_clear_caches():
    c = Calc(1)
    assert c.mul(3) == 6
    c.scale = 10
    assert c.mul(3) == 6  # stale: instance attributes aren't in the key
    clear_caches(c)
    assert c.mul(3) == 60
