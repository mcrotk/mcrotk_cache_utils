import copy
import dataclasses
import pickle
import threading
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


def test_bare_and_called_forms_cache():
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
