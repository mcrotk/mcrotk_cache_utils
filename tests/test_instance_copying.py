"""Copying instances with cached methods: on_copy, the installed __copy__/__deepcopy__, and their checks.

Covers the problems found along the way: copy.copy running the copy's methods on the original
and sharing its caches (also after only accessing a method, and through super()), stale results
in changed copies, cachetools caches whose copies share internals (LRU order, TTL expiry),
deepcopy copying results that are then thrown away, and classes with their own copy methods.
"""

import copy
import dataclasses
import pickle
import sys
import threading
import warnings
from typing import NamedTuple

import cachetools
import pytest

from mcrotk_cache_utils import CacheCopyWarning, cached_method, clear_caches, copy_instance, rng_cached_method

COPIERS = pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])


class Pair(NamedTuple):
    value: int
    token: object


@dataclasses.dataclass
class Base:
    scale: int = 2

    @cached_method
    def g(self, x):
        return ("Base.g", id(self), self.scale * x)


@dataclasses.dataclass
class Sim(Base):
    data: list = dataclasses.field(default_factory=list)

    @cached_method(info=True)
    def f(self, x):  # on_copy='empty' (the default)
        return id(self), self.scale * x

    @cached_method(on_copy="copy")
    def kept(self, x):
        return id(self), self.scale * x

    @cached_method(passthrough={"t": -1}, on_copy="copy")
    def p(self, x, t):
        return Pair(self.scale * x, t)

    @cached_method
    def g(self, x):  # an override calling the parent's cached method
        return ("Sim.g", id(self), super().g(x))


def used_sim():
    s = Sim()
    s.f(1)
    s.kept(1)
    s.p(1, "a")
    s.g(1)
    return s


def cache(obj, attr):
    return vars(obj).get(attr)


# --- Which instance the copy's methods run on ---


@COPIERS
def test_copy_before_any_use(copier):
    o = Sim()
    c = copier(o)
    c.scale = 10
    assert c.f(1) == (id(c), 10) and o.f(1) == (id(o), 2)
    assert cache(c, "_Sim_f_cache") is not cache(o, "_Sim_f_cache")


@COPIERS
def test_copys_methods_run_on_the_copy_after_use(copier):
    o = used_sim()
    c = copier(o)
    c.scale = 10
    assert c.f(7) == (id(c), 70)
    assert c.kept(7) == (id(c), 70)
    assert c.p(7, "b") == Pair(70, "b")
    assert c.g(7) == ("Sim.g", id(c), ("Base.g", id(c), 70))  # the super() path too


def test_copy_after_only_accessing_a_method():
    o = Sim()
    o.f.cache_info()  # creates cachetools' wrapper in o.__dict__ without calling f
    c = copy.copy(o)
    c.scale = 10
    assert c.f(7) == (id(c), 70)


def test_copy_of_a_copy():
    o = used_sim()
    c2 = copy.copy(copy.copy(o))
    c2.scale = 10
    assert c2.f(7) == (id(c2), 70) and c2.g(7)[2] == ("Base.g", id(c2), 70)


@COPIERS
def test_original_unaffected_by_using_the_copy(copier):
    o = used_sim()
    c = copier(o)
    c.scale = 10
    for call in (lambda: c.f(7), lambda: c.kept(7), lambda: c.p(7, "b"), lambda: c.g(7)):
        call()
    assert o.f(7) == (id(o), 14) and o.kept(7) == (id(o), 14) and o.p(7, "q") == Pair(14, "q")
    assert o.g(7) == ("Sim.g", id(o), ("Base.g", id(o), 14))
    assert (7,) in cache(c, "_Sim_kept_cache") and cache(o, "_Sim_kept_cache")[(7,)] == (id(o), 14)


# --- on_copy='empty' (default) and 'copy' ---


@COPIERS
def test_empty_default_gives_fresh_caches(copier):
    o = used_sim()
    c = copier(o)
    for attr in ("_Sim_f_cache", "_Sim_g_cache", "_Base_g_cache"):
        assert attr not in vars(c) and attr in vars(o)
    c.scale = 10
    assert c.f(1) == (id(c), 10)  # recomputed, not stale
    assert c == dataclasses.replace(o, scale=10)  # caches don't affect dataclass equality


@COPIERS
def test_copy_mode_keeps_results_in_an_independent_cache(copier):
    o = used_sim()
    c = copier(o)
    c.scale = 10
    assert c.kept(1) == (id(o), 2)  # kept, and stale since the copy changed (by design)
    assert cache(c, "_Sim_kept_cache") is not cache(o, "_Sim_kept_cache")
    c.kept(5)
    o.kept(6)
    assert (5,) not in cache(o, "_Sim_kept_cache") and (6,) not in cache(c, "_Sim_kept_cache")
    clear_caches(c)
    assert len(cache(o, "_Sim_kept_cache")) == 2  # clearing the copy leaves the original's


class Lists:
    @cached_method(on_copy="copy")
    def items(self, n):
        return list(range(n))


def test_copy_mode_shallow_shares_results_deep_duplicates_them():
    o = Lists()
    r = o.items(3)
    assert copy.copy(o).items(3) is r
    deep = copy.deepcopy(o).items(3)
    assert deep == r and deep is not r


@COPIERS
def test_passthrough_entries_kept(copier):
    o = used_sim()
    c = copier(o)
    hit = c.p(1, "new")
    assert hit == Pair(2, "new") and type(hit) is Pair and len(cache(c, "_Sim_p_cache")) == 1


def test_mixed_modes_in_one_class():
    o = used_sim()
    c = copy.copy(o)
    assert "_Sim_f_cache" not in vars(c) and len(cache(c, "_Sim_kept_cache")) == 1


@COPIERS
def test_info_counters_start_at_zero(copier):
    o = Sim()
    o.f(1)
    o.f(1)
    c = copier(o)
    assert (c.f.cache_info().hits, c.f.cache_info().misses) == (0, 0)
    assert o.f.cache_info().hits == 1


# --- Copying cachetools caches for on_copy='copy' ---


class LRU:
    def __init__(self):
        self.runs = []

    @cached_method(cache_factory=lambda: cachetools.LRUCache(maxsize=3), on_copy="copy")
    def m(self, x):
        self.runs.append(x)
        return x


def test_lru_cache_copy_keeps_order_and_is_independent():
    o = LRU()
    for x in (1, 2, 3, 1):  # least recently used is now 2
        o.m(x)
    c = copy.copy(o)
    c.m(4)  # evicts 2 from the copy only
    assert sorted(cache(c, "_LRU_m_cache")) == [(1,), (3,), (4,)]
    assert sorted(cache(o, "_LRU_m_cache")) == [(1,), (2,), (3,)]
    o.m(5)  # copying didn't change the original's order: 2 is still its least recently used
    assert sorted(cache(o, "_LRU_m_cache")) == [(1,), (3,), (5,)]


NOW = [0.0]


def clock():
    return NOW[0]


def test_ttl_cache_copy_keeps_expiry_times():
    class TTL:
        def __init__(self):
            self.runs = []

        @cached_method(cache_factory=lambda: cachetools.TTLCache(maxsize=10, ttl=10, timer=clock), on_copy="copy")
        def m(self, x):
            self.runs.append(x)
            return x

    NOW[0] = 0
    o = TTL()
    o.m(1)  # expires at 10
    NOW[0] = 5
    c = copy.copy(o)
    c.m(1)
    assert c.runs == [1]  # a hit in the copy
    NOW[0] = 11
    c.m(1)
    assert c.runs == [1, 1]  # expired at the original time, not 10 after copying


def test_custom_getsizeof_cache_copy():
    class Sized:
        @cached_method(cache_factory=lambda: cachetools.LRUCache(maxsize=100, getsizeof=len), on_copy="copy")
        def m(self, n):
            return "x" * n

    o = Sized()
    o.m(3)
    o.m(4)
    c = copy.copy(o)
    assert cache(c, "_Sized_m_cache").currsize == 7 and c.m(3) == "xxx"


# --- deepcopy specifics ---


class Counted:
    deep_copies = 0

    def __deepcopy__(self, memo):
        Counted.deep_copies += 1
        return Counted()


class Results:
    @cached_method
    def empty(self):
        return Counted()

    @cached_method(on_copy="copy")
    def kept(self):
        return Counted()


def test_deepcopy_doesnt_copy_results_it_throws_away():
    o = Results()
    o.empty()
    Counted.deep_copies = 0
    copy.deepcopy(o)
    assert Counted.deep_copies == 0
    o.kept()
    copy.deepcopy(o)
    assert Counted.deep_copies == 1


def test_deepcopy_keeps_shared_and_self_references():
    o = used_sim()
    o.alias = o.data  # pyright: ignore[reportAttributeAccessIssue]
    o.me = o  # pyright: ignore[reportAttributeAccessIssue]
    c = copy.deepcopy(o)
    assert c.alias is c.data and c.data is not o.data and c.me is c  # pyright: ignore[reportAttributeAccessIssue]
    pair = copy.deepcopy([o, o])
    assert pair[0] is pair[1] is not o


# --- Other ways of copying, unchanged ---


def test_pickle_carries_all_caches():
    o = used_sim()
    p = pickle.loads(pickle.dumps(o))
    assert p.f(1) == (id(o), 2)  # carried along, even for on_copy='empty'
    assert p.f(7) == (id(p), 14)  # new calls run on the unpickled instance


@pytest.mark.parametrize(
    "replace",
    [
        dataclasses.replace,
        pytest.param(
            getattr(copy, "replace", None),
            marks=pytest.mark.skipif(sys.version_info < (3, 13), reason="copy.replace is new in 3.13"),
        ),
    ],
    ids=["dataclasses.replace", "copy.replace"],
)
def test_replace_starts_with_empty_caches(replace):
    o = used_sim()
    c = replace(o, scale=10)
    assert c.f(1) == (id(c), 10) and c.kept(1) == (id(c), 10)


@COPIERS
def test_frozen_dataclass(copier):
    @dataclasses.dataclass(frozen=True)
    class Frozen:
        n: int

        @cached_method(on_copy="copy")
        def double(self):
            return [2 * self.n]

    o = Frozen(1)
    r = o.double()
    c = copier(o)
    assert c == o and c.double() == r and (c.double() is r) == (copier is copy.copy)


# --- Classes with custom state, slots, inheritance ---


def test_custom_getstate_is_honored():
    class C:
        def __init__(self):
            self.keep, self.scratch = 1, 2

        def __getstate__(self):
            return {k: v for k, v in vars(self).items() if k != "scratch"}

        @cached_method
        def f(self, x):
            return id(self), x

    o = C()
    o.f(1)
    c = copy.copy(o)
    assert not hasattr(c, "scratch") and c.keep == 1
    assert "_C_f_cache" not in vars(c) and c.f(1) == (id(c), 1)


def test_custom_state_format_is_fixed_after_copying():
    class C:
        def __getstate__(self):
            return ("v1", dict(vars(self)))  # not a dict: handled after Python's copy

        def __setstate__(self, state):
            vars(self).update(state[1])

        @cached_method
        def f(self, x):
            return id(self), x

        @cached_method(on_copy="copy")
        def kept(self, x):
            return id(self), x

    o = C()
    o.f(1)
    o.kept(1)
    for copier in (copy.copy, copy.deepcopy):
        c = copier(o)
        assert "_C_f_cache" not in vars(c) and c.f(1) == (id(c), 1)
        assert c.kept(1) == (id(o), 1) and cache(c, "_C_kept_cache") is not cache(o, "_C_kept_cache")


def test_slots_with_dict():
    class C:
        __slots__ = ("a", "__dict__")

        def __init__(self):
            self.a = 1

        @cached_method
        def f(self, x):
            return id(self), self.a * x

    o = C()
    o.f(1)
    c = copy.copy(o)
    c.a = 10
    assert c.f(1) == (id(c), 10) and o.f(1) == (id(o), 1)


def test_subclass_inherits_copy_methods_and_handles_its_own_caches():
    @dataclasses.dataclass
    class Child(Sim):
        @cached_method
        def h(self, x):
            return id(self), x

    assert "__copy__" in vars(Base) and "__copy__" not in vars(Sim) and "__copy__" not in vars(Child)  # inherited
    o = Child()
    o.h(1)
    o.f(1)
    c = copy.copy(o)
    assert c.h(1) == (id(c), 1) and "_Sim_f_cache" not in vars(c)


def test_classes_without_cached_methods_are_left_alone():
    class Plain:
        pass

    assert not hasattr(Plain, "__copy__")


def test_lock_function_attribute_is_shared_by_shallow_copy():
    class C:
        def __init__(self):
            self.lock = threading.Lock()

        @cached_method(lock=lambda self: self.lock, on_copy="copy")
        def f(self, x):
            return id(self), x

    o = C()
    o.f(1)
    c = copy.copy(o)
    assert c.lock is o.lock and c.f(1) == (id(o), 1) and c.f(2) == (id(c), 2)


def test_rng_cached_method_on_copy():
    np = pytest.importorskip("numpy")

    class S:
        @rng_cached_method(on_copy="copy")
        def draw(self, n, rng=None):
            rng = np.random.default_rng(rng)
            return rng.random(n).sum(), rng

    o = S()
    v, _ = o.draw(3, rng=1)
    g = np.random.default_rng(9)
    v2, r = copy.copy(o).draw(3, rng=g)
    assert v2 == v and r is g


# --- Classes with their own __copy__/__deepcopy__ ---


def error_text(e: BaseException) -> str:
    # Python 3.11 wraps errors raised in __set_name__ in a RuntimeError, with ours as the cause.
    return f"{e} {e.__cause__ or ''}"


def test_own_copy_warns_once_per_class_and_keeps_theirs():
    with pytest.warns(CacheCopyWarning) as record:

        class C:
            def __copy__(self):
                return "theirs"

            @cached_method
            def f(self, x):
                return x

            @cached_method
            def g(self, x):
                return x

    assert len(record) == 1 and record[0].filename == __file__  # points at this class definition
    assert "C defines its own __copy__" in str(record[0].message)
    assert copy.copy(C()) == "theirs"
    o = C()
    o.f(1)
    assert "_C_f_cache" not in vars(copy.deepcopy(o))  # our __deepcopy__ is still installed


def test_own_deepcopy_only_warns_and_copy_copy_is_still_handled():
    with pytest.warns(CacheCopyWarning, match="C defines its own __deepcopy__"):

        class C:
            def __deepcopy__(self, memo):
                return "theirs"

            @cached_method
            def f(self, x):
                return id(self), x

    o = C()
    o.f(1)
    c = copy.copy(o)
    assert c.f(1) == (id(c), 1) and copy.deepcopy(o) == "theirs"


def test_warning_can_be_filtered_by_category():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=CacheCopyWarning)

        class C:
            def __copy__(self):
                return self

            @cached_method
            def f(self, x):
                return x


@pytest.mark.parametrize("on_copy", ["empty", "copy"])
@pytest.mark.parametrize("dunder", ["__copy__", "__deepcopy__"])
def test_explicit_on_copy_with_own_copy_method_is_an_error(on_copy, dunder):
    with pytest.raises((TypeError, RuntimeError)) as e:
        type("C", (), {dunder: lambda self, *memo: self, "f": cached_method(on_copy=on_copy)(lambda self, x: x)})
    assert f"on_copy={on_copy!r} can't take effect" in error_text(e.value)


def test_manual_with_own_copy_no_warning_nothing_installed():
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class C:
            def __copy__(self):
                return "theirs"

            @cached_method(on_copy="manual")
            def f(self, x):
                return x

    assert copy.copy(C()) == "theirs" and not hasattr(C, "__deepcopy__")
    C().f(1)  # no warning at first use either


def test_manual_satisfied_by_a_base_class():
    class CopyMixin:
        def __copy__(self):
            return "mixin"

    with warnings.catch_warnings():
        warnings.simplefilter("error")

        class C(CopyMixin):
            @cached_method(on_copy="manual")
            def f(self, x):
                return x

    assert copy.copy(C()) == "mixin"


def test_manual_without_own_copy_is_an_error():
    with pytest.raises((TypeError, RuntimeError)) as e:

        class C:
            @cached_method(on_copy="manual")
            def f(self, x):
                return x

    text = error_text(e.value)
    assert "C.f: on_copy='manual', but " in text and "C has no __copy__ of its own" in text


def test_manual_when_only_our_copy_is_inherited_is_an_error():
    class Parent:
        @cached_method
        def f(self, x):
            return x

    with pytest.raises((TypeError, RuntimeError)) as e:

        class Child(Parent):
            @cached_method(on_copy="manual")
            def g(self, x):
                return x

    assert "has no __copy__ of its own" in error_text(e.value)


def test_subclass_adding_own_copy_warns_at_first_use_once():
    class Parent:
        @cached_method
        def f(self, x):
            return x

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # nothing at class creation (no hook for the subclass)

        class Child(Parent):
            def __copy__(self):
                return self

    Parent().f(1)  # no warning for the parent
    with pytest.warns(CacheCopyWarning, match="Child defines its own __copy__") as record:
        Child().f(1)
    assert record[0].filename == __file__  # points at the call
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Child().f(2)  # once per class


def test_invalid_on_copy_value():
    with pytest.raises(ValueError, match="on_copy must be one of"):
        cached_method(on_copy="deep")  # pyright: ignore[reportArgumentType]


# --- copy_instance ---

DEPTHS = pytest.mark.parametrize("deep", [False, True], ids=["shallow", "deep"])


@DEPTHS
def test_copy_instance_follows_on_copy_by_default(deep):
    o = used_sim()
    c = copy_instance(o, deep=deep)
    assert "_Sim_f_cache" not in vars(c) and len(cache(c, "_Sim_kept_cache")) == 1
    assert (c.data is o.data) != deep


@DEPTHS
@pytest.mark.parametrize("caches", ["empty", "copy"])
def test_copy_instance_overrides_every_method(deep, caches):
    o = used_sim()
    c = copy_instance(o, caches, deep=deep)
    for attr in ("_Sim_f_cache", "_Sim_kept_cache", "_Sim_p_cache", "_Sim_g_cache", "_Base_g_cache"):
        if caches == "empty":
            assert attr not in vars(c)
        else:
            assert len(cache(c, attr)) == 1 and cache(c, attr) is not cache(o, attr)
    c.scale = 10
    assert c.f(7) == (id(c), 70) and c.g(7) == ("Sim.g", id(c), ("Base.g", id(c), 70))  # runs on the copy


def test_copy_instance_copy_shares_results_deep_duplicates_them():
    o = Lists()
    r = o.items(3)
    assert copy_instance(o, "copy").items(3) is r
    deep = copy_instance(o, "copy", deep=True).items(3)
    assert deep == r and deep is not r


class OwnCopy:
    """A class with its own copy methods, using copy_instance inside them."""

    def __init__(self):
        self.scratch = "work in progress"
        self.me = self
        self.items = []

    @cached_method(on_copy="manual")
    def f(self, x):
        return id(self), x

    def __copy__(self):
        new = copy_instance(self)
        new.scratch = None
        return new

    def __deepcopy__(self, memo):
        new = copy_instance(self, "copy", memo=memo)
        new.scratch = None
        return new


def test_copy_instance_inside_own_copy_methods():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 'manual' with own copy methods: no warning at creation or first use
        o = OwnCopy()
        o.f(1)
    c = copy.copy(o)
    assert c.scratch is None and c.me is o  # their __copy__ ran (shallow: me still refers to the original)
    assert "_OwnCopy_f_cache" not in vars(c) and c.f(1) == (id(c), 1)  # 'manual' emptied by default
    d = copy.deepcopy(o)
    assert d.scratch is None and d.me is d  # memo passed through: the self-reference points at the copy
    assert d.f(1) == (id(o), 1) and d.f(2) == (id(d), 2)  # cache copied, new calls on the copy
    pair = copy.deepcopy([o, o])
    assert pair[0] is pair[1]
    copied, items = copy.deepcopy([o, o.items])  # an attribute shared with another part of what's copied
    assert isinstance(copied, OwnCopy) and copied.items is items and items is not o.items


def test_copy_instance_without_cached_methods():
    @dataclasses.dataclass
    class Plain:
        items: list

    o = Plain([1])
    assert copy_instance(o) == o and copy_instance(o).items is o.items
    assert copy_instance(o, deep=True).items is not o.items


def test_copy_instance_bad_caches_value():
    with pytest.raises(ValueError, match="caches must be"):
        copy_instance(Sim(), "manual")  # pyright: ignore[reportArgumentType]
