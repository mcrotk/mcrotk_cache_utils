"""cached_function: options, caches, static and class methods, threading, pickling.

Most tests decorate a fresh local function, since a function's cache lasts as long as the function.
"""

import pickle
import threading
import time
from typing import NamedTuple

import cachetools
import pytest

from mcrotk_cache_utils import cached_function


def test_repeated_call_runs_once():
    runs = []

    @cached_function
    def f(x, y=2):
        runs.append(x)
        return x * y

    assert f(3) == f(3) == 6
    assert runs == [3]


def test_bound_arguments_give_one_key():
    runs = []

    @cached_function
    def f(x, y=2):
        runs.append((x, y))
        return x * y

    f(1)
    f(1, 2)
    f(x=1)
    f(y=2, x=1)
    assert runs == [(1, 2)]
    assert f.cache == {(1, 2): 2}


def test_omit_by_name_and_position_zero():
    runs = []

    @cached_function(omit=["verbose", 0])  # position 0 is the first parameter: there's no self
    def f(label, x, verbose=False):
        runs.append(label)
        return x

    f("a", 1)
    f("b", 1, verbose=True)
    assert runs == ["a"] and f.cache == {(1,): 1}


def test_negative_position_and_out_of_range_message():
    @cached_function(omit=-1)
    def f(x, verbose=False):
        return x

    assert f.cache == {} and f(1, True) == 1 and f.cache == {(1,): 1}
    with pytest.raises(ValueError, match=r"out of range for 2 parameters$"):  # no "positions count self"
        cached_function(omit=2)(lambda x, y: x)


def test_key_converter_and_var_keyword():
    runs = []

    @cached_function(key_converters={"xs": tuple})
    def total(xs, **options):
        runs.append(1)
        return sum(xs), options

    total([1, 2], b=1, a=2)
    total([1, 2], a=2, b=1)
    assert len(runs) == 1


def test_function_with_no_parameters():
    runs = []

    @cached_function
    def f():
        runs.append(1)
        return 5

    assert f() == f() == 5 and runs == [1]


def test_called_form_without_options():
    @cached_function()
    def f(x):
        return x

    assert f(1) == 1 and f.cache == {(1,): 1}


def test_functools_attributes_kept():
    @cached_function
    def area(width, height):
        """Docstring."""
        return width * height

    assert area.__name__ == "area" and area.__doc__ == "Docstring." and area.__wrapped__(2, 3) == 6


def test_cache_factory_info_and_clear():
    @cached_function(cache_factory=lambda: cachetools.LRUCache(maxsize=2), info=True)
    def f(x):
        return x

    for x in (1, 2, 3, 3):
        f(x)
    info = f.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 3, 2, 2)
    assert isinstance(f.cache, cachetools.LRUCache)
    f.cache_clear()
    info = f.cache_info()
    assert (info.hits, info.misses, info.currsize) == (0, 0, 0)


def test_default_cache_is_a_new_unbounded_dict_per_function():
    @cached_function
    def f(x):
        return x

    @cached_function
    def g(x):
        return x

    for x in range(1000):
        f(x)
    assert type(f.cache) is dict and len(f.cache) == 1000
    assert g.cache == {}


class Pair(NamedTuple):
    value: int
    token: object


def test_passthrough_with_transform_and_named_tuple():
    runs = []

    @cached_function(passthrough={"token": (-1, str.upper)})
    def f(x, token):
        runs.append(token)
        return Pair(x, token)

    assert f(1, "a") == Pair(1, "a")  # miss: the function's own outputs
    hit = f(1, "b")
    assert hit == Pair(1, "B") and type(hit) is Pair  # hit: the caller's token, transformed
    assert runs == ["a"]


def test_passthrough_first_argument_by_position():
    @cached_function(passthrough={0: 1})  # the first parameter goes to output position 1
    def f(token, x):
        return x, token

    assert f("a", 1) == (1, "a")
    assert f("b", 1) == (1, "b")
    assert list(f.cache) == [(1,)]


def test_passthrough_info_through_view():
    @cached_function(passthrough={"t": -1}, info=True, cache_factory=lambda: cachetools.LRUCache(maxsize=2))
    def f(x, t):
        return x, t

    for x in (1, 2, 3):
        f(x, "a")
    assert f(3, "b") == (3, "b")
    info = f.cache_info()
    assert (info.hits, info.misses, info.maxsize, info.currsize) == (1, 3, 2, 2)


class Model:
    runs: list = []

    @staticmethod
    @cached_function
    def kernel(n):
        Model.runs.append(("kernel", n))
        return n + 1

    @classmethod
    @cached_function(info=True)
    def defaults(cls, n):
        Model.runs.append((cls.__name__, n))
        return cls.__name__, n

    @classmethod
    @cached_function(omit="cls")
    def shared(cls, n):
        Model.runs.append(("shared", cls.__name__, n))
        return n


class SubModel(Model):
    pass


@pytest.fixture
def model():
    Model.runs.clear()
    for f in (Model.kernel, Model.defaults, Model.shared):
        f.cache_clear()
    return Model


def test_static_method(model):
    assert model.kernel(1) == model().kernel(1) == SubModel.kernel(1) == 2
    assert model.runs == [("kernel", 1)]


def test_class_method_keys_on_cls(model):
    assert model.defaults(1) == ("Model", 1)
    assert SubModel.defaults(1) == ("SubModel", 1)  # its own entry in the shared cache
    assert model().defaults(1) == ("Model", 1)  # hit
    assert model.runs == [("Model", 1), ("SubModel", 1)]
    info = model.defaults.cache_info()  # cache attributes reachable through the bound method
    assert (info.hits, info.misses, info.currsize) == (1, 2, 2)


def test_class_method_omitting_cls_shares_results(model):
    model.shared(1)
    SubModel.shared(1)
    assert model.runs == [("shared", "Model", 1)]


@cached_function
def module_level(x):
    return x


@pytest.mark.parametrize(
    "get",
    [lambda: module_level, lambda: Model.kernel, lambda: Model.defaults],
    ids=["function", "staticmethod", "classmethod"],
)
def test_pickled_by_reference(get, model):
    f = get()
    f(1)
    copied = pickle.loads(pickle.dumps(f))
    assert copied == f and copied(1) == f(1)  # the same function, with its cache (no copy made)


def test_lock_and_condition_objects():
    lock, cond = threading.Lock(), threading.Condition()
    runs = []

    @cached_function(lock=lock, cache_factory=lambda: cachetools.LRUCache(maxsize=10))
    def locked(x):
        runs.append(x)
        return x

    @cached_function(condition=cond)
    def conditioned(x):
        return x

    threads = [threading.Thread(target=lambda i=i: (locked(i % 3), conditioned(i % 3))) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert sorted(locked.cache.items()) == [((0,), 0), ((1,), 1), ((2,), 2)]
    assert locked.cache_lock is lock and conditioned.cache_condition is cond


def test_condition_prevents_duplicate_computation():
    entered, release = threading.Event(), threading.Event()
    runs = []

    @cached_function(condition=threading.Condition())
    def slow(x):
        runs.append(x)
        entered.set()
        release.wait(5)
        return x

    results = []
    first = threading.Thread(target=lambda: results.append(slow(1)))
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=lambda: results.append(slow(1)))
    second.start()
    time.sleep(0.05)  # give the second thread time to reach the condition
    assert runs == [1]
    release.set()
    first.join(5)
    second.join(5)
    assert results == [1, 1] and runs == [1]


def test_lock_race_each_caller_keeps_its_own_passthrough_outputs():
    barrier = threading.Barrier(2, timeout=5)

    @cached_function(lock=threading.Lock(), passthrough={"token": -1})
    def f(x, token):
        barrier.wait()  # both threads have missed before either stores
        return x, token + "-out"

    results = {}
    threads = [threading.Thread(target=lambda t=t: results.__setitem__(t, f(1, t))) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert results == {"a": (1, "a-out"), "b": (1, "b-out")}
    assert f(1, "c") == (1, "c")


def test_caveat_returned_objects_are_the_cached_objects():
    @cached_function
    def items(n):
        return list(range(n))

    items(3).append("changed")
    assert items(3) == [0, 1, 2, "changed"]
