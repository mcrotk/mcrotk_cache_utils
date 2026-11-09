"""container_key: lists, tuples, dicts and sets as hashable keys, with or without type tags."""

import pickle
from collections import OrderedDict, namedtuple
from functools import partial

import pytest

from mcrotk_cache_utils import array_key, cache_controls, cached_function, cached_method, container_key

Point = namedtuple("Point", "x y")


def same(a, b, **options):
    """Whether ``a`` and ``b`` get the same key (checking the keys are hashable)."""
    ka, kb = container_key(a, **options), container_key(b, **options)
    hash(ka)
    hash(kb)
    return ka == kb


@pytest.mark.parametrize(
    "a, b",
    [
        ([2, 3], (2, 3)),
        (Point(2, 3), (2, 3)),
        ({1, 2}, frozenset({1, 2})),
        ({"a": 1}, OrderedDict(a=1)),
        ({1: 2}, {(1, 2)}),
        ([[1], {2: [3]}], ((1,), {2: (3,)})),
    ],
    ids=["list-tuple", "namedtuple-tuple", "set-frozenset", "dict-ordereddict", "dict-set-of-pairs", "nested"],
)
def test_type_tags_separate_types(a, b):
    assert not same(a, b)
    assert same(a, b, type_tags=False)


def test_type_tags_off_gives_plain_builtins():
    assert container_key([2, [3, 4]], type_tags=False) == (2, (3, 4))
    assert container_key({1, 2}, type_tags=False) == frozenset({1, 2})
    assert container_key({"a": [1]}, type_tags=False) == frozenset({("a", (1,))})


@pytest.mark.parametrize("type_tags", [True, False])
def test_equal_contents_share_a_key(type_tags):
    assert same([1, [2, {"a": {3}}]], [1, [2, {"a": {3}}]], type_tags=type_tags)
    assert same({"a": 1, "b": 2}, {"b": 2, "a": 1}, type_tags=type_tags)  # dict order doesn't matter, as for ==
    assert same({3, 1, 2}, {1, 2, 3}, type_tags=type_tags)
    assert not same([1, 2], [2, 1], type_tags=type_tags)  # sequence order does
    assert not same({"a": 1}, {"a": 2}, type_tags=type_tags)
    assert not same([1, [2]], [1, [3]], type_tags=type_tags)


def test_dict_keys_are_converted_too():
    assert same({(1, 2): "a"}, {(1, 2): "a"})
    assert not same({(1, 2): "a"}, {Point(1, 2): "a"})


@pytest.mark.parametrize("value", [1, "a", None, 2.5, b"x"])
def test_other_values_unchanged(value):
    assert container_key(value) is value


def test_numpy_scalars_unchanged():
    np = pytest.importorskip("numpy")
    value = np.float64(1)
    assert container_key(value) is value


def test_arrays_inside_are_keyed_by_content():
    np = pytest.importorskip("numpy")
    a = np.arange(3)
    assert container_key(a) == array_key(a)
    before = container_key({"w": [a, 1]})
    assert before == container_key({"w": [np.arange(3), 1]})
    a[0] = 99
    assert container_key({"w": [a, 1]}) != before


def test_object_array_inside_raises():
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError, match="object dtype"):
        container_key([np.array([None])])


def test_unhashable_leaf_left_for_the_key_check():
    class Unhashable:
        __hash__ = None  # pyright: ignore[reportAssignmentType]

    @cached_function(key_converters={"x": container_key})
    def f(x):
        return 1

    with pytest.raises(TypeError, match="converted value of argument 'x'"):
        f([Unhashable()])


def test_keys_pickle():
    for value in ([1, {"a": {2}}], Point(1, [2]), {1: frozenset({(2, 3)})}):
        key = container_key(value)
        loaded = pickle.loads(pickle.dumps(key))
        assert loaded == key and hash(loaded) == hash(key)


class Model:
    def __init__(self):
        self.calls = 0

    @cached_method(key_converters={"weights": container_key})
    def fit(self, weights):
        self.calls += 1
        return sum(weights)

    @cached_method(key_converters={"weights": partial(container_key, type_tags=False)})
    def fit_any(self, weights):
        self.calls += 1
        return sum(weights)


def test_as_key_converter():
    m = Model()
    weights = [1, 2]
    m.fit(weights)
    m.fit([1, 2])
    assert m.calls == 1
    weights.append(3)  # changed in place: a new key, not the stale result
    assert m.fit(weights) == 6
    m.fit((1, 2))
    assert m.calls == 3


def test_as_key_converter_without_type_tags():
    m = Model()
    m.fit_any([1, 2])
    m.fit_any((1, 2))
    assert m.calls == 1


# --- Applied automatically (as container_key with type_tags=False) ---


class Auto:
    def __init__(self):
        self.calls = 0

    @cached_method
    def total(self, xs):
        self.calls += 1
        return len(xs)

    @cached_method
    def var(self, *args, **kwargs):
        self.calls += 1
        return len(args)

    @cached_method(omit="xs")
    def ignores(self, xs, n):
        self.calls += 1
        return n

    @cached_method(passthrough={"t": -1})
    def through(self, xs, t):
        self.calls += 1
        return len(xs), t

    @cached_method(on_copy="copy")
    def kept(self, xs):
        self.calls += 1
        return len(xs)


def calls_for(method, *argument_lists, **kwargs_each):
    """Call ``method`` of a new Auto with each argument list in turn; return how many calls ran."""
    obj = Auto()
    for args in argument_lists:
        getattr(obj, method)(*args, **kwargs_each)
    return obj.calls


def test_equal_lists_share_a_result():
    assert calls_for("total", [[1, 2]], [[1, 2]]) == 1
    assert calls_for("total", [[1, 2]], [[2, 1]]) == 2


@pytest.mark.parametrize(
    "a, b",
    [
        ([1, 2], (1, 2)),
        (Point(1, 2), [1, 2]),
        ({1, 2}, frozenset({1, 2})),
        ({"a": [1], "b": 2}, {"b": 2, "a": [1]}),
        ({"a": [1]}, OrderedDict(a=(1,))),
        (([1], {2}), [(1,), frozenset({2})]),
    ],
    ids=["list-tuple", "namedtuple-list", "set-frozenset", "dict-order", "dict-ordereddict", "nested"],
)
def test_same_kind_shares_a_result(a, b):
    assert calls_for("total", [a], [b]) == 1


def test_changed_in_place_gets_a_new_result():
    obj, xs = Auto(), [1, 2]
    assert obj.total(xs) == 2
    xs.append(3)
    assert obj.total(xs) == 3
    d = {"a": [1]}
    obj.total(d)
    d["a"].append(2)
    obj.total(d)
    assert obj.calls == 4


def test_in_var_args_and_kwargs():
    obj = Auto()
    obj.var([1], {"a": {2}}, k=[3], j={4: [5]})
    obj.var((1,), {"a": frozenset({2})}, j={4: (5,)}, k=(3,))
    assert obj.calls == 1
    obj.var([1], {"a": {2}}, k=[3, 0], j={4: [5]})
    assert obj.calls == 2


def test_function():
    calls = []

    @cached_function
    def f(xs, *, opts=None):
        calls.append(1)
        return xs

    f([1, 2], opts={"a": 1})
    f((1, 2), opts={"a": 1})
    assert len(calls) == 1
    assert f((1, 2), opts={"a": 1}) == [1, 2]  # the stored result: [1, 2] and (1, 2) share an entry
    f([1, 2], opts={"a": 2})
    assert len(calls) == 2


def test_key_converter_overrides():
    m = Model()
    m.fit([1, 2])
    m.fit((1, 2))
    assert m.calls == 2  # container_key with type tags, given in key_converters


def test_omitted_and_passed_through():
    assert calls_for("ignores", [[1], 5], [{2}, 5]) == 1
    obj = Auto()
    obj.through([1, 2], "a")
    assert obj.through([1, 2], "b") == (2, "b")
    assert obj.calls == 1


def test_unhashable_inside_names_the_argument():
    class Unhashable:
        __hash__ = None  # pyright: ignore[reportAssignmentType]

    for value in ([Unhashable()], {"a": Unhashable()}, {Unhashable: [Unhashable()]}, ([Unhashable()],)):
        with pytest.raises(TypeError) as e:
            Auto().total(value)
        msg = str(e.value)
        assert msg.startswith("Auto.total: can't use argument 'xs' in the cache key: ")
        assert "unhashable type: 'Unhashable'" in msg  # within Python's own message, which varies by version
        assert "key_converters={'xs': ...}" in msg


def test_object_array_inside_names_the_argument():
    np = pytest.importorskip("numpy")
    with pytest.raises(TypeError, match="argument 'xs' in the cache key: arrays with object dtype"):
        Auto().total([np.array([None])])


def test_arrays_inside():
    np = pytest.importorskip("numpy")
    assert calls_for("total", [[np.arange(3), {"w": np.ones(2)}]], [(np.arange(3), {"w": np.ones(2)})]) == 1
    assert calls_for("total", [[np.arange(3)]], [[np.arange(4)]]) == 2


def test_hashable_tuples_unchanged():
    # Hashable containers go in the key as they are (converting would give an equal key anyway).
    calls = []

    @cached_function
    def f(t):
        calls.append(1)
        return t

    t = (1, (2, "a"))
    f(t)
    f([1, [2, "a"]])
    assert len(calls) == 1
    [key] = cache_controls(f).cache
    assert key[0] is t


def test_pickled_and_copied_instances_keep_results():
    obj = Auto()
    obj.kept([1, {"a": {2}}])
    for other in (pickle.loads(pickle.dumps(obj)), __import__("copy").copy(obj), __import__("copy").deepcopy(obj)):
        other.kept([1, {"a": {2}}])
        assert other.calls == 1


_CHILD = """
import pickle, sys
from mcrotk_cache_utils import container_key
keys = pickle.loads(sys.stdin.buffer.read())
cache = dict.fromkeys(keys)
fresh = [container_key(v, type_tags=False) for v in ({"a": ["x", {"y"}]}, ["s", ("t", {"u": 1})])]
print(all(k in cache for k in fresh))
"""


def test_keys_match_in_another_process():
    # Pickled caches go to worker processes; str hashes differ between processes, so the keys
    # must still be found once rebuilt there.
    import os
    import subprocess
    import sys

    keys = [container_key(v, type_tags=False) for v in ({"a": ["x", {"y"}]}, ["s", ("t", {"u": 1})])]
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD],
            input=pickle.dumps(keys),
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        assert result.stdout.decode().strip() == "True"
