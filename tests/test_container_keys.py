"""container_key: lists, tuples, dicts and sets as hashable keys, with or without type tags."""

import pickle
from collections import OrderedDict, namedtuple
from functools import partial

import numpy as np
import pytest

from mcrotk_cache_utils import array_key, cached_function, cached_method, container_key

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


@pytest.mark.parametrize("value", [1, "a", None, 2.5, b"x", np.float64(1)])
def test_other_values_unchanged(value):
    assert container_key(value) is value


def test_arrays_inside_are_keyed_by_content():
    a = np.arange(3)
    assert container_key(a) == array_key(a)
    before = container_key({"w": [a, 1]})
    assert before == container_key({"w": [np.arange(3), 1]})
    a[0] = 99
    assert container_key({"w": [a, 1]}) != before


def test_object_array_inside_raises():
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
    for value in ([1, {"a": {2}}], Point(1, [2]), {1: np.arange(2)}):
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
