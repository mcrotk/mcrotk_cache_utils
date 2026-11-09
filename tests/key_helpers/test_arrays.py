"""numpy arrays as arguments: keyed by shape, dtype and a hash of their data."""

import copy
import os
import pickle
import subprocess
import sys
import tempfile

import pytest

from mcrotk_cache_utils import array_key, cache_controls, cached_function, cached_method, rng_passthrough_cached_method

np = pytest.importorskip("numpy")


class Sim:
    def __init__(self):
        self.calls = 0

    @cached_method
    def total(self, arr):
        self.calls += 1
        return float(np.sum(arr))

    @cached_method
    def describe(self, arr):
        self.calls += 1
        return arr.shape, arr.dtype.str

    @cached_method
    def var(self, *arrays, **named):
        self.calls += 1
        return len(arrays), sorted(named)

    @cached_method(key_converters={"arr": lambda a: a.shape})
    def by_shape(self, arr):
        self.calls += 1
        return arr.sum()

    @cached_method(omit="arr")
    def ignores(self, arr, n):
        self.calls += 1
        return n

    @rng_passthrough_cached_method
    def draw(self, assignments, rng=None):
        self.calls += 1
        rng = np.random.default_rng(rng)
        return assignments * rng.normal(size=assignments.shape), rng

    @cached_method(on_copy="copy")
    def kept(self, arr):
        self.calls += 1
        return arr.sum()


def calls_for(method, *argument_lists):
    """Call ``method`` of a new Sim with each argument list in turn; return how many calls ran."""
    sim = Sim()
    for args in argument_lists:
        getattr(sim, method)(*args)
    return sim.calls


def test_equal_arrays_share_a_result():
    assert calls_for("total", [np.arange(5.0)], [np.arange(5.0)]) == 1


def test_changed_in_place_gets_a_new_result():
    sim, arr = Sim(), np.arange(5.0)
    assert sim.total(arr) == 10
    arr[0] = 100
    assert sim.total(arr) == 110
    assert sim.calls == 2


def test_same_bytes_different_shape():
    a = np.arange(6.0)
    assert calls_for("describe", [a.reshape(2, 3)], [a.reshape(3, 2)], [a]) == 3
    assert calls_for("describe", [np.zeros(4)], [np.zeros((2, 2))]) == 2


def test_same_bytes_different_dtype():
    a = np.arange(3, dtype="<i4")
    assert a.tobytes() == a.view("<f4").tobytes()
    assert calls_for("describe", [a], [a.view("<f4")]) == 2
    assert calls_for("describe", [np.zeros(4, dtype="i1")], [np.zeros(2, dtype="i2")]) == 2
    assert calls_for("describe", [np.zeros(2, dtype="<i4")], [np.zeros(2, dtype=">i4")]) == 2  # byte order


def test_memory_layout_doesnt_matter():
    # Equal values in different memory layouts share a result.
    c = np.arange(6.0).reshape(2, 3)
    fortran = np.asfortranarray(c)
    strided = (np.arange(12.0) / 2).reshape(2, 6)[:, ::2]  # a view with gaps between elements
    for a in (fortran, strided):
        np.testing.assert_array_equal(a, c)
        assert not a.flags.c_contiguous
    assert calls_for("total", [c], [fortran], [strided]) == 1
    reversed_view = np.arange(5.0, -1, -1)[::-1]  # negative stride
    assert not reversed_view.flags.c_contiguous
    assert calls_for("total", [np.arange(6.0)], [reversed_view]) == 1
    t = np.arange(6.0).reshape(3, 2).T
    assert calls_for("describe", [t], [np.ascontiguousarray(t)]) == 1


def test_zero_dimensional_and_empty_arrays():
    assert calls_for("describe", [np.array(3.0)], [np.array(3.0)], [np.array([3.0])]) == 2
    assert calls_for("describe", [np.zeros((0, 3))], [np.zeros((0, 3))], [np.zeros((3, 0))], [np.zeros(0)]) == 3


def test_nan_arrays_share_a_result():
    # By their bytes, so equal arrays with NaNs match (although NaN != NaN).
    assert calls_for("total", [np.array([np.nan, 1.0])], [np.array([np.nan, 1.0])]) == 1


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.array(["2020-01-01", "2021-06-30"], dtype="datetime64[D]"),
        lambda: np.array([1, 2], dtype="timedelta64[s]"),
        lambda: np.array(["ab", "c"]),
        lambda: np.array([b"ab", b"c"]),
        lambda: np.array([True, False]),
        lambda: np.array([1 + 2j, 3j]),
        lambda: np.array([1.5, 2.5], dtype="f2"),
        lambda: np.array([(1, 2.0), (3, 4.0)], dtype=[("a", "i4"), ("b", "f8")]),
        lambda: np.broadcast_to(np.arange(3.0), (4, 3)),
    ],
    ids=["datetime", "timedelta", "str", "bytes", "bool", "complex", "float16", "structured", "broadcast"],
)
def test_dtypes_and_kinds(make):
    assert calls_for("describe", [make()], [make()]) == 1


def test_structured_dtypes_differing_only_in_field_names():
    a = np.zeros(2, dtype=[("a", "i4")])
    b = np.zeros(2, dtype=[("b", "i4")])
    assert calls_for("describe", [a], [b]) == 2


def test_arrays_in_var_args_and_kwargs():
    sim = Sim()
    sim.var(np.arange(3), np.ones(2), x=np.zeros(2), y=5)
    sim.var(np.arange(3), np.ones(2), y=5, x=np.zeros(2))
    assert sim.calls == 1
    sim.var(np.arange(3), np.ones(2), x=np.ones(2), y=5)
    sim.var(np.arange(3), np.zeros(2), x=np.zeros(2), y=5)
    assert sim.calls == 3


def test_memmap_matches_equal_ndarray():
    path = os.path.join(tempfile.mkdtemp(), "data")
    m = np.memmap(path, dtype="f8", mode="w+", shape=(3,))
    m[:] = [0.0, 1.0, 2.0]
    assert calls_for("total", [m], [np.arange(3.0)]) == 1
    del m


def test_key_converter_replaces_array_handling():
    sim = Sim()
    assert sim.by_shape(np.arange(3.0)) == 3
    assert sim.by_shape(np.ones(3)) == 3  # same shape: the cached result
    assert sim.calls == 1


def test_omitted_array():
    assert calls_for("ignores", [np.arange(3), 1], [np.ones(9), 1]) == 1


def test_numpy_scalars_are_unchanged():
    assert calls_for("total", [np.float64(2.0)], [2.0]) == 1  # np.float64 is hashable and == 2.0


def test_with_rng_passthrough():
    sim, rng = Sim(), np.random.default_rng(0)
    first, rng = sim.draw(np.ones(3), rng=rng)
    again, rng2 = sim.draw(np.ones(3), rng=rng)
    assert sim.calls == 1 and again is first and rng2 is rng
    sim.draw(np.zeros(3), rng=rng)
    assert sim.calls == 2


def test_function():
    calls = []

    @cached_function
    def total(arr, *, scale=1):
        calls.append(1)
        return arr.sum() * scale

    total(np.arange(4))
    total(np.arange(4), scale=1)
    total(np.arange(4).reshape(2, 2))
    assert len(calls) == 2
    assert len(cache_controls(total).cache) == 2


@pytest.mark.parametrize(
    "arr",
    [np.array([1, None], dtype=object), np.zeros(2, dtype=[("a", "i4"), ("b", "O")])],
    ids=["object", "structured-with-object"],
)
def test_object_arrays_raise(arr):
    with pytest.raises(TypeError) as e:
        Sim().total(arr)
    msg = str(e.value)
    assert msg.startswith("Sim.total: can't use argument 'arr' in the cache key: arrays with object dtype")
    assert "key_converters={'arr': ...}" in msg and "omit=['arr']" in msg


def test_object_array_in_var_args_raises():
    with pytest.raises(TypeError, match="argument 'arrays'.*object dtype"):
        Sim().var(np.array([None]))


class Tagged(np.ndarray):
    pass


@pytest.mark.parametrize(
    "arr", [np.ma.masked_array([1.0, 2.0], mask=[True, False]), np.zeros(2).view(Tagged)], ids=["masked", "custom"]
)
def test_array_subclasses_are_unhashable(arr):
    with pytest.raises(TypeError, match=f"argument 'arr'.*{type(arr).__name__} is unhashable.*key_converters"):
        Sim().total(arr)


def test_pickled_instance_keeps_array_results():
    sim = Sim()
    sim.total(np.arange(5.0))
    loaded = pickle.loads(pickle.dumps(sim))
    loaded.total(np.arange(5.0))
    assert loaded.calls == 1


def test_copies_with_on_copy_copy_keep_array_results():
    sim = Sim()
    sim.kept(np.arange(5.0))
    for copied in (copy.copy(sim), copy.deepcopy(sim)):
        copied.kept(np.arange(5.0))
        assert copied.calls == 1


_CHILD = """
import pickle, sys
import numpy as np
from mcrotk_cache_utils import array_key
keys = pickle.loads(sys.stdin.buffer.read())
cache = dict.fromkeys(keys)
fresh = [array_key(np.arange(5.0)), array_key(np.zeros(2, dtype=[("name", "i4"), ("other", "f8")]))]
print(all(k in cache for k in fresh))
"""


def test_keys_match_in_another_process():
    # Pickled caches go to worker processes, so keys must not depend on per-process hashing.
    keys = [array_key(np.arange(5.0)), array_key(np.zeros(2, dtype=[("name", "i4"), ("other", "f8")]))]
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD],
            input=pickle.dumps(keys),
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        assert result.stdout.decode().strip() == "True"


def test_numpy_not_imported_by_the_library():
    code = (
        "import sys, mcrotk_cache_utils as m\n"
        "f = m.cached_function(lambda x, y: x)\n"
        "f(1, (2, 3)); f(1, y=(2, 3))\n"
        "print('numpy' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, check=True)
    assert result.stdout.decode().strip() == "False"
