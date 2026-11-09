"""dataframe_key: DataFrames as keys, by labels, dtypes and values, with or without the index."""

import os
import pickle
import subprocess
import sys

import pytest

from mcrotk_cache_utils import cached_method, dataframe_key

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


def frame(**columns):
    return pd.DataFrame(columns)


def same(a, b, **options):
    """Whether ``a`` and ``b`` get the same key (checking the keys are hashable)."""
    ka, kb = dataframe_key(a, **options), dataframe_key(b, **options)
    hash(ka)
    hash(kb)
    return ka == kb


def test_equal_frames_share_a_key():
    df = frame(x=[1.0, 2.0], n=[1, 2], s=["a", "b"])
    assert same(df, df.copy())
    assert same(df, frame(x=[1.0, 2.0], n=[1, 2], s=["a", "b"]))


@pytest.mark.parametrize(
    "a, b",
    [
        (frame(x=[1.0, 2.0]), frame(x=[1.0, 3.0])),
        (frame(x=[1.0, 2.0]), frame(y=[1.0, 2.0])),
        (frame(x=[1, 2]), frame(x=[1.0, 2.0])),
        (frame(a=[1], b=[2]), frame(b=[2], a=[1])),
        (frame(a=[1], b=[2]), frame(a=[2], b=[1])),
        (frame(x=[1.0, 2.0]), frame(x=[2.0, 1.0], index=[1, 0]).sort_index(ascending=False)),
        (frame(x=[1.0]), frame(x=[1.0, 1.0])),
        (frame(x=[1.0]), frame(x=[1.0], y=[1.0])),
    ],
    ids=["values", "labels", "dtypes", "column-order", "swapped-values", "row-order", "rows", "columns"],
)
def test_differences(a, b):
    assert not same(a, b)


def test_index_included_by_default():
    df = frame(x=[1.0, 2.0])
    relabeled = df.set_axis([10, 20])
    assert not same(df, relabeled)
    assert same(df, relabeled, index=False)
    assert not same(df, df.rename_axis("id"))  # index names
    assert same(df, df.rename_axis("id"), index=False)


def test_index_dtype_counts():
    # pandas' own hashes don't tell 0 from 0.0 in an index
    assert not same(frame(x=[1.0]).set_axis([0]), frame(x=[1.0]).set_axis([0.0]))


def test_multiindex():
    def make(names=("n", "s")):
        idx = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=list(names))
        return pd.DataFrame({"x": [1, 2]}, index=idx)

    assert same(make(), make())
    assert same(make(("n", "n")), make(("n", "n")))  # duplicate level names
    assert not same(make(), make(("n", "t")))
    swapped = make().reorder_levels([1, 0])
    assert not same(make(), swapped)
    assert same(make(), swapped, index=False)


@pytest.mark.parametrize(
    "a, b",
    [
        ([1, 2], ["1", 2]),
        ([None, "a"], [np.nan, "a"]),
        ([1, "a"], [1.0, "a"]),
        ([True, "a"], [1, "a"]),
    ],
    ids=["int-str", "none-nan", "int-float", "bool-int"],
)
def test_object_columns_keep_types(a, b):
    # pandas' own hashes would match each of these (it hashes mixed objects by str())
    assert not same(frame(x=pd.Series(a, dtype=object)), frame(x=pd.Series(b, dtype=object)))


def test_object_columns():
    def obj(values):
        return frame(x=pd.Series(values, dtype=object))

    assert same(obj(["a", "b"]), obj(["a", "b"]))  # string-only (pandas' hashes)
    assert not same(obj(["a", "b"]), obj(["a", "c"]))
    assert same(obj([1, "a", None]), obj([1, "a", None]))  # mixed (values kept)
    assert not same(obj([1, "a"]), obj([2, "a"]))
    assert same(obj([[1, 2], {"k": [3]}]), obj([[1, 2], {"k": [3]}]))  # containers, via container_key
    assert not same(obj([[1, 2]]), obj([(1, 2)]))
    assert same(obj([]), obj([]))


@pytest.mark.parametrize(
    "make",
    [
        lambda: frame(x=pd.Categorical(["a", "b"], categories=["a", "b", "z"])),
        lambda: frame(x=pd.array([1, None], dtype="Int64")),
        lambda: frame(x=pd.to_datetime(["2020-01-01", None]).tz_localize("UTC")),
        lambda: frame(x=pd.Series(["a", None], dtype="string")),
        lambda: frame(x=[True, False]),
        lambda: frame(x=pd.Series([], dtype=float)),
        lambda: pd.DataFrame([[1, 2]], columns=["a", "a"]),
        lambda: pd.DataFrame(index=[1, 2]),
    ],
    ids=["categorical", "nullable-int", "datetime-tz", "string", "bool", "empty", "duplicate-labels", "no-columns"],
)
def test_dtypes_and_shapes(make):
    assert same(make(), make())


def test_categories_count():
    a = frame(x=pd.Categorical(["a", "b"], categories=["a", "b"]))
    b = frame(x=pd.Categorical(["a", "b"], categories=["a", "b", "z"]))
    assert not same(a, b)


def test_not_a_dataframe():
    with pytest.raises(TypeError, match="needs a pandas DataFrame, got Series"):
        dataframe_key(pd.Series([1]))


class Model:
    def __init__(self):
        self.calls = 0

    @cached_method(key_converters={"df": dataframe_key})
    def doubled(self, df):
        self.calls += 1
        return df["x"] * 2

    @cached_method(key_converters={"df": lambda df: dataframe_key(df, index=False)})
    def doubled_ignoring_index(self, df):
        self.calls += 1
        return df["x"] * 2


def test_as_key_converter():
    m, df = Model(), frame(x=[1.0, 2.0])
    m.doubled(df)
    m.doubled(df.copy())
    assert m.calls == 1
    df.loc[0, "x"] = 5.0  # changed in place: a new key
    assert m.doubled(df).tolist() == [10.0, 4.0]
    assert m.calls == 2


def test_index_option_and_its_hazard():
    m, df = Model(), frame(x=[1.0, 2.0])
    relabeled = df.set_axis([10, 20])
    assert m.doubled(relabeled).index.tolist() == [10, 20]
    assert m.calls == 1
    m.doubled_ignoring_index(df)
    result = m.doubled_ignoring_index(relabeled)
    assert m.calls == 2  # a hit, as documented: the stored result keeps the first call's index
    assert result.index.tolist() == [0, 1]


def test_pickled_instance_keeps_results():
    m = Model()
    m.doubled(frame(x=[1.0, 2.0], s=pd.Series([1, "a"], dtype=object)))
    loaded = pickle.loads(pickle.dumps(m))
    loaded.doubled(frame(x=[1.0, 2.0], s=pd.Series([1, "a"], dtype=object)))
    assert loaded.calls == 1


_CHILD = """
import pickle, sys
import pandas as pd
from mcrotk_cache_utils import dataframe_key
from tests_frames import make
keys = pickle.loads(sys.stdin.buffer.read())
cache = dict.fromkeys(keys)
print(all(dataframe_key(df) in cache for df in make()))
"""

_FRAMES = """
import pandas as pd

def make():
    return [
        pd.DataFrame({"x": [1.0, 2.0], "s": ["a", "b"], "c": pd.Categorical(["u", "v"])}, index=["r", "q"]),
        pd.DataFrame({"o": pd.Series([1, "a", None, [2]], dtype=object)}),
    ]
"""


def test_keys_match_in_another_process(tmp_path):
    # Pickled caches go to worker processes, so keys must not depend on per-process hashing.
    (tmp_path / "tests_frames.py").write_text(_FRAMES)
    sys.path.insert(0, str(tmp_path))
    try:
        from tests_frames import make  # pyright: ignore[reportMissingImports]
    finally:
        sys.path.remove(str(tmp_path))
    keys = [dataframe_key(df) for df in make()]
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD],
            input=pickle.dumps(keys),
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(tmp_path)},
            check=True,
        )
        assert result.stdout.decode().strip() == "True"
