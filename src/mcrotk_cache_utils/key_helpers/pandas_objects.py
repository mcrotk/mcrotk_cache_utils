"""Cache keys for pandas DataFrames, Series and Index objects, from their contents.

Use these only when there's no better choice. Keying a pandas object by its contents reads all
of it on every call (slowly for object columns), and equality between pandas objects has many
corners (dtypes, index labels and names, order, exact floats), each of which these keys must
get right. An argument that identifies the data (a path, an ID, the parameters it was made
from) is usually a better key. They are never applied automatically.

pandas is imported only when a key is made.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

from .containers import container_key


@dataclass(frozen=True)
class IndexKey:
    """Stands in for a pandas Index (or MultiIndex) in a cache key."""

    multi: bool  # a MultiIndex (other Index types differ by dtype, and RangeIndex matches Index of int64)
    names: tuple  # level names
    dtypes: tuple  # level dtypes
    data: tuple  # one entry per level, from _values_key


@dataclass(frozen=True)
class SeriesKey:
    """Stands in for a pandas Series in a cache key."""

    name: tuple  # (name,), or () if left out
    dtype: Any
    data: Any  # from _values_key
    index: IndexKey | None  # None if left out


@dataclass(frozen=True)
class DataFrameKey:
    """Stands in for a pandas DataFrame in a cache key."""

    columns: tuple  # column labels, in order
    dtypes: tuple  # column dtypes (pandas or numpy dtype objects)
    data: tuple  # one entry per column, from _values_key
    index: IndexKey | None  # None if left out


def dataframe_key(df: Any, *, index: bool = True) -> DataFrameKey:
    """A hashable stand-in for a pandas DataFrame, for use in ``key_converters``. Use sparingly.

    DataFrames get the same key if they have the same column labels, dtypes and values, in the
    same order (rows and columns). Values are compared exactly (no float tolerance). See the
    module notes on when to use this; :func:`series_key` describes how values are hashed.

    Args:
        df: The DataFrame.
        index: Include the index (as :func:`index_key`), the default. Leave it out only if the
            result doesn't depend on the index: e.g. if the function returns ``df['x'] * 2``,
            the result carries the index, and a later call with the same values under a
            different index would get the stored result with the old one.
    """
    pd = _pandas("dataframe_key")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"dataframe_key needs a pandas DataFrame, got {type(df).__name__}")
    data = tuple(_values_key(df.iloc[:, i]) for i in range(df.shape[1]))
    return DataFrameKey(tuple(df.columns), tuple(df.dtypes), data, _index_key(df.index) if index else None)


def series_key(s: Any, *, name: bool = True, index: bool = True) -> SeriesKey:
    """A hashable stand-in for a pandas Series, for use in ``key_converters``. Use sparingly.

    Series get the same key if they have the same dtype and values in the same order, and by
    default the same name and index. Values are hashed with ``pandas.util.hash_pandas_object``,
    except in object columns holding anything other than strings, where pandas would hash
    values by their ``str()`` (so ``1`` and ``'1'`` would match): those values go in the key
    themselves, with their types, and must be hashable or containers that :func:`container_key`
    converts.

    Args:
        s: The Series.
        name: Include the Series' name, the default. Results often carry it (arithmetic keeps
            it, ``to_frame`` uses it as the column label).
        index: Include the index (as :func:`index_key`), the default. Leave it out only if the
            result doesn't depend on it (see :func:`dataframe_key`).
    """
    pd = _pandas("series_key")
    if not isinstance(s, pd.Series):
        raise TypeError(f"series_key needs a pandas Series, got {type(s).__name__}")
    return SeriesKey((s.name,) if name else (), s.dtype, _values_key(s), _index_key(s.index) if index else None)


def index_key(idx: Any) -> IndexKey:
    """A hashable stand-in for a pandas Index or MultiIndex, for use in ``key_converters``. Use sparingly.

    Indexes get the same key if they have the same level names, dtypes and values, in the same
    order, and are both MultiIndexes or both not. Values are hashed as for :func:`series_key`.
    """
    pd = _pandas("index_key")
    if not isinstance(idx, pd.Index):
        raise TypeError(f"index_key needs a pandas Index, got {type(idx).__name__}")
    return _index_key(idx)


def _index_key(idx: Any) -> IndexKey:
    import pandas as pd

    levels = [idx.get_level_values(i) for i in range(idx.nlevels)]
    return IndexKey(
        isinstance(idx, pd.MultiIndex),
        tuple(idx.names),
        tuple(level.dtype for level in levels),
        tuple(_values_key(level) for level in levels),
    )


def _values_key(values: Any) -> Any:
    """A column or index level: a digest of pandas' hashes of its values, or for non-string objects, the values."""
    import pandas as pd

    if pd.api.types.is_object_dtype(values.dtype) and pd.api.types.infer_dtype(values, skipna=False) != "string":
        return tuple(type(v) for v in values), tuple(container_key(v) for v in values)
    hashes = pd.util.hash_pandas_object(values, index=False).to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    return hashlib.blake2b(hashes.tobytes(), digest_size=32).digest()


def _pandas(helper: str) -> Any:
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(f"{helper} needs pandas: pip install 'mcrotk-cache-utils[pandas]'") from e
    return pd
