"""Cache keys for pandas DataFrames, from their contents.

pandas is imported only when a key is made.
"""

import hashlib
from dataclasses import dataclass
from typing import Any

from .containers import container_key


@dataclass(frozen=True)
class DataFrameKey:
    """Stands in for a DataFrame in a cache key. Equal for DataFrames with the same labels, dtypes and data."""

    columns: tuple  # column labels, in order
    dtypes: tuple  # column dtypes (pandas or numpy dtype objects)
    data: tuple  # one entry per column, from _values_key
    index: tuple | None  # (level names, level dtypes, one _values_key entry per level), or None if left out


def dataframe_key(df: Any, *, index: bool = True) -> DataFrameKey:
    """A hashable stand-in for a pandas DataFrame, for use in ``key_converters``.

    DataFrames get the same key if they have the same column labels, dtypes and values, in the
    same order (rows and columns). Values are hashed with ``pandas.util.hash_pandas_object``,
    except in object columns holding anything other than strings, where pandas would hash
    values by their ``str()`` (so ``1`` and ``'1'`` would match): those values go in the key
    themselves, with their types, and must be hashable or containers that :func:`container_key`
    converts.

    Args:
        df: The DataFrame.
        index: Include the index (its values, dtypes and level names), the default. Leave it out
            only if the result doesn't depend on the index: e.g. if the function returns
            ``df['x'] * 2``, the result carries the index, and a later call with the same values
            under a different index would get the stored result with the old one.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("dataframe_key needs pandas: pip install 'mcrotk-cache-utils[pandas]'") from e

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"dataframe_key needs a pandas DataFrame, got {type(df).__name__}")
    data = tuple(_values_key(df.iloc[:, i]) for i in range(df.shape[1]))
    index_key = None
    if index:
        levels = [df.index.get_level_values(i) for i in range(df.index.nlevels)]
        index_key = (
            tuple(df.index.names),
            tuple(level.dtype for level in levels),
            tuple(_values_key(level) for level in levels),
        )
    return DataFrameKey(tuple(df.columns), tuple(df.dtypes), data, index_key)


def _values_key(values: Any) -> Any:
    """A column or index level: a digest of pandas' hashes of its values, or for non-string objects, the values."""
    import pandas as pd

    if pd.api.types.is_object_dtype(values.dtype) and pd.api.types.infer_dtype(values, skipna=False) != "string":
        return tuple(type(v) for v in values), tuple(container_key(v) for v in values)
    hashes = pd.util.hash_pandas_object(values, index=False).to_numpy()  # pyright: ignore[reportAttributeAccessIssue]
    return hashlib.blake2b(hashes.tobytes(), digest_size=32).digest()
