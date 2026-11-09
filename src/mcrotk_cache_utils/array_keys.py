"""Cache keys for numpy arrays, from their contents: shape, dtype and a hash of the data.

numpy isn't imported here: if the user's code hasn't imported it, no arrays can be passed.
"""

import hashlib
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArrayKey:
    """Stands in for a numpy array in a cache key. Equal for arrays with the same shape, dtype and data."""

    shape: tuple[int, ...]
    dtype: Any  # numpy.dtype (hashable, and includes the byte order and any field names)
    digest: bytes  # BLAKE2b hash of the data in C order


def is_array(value: Any) -> bool:
    """Whether ``value`` is a plain numpy array (``ndarray`` or ``memmap``), whose key can come from its contents.

    Other subclasses may hold state beyond the data (a masked array's mask, a quantity's unit), so they
    aren't covered.
    """
    np = sys.modules.get("numpy")
    return np is not None and type(value) in (np.ndarray, np.memmap)


def array_key(array: Any) -> ArrayKey:
    """The cache key of a plain numpy array. Raises ``TypeError`` for object arrays (their data is pointers)."""
    import numpy as np

    if array.dtype.hasobject:
        raise TypeError(f"arrays with object dtype ({array.dtype}) can't be hashed by content")
    data = np.ascontiguousarray(array).reshape(-1).view(np.uint8)  # the bytes in C order, without a copy if possible
    return ArrayKey(array.shape, array.dtype, hashlib.blake2b(data, digest_size=32).digest())
