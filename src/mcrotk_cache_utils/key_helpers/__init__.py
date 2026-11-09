"""Hashable stand-ins for arguments that can't go in a cache key as they are.

``array_key`` is applied automatically to numpy array arguments; all three can be used in
``key_converters``.
"""

from .arrays import array_key
from .containers import container_key
from .dataframes import dataframe_key

__all__ = ["array_key", "container_key", "dataframe_key"]
