"""Hashable stand-ins for arguments that can't go in a cache key as they are.

``array_key`` is applied automatically to numpy array arguments; all of them can be used in
``key_converters``. The pandas ones are never applied automatically, and are best avoided where an
argument identifying the data will do (see ``pandas_objects``).
"""

from .arrays import array_key
from .containers import container_key
from .pandas_objects import dataframe_key, index_key, series_key

__all__ = ["array_key", "container_key", "dataframe_key", "index_key", "series_key"]
