"""Caching decorators for methods and functions, built on cachetools.

Per-instance caches for methods, argument omission, passthrough (e.g. for RNGs), key converters, and
key helpers for arrays, containers and pandas objects.
"""

from .function_caching import cached_function
from .instance_copying import CacheCopyWarning, copy_instance
from .key_helpers import array_key, container_key, dataframe_key, index_key, series_key
from .method_caching import cached_method, clear_caches
from .rng_caching import rng_cached_function, rng_cached_method

__all__ = [
    "CacheCopyWarning",
    "array_key",
    "cached_function",
    "cached_method",
    "clear_caches",
    "container_key",
    "copy_instance",
    "dataframe_key",
    "index_key",
    "rng_cached_function",
    "rng_cached_method",
    "series_key",
]
