"""Caching decorators for methods and functions, built on cachetools.

Per-instance caches for methods, argument omission, passthrough (e.g. for RNGs), key converters, and
key helpers for arrays, containers and pandas objects.
"""

from .copying.instance_copying import CacheCopyWarning, copy_instance
from .decorators.cache_controls import CacheControls, cache_controls, clear_caches
from .decorators.function_caching import cached_function
from .decorators.method_caching import cached_method
from .decorators.rng_caching import (
    rng_omit_cached_function,
    rng_omit_cached_method,
    rng_passthrough_cached_function,
    rng_passthrough_cached_method,
)
from .keys.converters import array_key, container_key, dataframe_key, index_key, series_key

__all__ = [
    "CacheControls",
    "CacheCopyWarning",
    "array_key",
    "cache_controls",
    "cached_function",
    "cached_method",
    "clear_caches",
    "container_key",
    "copy_instance",
    "dataframe_key",
    "index_key",
    "rng_omit_cached_function",
    "rng_omit_cached_method",
    "rng_passthrough_cached_function",
    "rng_passthrough_cached_method",
    "series_key",
]
