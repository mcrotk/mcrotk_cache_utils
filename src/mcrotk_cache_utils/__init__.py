"""Caching decorators for methods and functions, built on cachetools.

Per-instance caches for methods, argument omission, passthrough (e.g. for RNGs) and key converters.
"""

from .function_caching import cached_function
from .instance_copying import CacheCopyWarning
from .method_caching import cached_method, clear_caches
from .rng_caching import rng_cached_function, rng_cached_method

__all__ = [
    "CacheCopyWarning",
    "cached_function",
    "cached_method",
    "clear_caches",
    "rng_cached_function",
    "rng_cached_method",
]
