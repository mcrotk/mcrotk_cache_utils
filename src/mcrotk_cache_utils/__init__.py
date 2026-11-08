"""Caching decorators for methods, built on cachetools: per-instance caches, argument omission, passthrough (e.g. for RNGs) and key converters."""

from .method import cached_method, clear_caches
from .rng import rng_cached_method

__all__ = ["cached_method", "clear_caches", "rng_cached_method"]
