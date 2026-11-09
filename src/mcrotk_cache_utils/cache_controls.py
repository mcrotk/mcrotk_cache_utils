"""Typed access to a cached function's or method's cache attributes (``cache_clear``, ``cache_info``, ...).

The decorators keep the decorated function's own type, so type checkers see a plain function and
don't know about the attributes cachetools adds. ``cache_controls`` gives them a type.
"""

from collections.abc import Callable
from typing import Any, Protocol

from cachetools import CacheInfo


class CacheControls(Protocol):
    """The cache attributes of a cached function, or of a cached method on an instance."""

    @property
    def cache(self) -> Any:
        """The cache itself (what ``cache_factory`` made)."""
        ...

    @property
    def cache_key(self) -> Callable[..., Any]: ...
    @property
    def cache_lock(self) -> Any: ...
    @property
    def cache_condition(self) -> Any: ...
    def cache_clear(self) -> None:
        """Empty the cache (and reset the ``info`` counters)."""
        ...

    def cache_info(self) -> CacheInfo:
        """Hit and miss counts. Only with ``info=True``: otherwise an ``AttributeError``."""
        ...


def cache_controls(cached: Callable[..., Any]) -> CacheControls:
    """The cache attributes of ``cached``, typed for type checkers.

    ``cached`` is a function decorated with ``cached_function`` (including a static or class
    method, accessed as ``Class.method``), or a method decorated with ``cached_method`` accessed
    on an instance (``obj.method``)::

        cache_controls(fit).cache_clear()
        cache_controls(model.fit).cache_info()

    The attributes are the same as ``fit.cache_clear()`` etc., which also work at runtime but
    which type checkers report as unknown. To empty an instance's caches, :func:`clear_caches`
    also works.
    """
    if not callable(getattr(cached, "cache_clear", None)):
        hint = (
            " (pass a cached method accessed on an instance, obj.method, not on the class)"
            if type(cached).__module__.startswith("cachetools")  # cachedmethod's class-level object
            else ""
        )
        raise TypeError(f"cache_controls: {cached!r} is not a cached function or method{hint}")
    return cached  # type: ignore[return-value]  # cachetools' wrapper (or a method bound to one) has these attributes
