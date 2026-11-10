"""Working with the caches of cached functions and methods: ``cache_controls`` and ``clear_caches``.

``cache_controls`` gives typed access to a cached function's or method's cache attributes
(``cache_clear``, ``cache_info``, ...).

The decorators keep the decorated function's own type, so type checkers see a plain function and
don't know about the attributes cachetools adds. ``cache_controls`` gives them a type.
"""

from collections.abc import Callable
from typing import Any, Protocol

from cachetools import CacheInfo

from ..copying.copy_handling import CACHE_ATTR


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


def clear_caches(obj: Any, *methods: str) -> None:
    """Empty ``obj``'s caches for the named cached methods, or for all of them if none are named.

    Only caches made by this library's decorators are touched. Use after changing an
    attribute that affects cached results, e.g. in a copy made with ``on_copy='copy'``.
    """
    found = set()
    for cls in type(obj).__mro__:
        for name, value in vars(cls).items():
            attr = getattr(value, CACHE_ATTR, None)
            if not isinstance(attr, str) or (methods and name not in methods):
                continue
            found.add(name)
            if attr in getattr(obj, "__dict__", {}):
                value.__get__(obj, cls).cache_clear()  # through cachetools, so locks and info counters are handled
    unknown = set(methods) - found
    if unknown:
        raise ValueError(f"{type(obj).__name__} has no cached method(s) named {sorted(unknown)}")
