"""Caching decorators for functions, including static and class methods, with one cache per function."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar, overload

import cachetools

from ..keys.cache_keys import ArgRef, ArgSpec, Passthrough
from ..keys.passthrough import make_view
from .support.decorator_options import FunctionOptions
from .support.placement_checks import FunctionNotInClassBody, add_check, reject_static_or_class_method

if TYPE_CHECKING:
    from typing_extensions import Unpack

F = TypeVar("F", bound=Callable[..., Any])  # the decorated function: its type is kept as it is


@overload
def cached_function(func: F, /, **options: Unpack[FunctionOptions]) -> F: ...
@overload
def cached_function(func: None = None, /, **options: Unpack[FunctionOptions]) -> Callable[[F], F]: ...
def cached_function(
    func: Callable[..., Any] | None = None,
    /,
    *,
    cache_factory: Callable[[], MutableMapping] = dict,
    omit: ArgRef | Iterable[ArgRef] = (),
    passthrough: Mapping[ArgRef, int | Passthrough] | None = None,
    key_converters: Mapping[ArgRef, Callable[[Any], Any]] | None = None,
    lock: Any = None,
    condition: Any = None,
    info: bool = False,
) -> Any:
    """Cache a function's results.

    Use as ``@cached_function`` or ``@cached_function(...)``. Built on
    :func:`cachetools.cached`, which does the calling, locking and statistics. The options are
    the same as for :func:`cached_method`, except that ``lock`` and ``condition`` are objects
    rather than functions of ``self``.

    **The cache.** One per function, made by ``cache_factory`` when the decorator is applied,
    and available as ``cache_controls(f).cache``. It lives as long as the function does,
    usually the whole program, so with the default (an unbounded ``dict``) it keeps every
    result until ``cache_controls(f).cache_clear()`` is called. Pass e.g.
    ``cache_factory=lambda: cachetools.LRUCache(maxsize=128)`` to bound it.

    **The key.** Built from the call's arguments bound to the signature with defaults filled
    in, so ``f(1)``, ``f(x=1)`` and ``f()`` with default ``x=1`` are the same call. Unhashable
    arrays and containers are converted as for :func:`cached_method`. Positions in the options
    count the parameters as in the ``def`` line, from 0.

    **Types.** Type checkers see the decorated function's own type (its signature and return
    type). The cache attributes cachetools adds (``f.cache_clear()``, ...) work, but aren't part
    of that type: :func:`cache_controls` gives typed access to them.

    **Static and class methods.** Put ``@staticmethod`` or ``@classmethod`` above this decorator.
    For a class method, ``cls`` is part of the key, so subclasses get their own entries (in one
    shared cache); ``omit=['cls']`` makes them share results. For ordinary methods use
    :func:`cached_method`: using this decorator directly in a class body is an error.

    Args:
        cache_factory: Makes the cache. Default ``dict`` (unbounded).
        omit, passthrough, key_converters: As for :func:`cached_method`.
        lock, condition, info: Passed to :func:`cachetools.cached`. A ``dict`` cache is safe
            to use from several threads without a lock (at worst two threads compute the same
            result); cachetools' caches (``LRUCache`` etc.) are not, so pass e.g.
            ``lock=threading.Lock()`` with them. ``condition`` also stops two threads from
            computing the same result. With ``info=True``, ``cache_controls(f).cache_info()``
            gives hit and miss counts.

    Caveats:
        - Results are returned as stored, not copied: changing a returned object in place
          changes the cached result.
        - Each process has its own cache: worker processes (e.g. in a ``ProcessPoolExecutor``)
          start with empty caches, and their results don't reach the parent's cache.
    """

    def decorate(func: Callable[..., Any]) -> Any:
        """Wrap ``func`` with cachetools, with its key spec and its one cache."""
        reject_static_or_class_method(
            func, "cached_function", "Put @staticmethod/@classmethod above @cached_function instead."
        )
        spec = ArgSpec(func, has_self=False, omit=omit, passthrough=passthrough, key_converters=key_converters)
        cache = cache_factory()
        if spec.passthrough:
            cache = make_view(cache, spec.qualname)
        wrapper = cachetools.cached(cache, key=spec.make_key, lock=lock, condition=condition, info=info)(func)
        add_check(wrapper, FunctionNotInClassBody)
        return wrapper

    return decorate if func is None else decorate(func)
