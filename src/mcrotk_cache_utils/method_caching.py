"""Caching decorators for methods, with a separate cache per instance."""

import re
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import Any

import cachetools

from .cache_keys import ArgRef, ArgSpec, Passthrough
from .passthrough_cache import make_view

_MARKER = "_mcrotk_cache_attr"  # set on our descriptors: the name of the instance attribute holding the cache


def cached_method(
    func: Callable[..., Any] | None = None,
    /,
    *,
    cache_factory: Callable[[], MutableMapping] = dict,
    omit: ArgRef | Iterable[ArgRef] = (),
    passthrough: Mapping[ArgRef, int | Passthrough] | None = None,
    key_converters: Mapping[ArgRef, Callable[[Any], Any]] | None = None,
    lock: Callable[[Any], Any] | None = None,
    condition: Callable[[Any], Any] | None = None,
    info: bool = False,
) -> Any:
    """Cache a method's results, with a separate cache for each instance.

    Use as ``@cached_method`` or ``@cached_method(...)``. Built on
    :func:`cachetools.cachedmethod`, which does the calling, binding, locking and statistics.

    **The cache.** Created on an instance's first call and stored in its ``__dict__`` as
    ``_<Class>_<method>_cache`` (``<Class>`` is the class the method is defined in, so a
    decorated override and the parent method it calls with ``super()`` don't share a cache).
    It is not a dataclass field, so it doesn't affect ``==``, ``repr`` or frozen dataclasses,
    and it disappears with the instance. The class needs a ``__dict__`` (no ``__slots__``-only
    classes).

    **The key.** Built from the call's arguments bound to the signature with defaults filled
    in, so ``f(1)``, ``f(x=1)`` and ``f()`` with default ``x=1`` are the same call. ``self`` is
    always left out. Instance attributes are not part of the key: if you change an attribute
    that affects results, call :func:`clear_caches`.

    Args:
        cache_factory: Makes each instance's cache. Default ``dict`` (unbounded). Any
            cachetools cache works, e.g. ``lambda: cachetools.LRUCache(maxsize=128)``.
        omit: Arguments to leave out of the key, by name or by position in the signature
            (positions count ``self``: in ``def f(self, a, b)``, ``a`` is 1; negative positions
            count from the end). Only omit arguments that don't affect the result: calls that
            differ only in an omitted argument get the same cached result.
        passthrough: ``{argument: output_position}`` or ``{argument: (output_position, transform)}``.
            For methods that take an argument and return it, possibly updated, as part of a
            tuple (e.g. a random number generator). The argument is left out of the key. On a
            miss the method's own outputs are returned. On a hit the stored result is returned
            with the caller's current argument (after ``transform``, if given) at
            ``output_position``. The method must return a tuple; NamedTuples are rebuilt as the
            same type.
        key_converters: ``{argument: function}``: the function's result is used in the key in
            place of the argument's value, e.g. to make an unhashable argument hashable
            (``{'arr': lambda a: a.tobytes()}``).
        lock, condition, info: Passed to :func:`cachetools.cachedmethod` (``lock`` and
            ``condition`` are functions of ``self`` returning the instance's lock/condition).
            With ``info=True``, ``obj.method.cache_info()`` gives hit and miss counts.

    Errors in the options (unknown arguments, conflicts) are raised when the decorator is
    applied. An unhashable key raises ``TypeError`` naming the argument.

    Caveats:
        - Results are returned as stored, not copied: changing a returned object in place
          changes the cached result.
        - Copying: use ``copy.deepcopy(obj)`` (the copy gets its own caches, with the cached
          results), or ``dataclasses.replace(obj, ...)`` (starts with empty caches; best when
          the copy changes attributes), or ``deepcopy`` then :func:`clear_caches`. Don't use
          ``copy.copy(obj)``: the copy shares the original's caches, and its cached methods
          may still run with the original as ``self`` (a cachetools behavior).
        - Pickling an instance (e.g. sending it to worker processes) carries its caches along.
        - With ``passthrough``, cached entries record the type of the returned tuple. If that's
          a NamedTuple class defined inside a function, an instance with cached results can't
          be pickled (the same limitation as pickling any object referring to a local class).
    """

    def decorate(func: Callable[..., Any]) -> Any:
        spec = ArgSpec(func, has_self=True, omit=omit, passthrough=passthrough, key_converters=key_converters)
        attr = _cache_attr_name(func)

        def get_cache(obj: Any) -> MutableMapping:
            try:
                instance_dict = obj.__dict__
            except AttributeError:
                raise TypeError(
                    f"{spec.qualname}: {type(obj).__name__} instances have no __dict__ to hold the cache"
                ) from None
            cache = instance_dict.get(attr)
            if cache is None:
                cache = instance_dict.setdefault(attr, cache_factory())
            elif not isinstance(cache, MutableMapping):
                raise TypeError(f"{spec.qualname}: attribute {attr!r} is already used for something other than a cache")
            return make_view(cache, spec.qualname) if spec.passthrough else cache

        descriptor = cachetools.cachedmethod(get_cache, key=spec.make_key, lock=lock, condition=condition, info=info)(
            func
        )
        setattr(descriptor, _MARKER, attr)
        return descriptor

    return decorate if func is None else decorate(func)


def clear_caches(obj: Any, *methods: str) -> None:
    """Empty ``obj``'s caches for the named cached methods, or for all of them if none are named.

    Only caches made by this library's decorators are touched. Use after changing an
    attribute that affects cached results, or after ``copy.deepcopy`` when the copy will be
    changed.
    """
    found = set()
    for cls in type(obj).__mro__:
        for name, value in vars(cls).items():
            attr = getattr(value, _MARKER, None)
            if not isinstance(attr, str) or (methods and name not in methods):
                continue
            found.add(name)
            if attr in getattr(obj, "__dict__", {}):
                value.__get__(obj, cls).cache_clear()  # through cachetools, so locks and info counters are handled
    unknown = set(methods) - found
    if unknown:
        raise ValueError(f"{type(obj).__name__} has no cached method(s) named {sorted(unknown)}")


def _cache_attr_name(func: Callable[..., Any]) -> str:
    """``_<Class>_<method>_cache`` from the function's qualified name, e.g. ``Sim.draw`` -> ``_Sim_draw_cache``."""
    qualname = getattr(func, "__qualname__", func.__name__)
    qualname = qualname.rsplit("<locals>.", 1)[-1]  # drop the enclosing function for local classes
    return "_" + re.sub(r"\W", "_", qualname.replace(".", "_")) + "_cache"
