"""Caching decorators for methods, with a separate cache per instance."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, TypeVar, overload

import cachetools

from .cache_keys import ArgRef, ArgSpec, Passthrough
from .decorator_options import MethodOptions
from .instance_copying import (
    CACHE_ATTR,
    ON_COPY,
    ON_COPY_VALUES,
    InstallsCopyMethods,
    OnCopy,
    check_copy_handling,
)
from .passthrough_cache import make_view
from .placement_checks import MethodInClassBody, add_check, reject_static_or_class_method


class _DescriptorChecks(MethodInClassBody, InstallsCopyMethods):
    """What we add to cachetools' method descriptor: the placement error and copy handling."""


if TYPE_CHECKING:
    from typing_extensions import Unpack

F = TypeVar("F", bound=Callable[..., Any])  # the decorated function: its type is kept as it is


@overload
def cached_method(func: F, /, **options: Unpack[MethodOptions]) -> F: ...
@overload
def cached_method(func: None = None, /, **options: Unpack[MethodOptions]) -> Callable[[F], F]: ...
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
    on_copy: OnCopy | None = None,
) -> Any:
    """Cache a method's results, with a separate cache for each instance.

    Use as ``@cached_method`` or ``@cached_method(...)``. Built on
    :func:`cachetools.cachedmethod`, which does the calling, binding, locking and statistics.
    For ordinary methods only: for functions, static methods and class methods use
    :func:`cached_function` (using this decorator on them is an error).

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

    **Types.** Type checkers see the method's own type, so calls are checked against its
    signature and overriding works as usual (cached or not, in either class). The cache
    attributes cachetools adds (``obj.method.cache_info()``, ...) work, but aren't part of that
    type: :func:`cache_controls` gives typed access to them.

    **Unhashable arguments** (also among ``*args`` and ``**kwargs``) are converted:

    - numpy arrays (``ndarray`` and ``memmap``) by their contents: shape, dtype and a hash of
      the data, computed on every call. So equal arrays share a result, and an array changed in
      place gets a new key. Object arrays and other array types (masked arrays, ...) need
      ``key_converters``.
    - lists, dicts and sets, and tuples and frozensets containing them, by
      :func:`container_key` with ``type_tags=False``: lists become tuples, sets frozensets and
      dicts frozensets of their ``(key, value)`` pairs, recursively. So ``[1, 2]`` and
      ``(1, 2)`` share a result. If the result depends on which was passed (e.g. it has the
      same type), keep the types with ``key_converters={'x': container_key}``.

    Anything else unhashable (including pandas objects) needs ``key_converters`` or ``omit``.

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
            place of the argument's value, e.g. to make an unhashable argument hashable, or to
            key an argument by part of it (``{'model': lambda m: m.name}``). It replaces the
            automatic handling of arrays and containers.
            Ready-made converters: :func:`container_key` (lists, dicts, sets, nested),
            :func:`array_key`, and for pandas objects (only when nothing simpler will do)
            :func:`dataframe_key`, :func:`series_key` and :func:`index_key`.
        lock, condition, info: Passed to :func:`cachetools.cachedmethod` (``lock`` and
            ``condition`` are functions of ``self`` returning the instance's lock/condition).
            With ``info=True``, ``cache_controls(obj.method).cache_info()`` gives hit and
            miss counts.
        on_copy: What a copy of the instance (``copy.copy`` or ``copy.deepcopy``) gets in
            this method's cache: ``'empty'`` (the default: never stale, even if the copy is
            then changed), ``'copy'`` (the cached results: shared with the original by
            ``copy.copy``, deep-copied by ``deepcopy``; stale if the copy is changed, until
            :func:`clear_caches`), or ``'manual'`` (the class's own ``__copy__`` handles it, e.g.
            with :func:`copy_instance`). For a different choice for one copy, use
            :func:`copy_instance`.

    **Copying.** The decorator gives the class a ``__copy__`` and ``__deepcopy__`` that do
    Python's usual copy but apply ``on_copy`` to each cached method's cache, and make the
    copy's cached methods run on the copy. Without them, ``copy.copy`` would give the copy the
    original's caches, and its cached methods would run on the original (a cachetools
    behavior). If the class (or a base) defines its own ``__copy__`` or ``__deepcopy__``,
    that one is kept: setting ``on_copy`` to ``'empty'`` or ``'copy'`` is then an error, and
    the default gives a :class:`CacheCopyWarning` (once per class) unless the methods use
    ``on_copy='manual'``. ``'manual'`` requires a ``__copy__`` of the class's own. Copies
    start with zeroed ``info`` counters. Pickling is unaffected: it carries the caches along.
    ``dataclasses.replace`` and ``copy.replace`` make new instances, with empty caches.

    Errors in the options (unknown arguments, conflicts) are raised when the decorator is
    applied. An unhashable key raises ``TypeError`` naming the argument.

    Caveats:
        - Results are returned as stored, not copied: changing a returned object in place
          changes the cached result.
        - Pickling an instance (e.g. sending it to worker processes) carries its caches along.
        - Copying a cache for ``on_copy='copy'`` isn't thread-safe: don't copy an instance
          while another thread is calling its cached methods.
        - With ``passthrough``, cached entries record the type of the returned tuple. If that's
          a NamedTuple class defined inside a function, an instance with cached results can't
          be pickled (the same limitation as pickling any object referring to a local class).
    """

    if on_copy is not None and on_copy not in ON_COPY_VALUES:
        raise ValueError(f"cached_method: on_copy must be one of {ON_COPY_VALUES}, got {on_copy!r}")

    def decorate(func: Callable[..., Any]) -> Any:
        reject_static_or_class_method(
            func,
            "cached_method",
            "For a static or class method, use @cached_function under @staticmethod/@classmethod.",
        )
        spec = ArgSpec(func, has_self=True, omit=omit, passthrough=passthrough, key_converters=key_converters)
        attr = _cache_attr_name(func)
        get_cache = _cache_getter(spec, attr, cache_factory)
        descriptor = cachetools.cachedmethod(get_cache, key=spec.make_key, lock=lock, condition=condition, info=info)(
            func
        )
        _mark_descriptor(descriptor, attr, on_copy)
        return descriptor

    return decorate if func is None else decorate(func)


def _cache_getter(
    spec: ArgSpec, attr: str, cache_factory: Callable[[], MutableMapping]
) -> Callable[[Any], MutableMapping]:
    """The function cachetools calls to get an instance's cache: made on first use, stored in ``__dict__[attr]``."""

    def get_cache(obj: Any) -> MutableMapping:
        try:
            instance_dict = obj.__dict__
        except AttributeError:
            raise TypeError(
                f"{spec.qualname}: {type(obj).__name__} instances have no __dict__ to hold the cache"
            ) from None
        cache = instance_dict.get(attr)
        if cache is None:
            check_copy_handling(type(obj))
            cache = instance_dict.setdefault(attr, cache_factory())  # setdefault: another thread may have made one
        elif not isinstance(cache, MutableMapping):
            raise TypeError(f"{spec.qualname}: attribute {attr!r} is already used for something other than a cache")
        return make_view(cache, spec.qualname) if spec.passthrough else cache

    return get_cache


def _mark_descriptor(descriptor: Any, attr: str, on_copy: OnCopy | None) -> None:
    """Add our checks to cachetools' descriptor, and the markers ``clear_caches`` and copying look for."""
    add_check(descriptor, _DescriptorChecks)
    setattr(descriptor, CACHE_ATTR, attr)
    setattr(descriptor, ON_COPY, ("empty", False) if on_copy is None else (on_copy, True))


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


def _cache_attr_name(func: Callable[..., Any]) -> str:
    """``_<Class>_<method>_cache`` from the function's qualified name, e.g. ``Sim.draw`` -> ``_Sim_draw_cache``."""
    qualname = getattr(func, "__qualname__", func.__name__)
    qualname = qualname.rsplit("<locals>.", 1)[-1]  # drop the enclosing function for local classes
    return "_" + re.sub(r"\W", "_", qualname.replace(".", "_")) + "_cache"
