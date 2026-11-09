"""Copying instances with cached methods: the ``__copy__``/``__deepcopy__`` that ``cached_method`` installs.

Python's default copy would duplicate the instance ``__dict__``, which holds our caches and
cachetools' bound wrappers (bound to the original instance). The installed methods do the
default copy (honoring ``__getstate__``, ``__reduce__``, ``__slots__``), but first drop the
wrappers (recreated for the copy on first use) and empty or copy each cache according to its
method's ``on_copy``.
"""

import copy
import os
import sys
import warnings
import weakref
from collections.abc import Iterator, MutableMapping
from typing import Any, Literal, NamedTuple

import cachetools

OnCopy = Literal["empty", "copy", "manual"]
ON_COPY_VALUES = ("empty", "copy", "manual")

CACHE_ATTR = "_mcrotk_cache_attr"  # on our descriptors: the name of the instance attribute holding the cache
ON_COPY = "_mcrotk_on_copy"  # on our descriptors: (on_copy value, whether it was given explicitly)
_OURS = "_mcrotk_copy"  # on the __copy__/__deepcopy__ functions we install

_WRAPPERS = "__cachetools_cachedmethods"  # cachetools' per-instance wrappers for super() lookups


class CacheCopyWarning(UserWarning):
    """A class with cached methods defines its own ``__copy__`` or ``__deepcopy__``.

    Then ``cached_method`` can't install its copy handling, and whether copies are correct
    depends on that method (:func:`copy_instance` can do the copying inside it). Silence with
    ``on_copy='manual'`` on the cached methods, or with
    ``warnings.filterwarnings('ignore', category=CacheCopyWarning)``.
    """


class CachedMethodInfo(NamedTuple):
    name: str  # the method's name in its class (also the key of its cachetools wrapper in instances)
    cache_attr: str
    on_copy: str
    explicit: bool


def cached_methods(cls: type) -> Iterator[tuple[type, CachedMethodInfo]]:
    """Our cached methods in ``cls`` and its bases, with the class each is defined in."""
    for klass in cls.__mro__:
        for name, value in vars(klass).items():
            attr = getattr(value, CACHE_ATTR, None)
            if isinstance(attr, str):
                on_copy, explicit = getattr(value, ON_COPY, ("empty", False))
                yield klass, CachedMethodInfo(name, attr, on_copy, explicit)


# --- Installing at class creation ---


class InstallsCopyMethods:
    """Mixin for ``cached_method``'s descriptor: copy handling for the class, set up when the class is created."""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)  # pyright: ignore[reportAttributeAccessIssue]  (cachetools' descriptor)
        on_copy, explicit = getattr(self, ON_COPY)
        theirs = user_copy_methods(owner)
        where = f"{owner.__qualname__}.{name}"
        if on_copy == "manual":
            if "__copy__" not in theirs:
                raise TypeError(
                    f"{where}: on_copy='manual', but {owner.__qualname__} has no __copy__ of its own, so copy.copy "
                    "would give the copy the original's cache and run its cached methods on the original. Define "
                    "__copy__, or use on_copy='empty' or 'copy'."
                )
            return
        if theirs and explicit:
            raise TypeError(
                f"{where}: on_copy={on_copy!r} can't take effect: {owner.__qualname__} defines its own "
                f"{' and '.join(theirs)}. Use on_copy='manual' and handle the cache there."
            )
        if theirs:
            _warn_once(owner, theirs)
        for dunder, function in (("__copy__", _copy), ("__deepcopy__", _deepcopy)):
            if dunder not in theirs and getattr(owner, dunder, None) is not function:
                setattr(owner, dunder, function)


def user_copy_methods(cls: type) -> list[str]:
    """Which of ``__copy__``/``__deepcopy__`` ``cls`` gets from its own code (itself or a base), not from us."""
    return [
        dunder
        for dunder in ("__copy__", "__deepcopy__")
        if (method := getattr(cls, dunder, None)) is not None and not getattr(method, _OURS, False)
    ]


_checked: "weakref.WeakSet[type]" = weakref.WeakSet()
_warned: "weakref.WeakSet[type]" = weakref.WeakSet()


def check_copy_handling(cls: type) -> None:
    """At an instance's first cache: warn (once per class) if ``cls`` replaced our copy methods.

    Catches subclasses that define ``__copy__``/``__deepcopy__``, which class creation can't see.
    """
    if cls in _checked:
        return
    _checked.add(cls)
    theirs = user_copy_methods(cls)
    if theirs and any(info.on_copy != "manual" for _, info in cached_methods(cls)):
        _warn_once(cls, theirs)


def _warn_once(cls: type, theirs: list[str]) -> None:
    if cls in _warned:
        return
    _warned.add(cls)
    warnings.warn(
        f"{cls.__qualname__} defines its own {' and '.join(theirs)}, so copies of its instances don't get "
        "mcrotk_cache_utils' handling of its cached methods' caches. Make sure "
        f"{'it doesn' if len(theirs) == 1 else 'they don'}'t give the copy the original's caches or the "
        "cachetools wrappers in the instance __dict__ (bound to the original): copy_instance(self) makes "
        "such a copy correctly. Then set on_copy='manual' on the cached methods to silence this warning.",
        CacheCopyWarning,
        stacklevel=_caller_outside_libraries(),
    )


_SKIP_DIRS = tuple(os.path.dirname(file) + os.sep for file in (__file__, cachetools.__file__))
_SKIP_FILES = (getattr(sys.modules.get("abc"), "__file__", None) or "",)  # ABCMeta.__new__ runs __set_name__


def _caller_outside_libraries() -> int:
    """The stacklevel of the nearest frame outside this package, cachetools and abc (the user's code)."""
    level, frame = 2, sys._getframe(2)  # 1: _warn_once; 2: its caller
    while frame is not None and (
        frame.f_code.co_filename.startswith(_SKIP_DIRS) or frame.f_code.co_filename in _SKIP_FILES
    ):
        level, frame = level + 1, frame.f_back
    return level


# --- The installed methods ---


def _copy(self: Any) -> Any:
    return _copy_instance(self, None)


def _deepcopy(self: Any, memo: dict[int, Any]) -> Any:
    return _copy_instance(self, memo)


_copy.__name__ = _copy.__qualname__ = "__copy__"
_deepcopy.__name__ = _deepcopy.__qualname__ = "__deepcopy__"
_copy.__doc__ = "Shallow copy; cached methods' caches are emptied or copied according to their on_copy."
_deepcopy.__doc__ = "Deep copy; cached methods' caches are emptied or deep-copied according to their on_copy."
setattr(_copy, _OURS, True)
setattr(_deepcopy, _OURS, True)


def copy_instance(
    obj: Any,
    caches: Literal["empty", "copy"] | None = None,
    *,
    deep: bool = False,
    memo: dict[int, Any] | None = None,
) -> Any:
    """Copy an instance with cached methods, choosing what the copy gets in their caches.

    Makes the same copy as ``copy.copy(obj)`` (or ``copy.deepcopy(obj)`` with ``deep=True``)
    does with the copy methods ``cached_method`` installs, but never calls the class's own
    ``__copy__``/``__deepcopy__``, so it can be used inside them::

        @cached_method(on_copy="manual")
        def simulate(self, n): ...

        def __copy__(self):
            new = copy_instance(self)
            ...  # anything else your copy needs
            return new

    Args:
        obj: The instance to copy.
        caches: ``'empty'`` or ``'copy'`` for every cached method's cache in the copy (see
            ``on_copy`` in :func:`cached_method`), or ``None`` (default) to follow each method's
            ``on_copy``, with ``'manual'`` methods' caches emptied.
        deep: Make a deep copy.
        memo: Inside your own ``__deepcopy__(self, memo)``, pass its ``memo`` (implies ``deep``),
            so references shared with other parts of the object being copied stay shared.
    """
    if caches not in (None, "empty", "copy"):
        raise ValueError(f"copy_instance: caches must be 'empty', 'copy' or None, got {caches!r}")
    if memo is None and deep:
        memo = {}
    methods = [
        info._replace(on_copy=caches or ("empty" if info.on_copy == "manual" else info.on_copy))
        for _, info in cached_methods(type(obj))
    ]
    return _copy_instance(obj, memo, methods)


def _copy_instance(obj: Any, memo: dict[int, Any] | None, methods: list[CachedMethodInfo] | None = None) -> Any:
    """Python's default copy (deep if ``memo`` is given), with the cache handling applied first."""
    deep = memo is not None
    if methods is None:
        methods = [info for _, info in cached_methods(type(obj))]
    rv = obj.__reduce_ex__(4)
    if isinstance(rv, str):  # a global (copied by reference), as in copy.copy
        return obj
    func, args, state, *rest = rv
    if isinstance(state, dict):
        state, prepared = _prepare(state, methods, deep), True
    elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], dict):  # (__dict__, __slots__ values)
        state, prepared = (_prepare(state[0], methods, deep), state[1]), True
    else:
        prepared = False  # a custom __getstate__/__reduce__ state: fix the copy's __dict__ afterwards
    new = copy._reconstruct(obj, memo, func, args, state, *rest)  # pyright: ignore[reportAttributeAccessIssue]
    if not prepared and isinstance(getattr(new, "__dict__", None), dict):
        fixed = _prepare(new.__dict__, methods, deep)
        new.__dict__.clear()
        new.__dict__.update(fixed)
    return new


def _prepare(instance_dict: dict[str, Any], methods: list[CachedMethodInfo], deep: bool) -> dict[str, Any]:
    """A copy of the instance's ``__dict__`` for the new instance: no wrappers; caches emptied or copied."""
    result = dict(instance_dict)
    result.pop(_WRAPPERS, None)
    for info in methods:
        if type(result.get(info.name)).__module__.startswith("cachetools"):
            del result[info.name]  # cachetools' wrapper, bound to the original
        if info.cache_attr not in result or info.on_copy == "manual":
            continue
        if info.on_copy == "empty":
            del result[info.cache_attr]  # created again on first use
        elif not deep:  # "copy", shallow (a deep copy copies the cache with the rest of the state)
            result[info.cache_attr] = copy_cache(result[info.cache_attr])
    return result


def copy_cache(cache: MutableMapping) -> MutableMapping:
    """A new cache with the same entries: the cached keys and results are shared, not copied.

    cachetools' caches don't support ``copy.copy`` (the copy would share their internal
    structures), so the cache is deep-copied with its keys and results kept as they are. This
    keeps e.g. an LRUCache's order and a TTLCache's expiry times.
    """
    if type(cache) is dict:
        return cache.copy()
    keys = list(cache)
    if isinstance(cache, cachetools.Cache):
        # Read the values without "using" them: LRUCache.__getitem__ would change the original's order.
        values = [cachetools.Cache.__getitem__(cache, key) for key in keys]
    else:
        values = [cache[key] for key in keys]
    memo: dict[int, Any] = {id(obj): obj for obj in (*keys, *values)}
    return copy.deepcopy(cache, memo)
