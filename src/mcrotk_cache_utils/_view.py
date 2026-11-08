"""The cache view used for passthrough: strips passed-through outputs when storing, puts the caller's back on a hit."""

from collections.abc import Iterator, MutableMapping
from typing import Any, NamedTuple

import cachetools

from ._spec import PassthroughKey


class Stored(NamedTuple):
    """What the cache holds for a passthrough method: the result without its passed-through outputs."""

    type: type
    length: int
    values: tuple


class PassthroughView(MutableMapping):
    """View of an instance's real cache, given to cachetools in place of the cache.

    cachetools hands the key built by our key function straight to the cache. That key
    carries the current call's passed-through argument values, which this view puts back
    into the stored result on a hit.
    """

    def __init__(self, cache: MutableMapping, qualname: str):
        self._cache = cache
        self._qualname = qualname

    def __getitem__(self, key: Any) -> Any:
        stored = self._cache[key]  # KeyError: a miss
        if not isinstance(key, PassthroughKey):  # looked up directly, not during a call
            return stored
        return _rebuild(stored, key.passthrough)

    def __setitem__(self, key: PassthroughKey, result: Any) -> None:
        self._cache[key.plain()] = self._strip(key, result)

    def setdefault(self, key: PassthroughKey, default: Any = None) -> Any:
        # Used by cachetools' locked wrapper after a miss. If another thread stored this key
        # meanwhile, keep this call's own result: its passed-through outputs (e.g. its own
        # advanced RNG) are the right ones for this caller.
        plain = key.plain()
        if plain not in self._cache:
            self._cache[plain] = self._strip(key, default)
        return default

    def __delitem__(self, key: Any) -> None:
        del self._cache[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: object) -> bool:
        return key in self._cache

    def clear(self) -> None:
        self._cache.clear()

    def __getattr__(self, name: str) -> Any:
        # Pass through cache attributes such as maxsize and currsize (used by cache_info).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._cache, name)

    def __repr__(self) -> str:
        return f"PassthroughView({self._cache!r})"

    def _strip(self, key: PassthroughKey, result: Any) -> Stored:
        # Errors here are TypeError, not ValueError: cachetools ignores a ValueError from
        # storing (it means "value too large"), which would hide the problem.
        if not isinstance(result, tuple):
            raise TypeError(
                f"{self._qualname}() must return a tuple when passthrough is used, got {type(result).__name__}"
            )
        length = len(result)
        positions = []
        for position in key.passthrough:
            if not -length <= position < length:
                raise TypeError(
                    f"{self._qualname}(): passthrough output position {position} is out of range "
                    f"for the returned {length}-tuple"
                )
            positions.append(position % length)
        if len(set(positions)) != len(positions):
            raise TypeError(
                f"{self._qualname}(): passthrough output positions {list(key.passthrough)} overlap "
                f"for the returned {length}-tuple"
            )
        drop = set(positions)
        return Stored(type(result), length, tuple(v for i, v in enumerate(result) if i not in drop))


class PassthroughCacheView(PassthroughView, cachetools.Cache):
    """A PassthroughView over a cachetools ``Cache``, which is itself a ``Cache``.

    cachetools' ``cache_info()`` reports ``maxsize`` and ``currsize`` only for ``Cache``
    instances. ``Cache.__init__`` is never called: every method that would use its own
    storage is overridden, by PassthroughView or here, to use the real cache instead.
    """

    @property
    def maxsize(self) -> Any:
        return self._cache.maxsize  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def currsize(self) -> Any:
        return self._cache.currsize  # pyright: ignore[reportAttributeAccessIssue]

    # A method, not a staticmethod as in Cache, since it delegates to the real cache (Cache
    # itself replaces it per instance when given a custom getsizeof).
    def getsizeof(self, value: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._cache.getsizeof(value)  # pyright: ignore[reportAttributeAccessIssue]


def make_view(cache: MutableMapping, qualname: str) -> PassthroughView:
    """The view to hand to cachetools in place of ``cache``."""
    if isinstance(cache, cachetools.Cache):
        return PassthroughCacheView(cache, qualname)
    return PassthroughView(cache, qualname)


def _rebuild(stored: Stored, passthrough: dict[int, tuple[Any, Any]]) -> tuple:
    """Put the current call's passed-through values (transformed) back into a stored result."""
    inserts = {}
    for position, (value, transform) in passthrough.items():
        inserts[position % stored.length] = transform(value) if transform is not None else value
    rest = iter(stored.values)
    items = [inserts[i] if i in inserts else next(rest) for i in range(stored.length)]
    if stored.type is tuple:
        return tuple(items)
    if hasattr(stored.type, "_make"):  # NamedTuple
        return stored.type._make(items)
    return stored.type(items)
