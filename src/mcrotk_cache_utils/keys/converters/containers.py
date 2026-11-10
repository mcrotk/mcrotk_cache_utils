"""Cache keys for lists, tuples, dicts and sets, converted recursively to hashable equivalents."""

from dataclasses import dataclass
from typing import Any

from .arrays import array_key, is_array


@dataclass(frozen=True)
class ContainerKey:
    """Stands in for a container in a cache key: its exact type and its converted contents."""

    type: type
    items: Any  # a tuple (lists, tuples) or a frozenset (sets; dicts as (key, value) pairs)


def container_key(value: Any, *, type_tags: bool = True) -> Any:
    """A hashable equivalent of ``value``, for use in ``key_converters``.

    Lists and tuples become tuples, sets and frozensets become frozensets, and dicts become
    frozensets of their ``(key, value)`` pairs (so their order doesn't matter, as for ``==``).
    Containers inside them are converted the same way, and numpy arrays inside them are keyed
    by their contents, as when passed directly. Anything else is left as it is.

    Args:
        value: The argument to convert.
        type_tags: Keep each container's exact type in the key (the default), so ``[2, 3]``
            and ``(2, 3)`` get separate cache entries, as do a NamedTuple and a plain tuple
            with the same values. With ``False``, containers of the same kind share entries:
            ``[2, 3]`` and ``(2, 3)``, a set and a frozenset, and also a dict and the set of
            its ``(key, value)`` pairs.
    """
    if isinstance(value, (list, tuple)):
        items: Any = tuple(container_key(v, type_tags=type_tags) for v in value)
    elif isinstance(value, (set, frozenset)):
        items = frozenset(container_key(v, type_tags=type_tags) for v in value)
    elif isinstance(value, dict):
        items = frozenset(
            (container_key(k, type_tags=type_tags), container_key(v, type_tags=type_tags)) for k, v in value.items()
        )
    elif is_array(value):
        return array_key(value)
    else:
        return value
    return ContainerKey(type(value), items) if type_tags else items
