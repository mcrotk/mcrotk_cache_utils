"""Copying an instance with cached methods, choosing what the copy gets in their caches.

The machinery (the ``__copy__``/``__deepcopy__`` that ``cached_method`` installs, and their checks)
is in ``copy_handling``; ``CacheCopyWarning`` is defined there, where it's given.
"""

from typing import Any, Literal, TypeVar

from .copy_handling import CacheCopyWarning, cached_methods, copy_with_caches

__all__ = ["CacheCopyWarning", "copy_instance"]

T = TypeVar("T")


def copy_instance(
    obj: T,
    caches: Literal["empty", "copy"] | None = None,
    *,
    deep: bool = False,
    memo: dict[int, Any] | None = None,
) -> T:
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
    return copy_with_caches(obj, memo, methods)
