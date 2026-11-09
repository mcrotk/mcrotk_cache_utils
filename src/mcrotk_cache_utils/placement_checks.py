"""Clear errors when a decorator is used in the wrong place (e.g. ``cached_function`` on a method).

cachetools' wrappers and descriptors are ordinary instances, so we give each one a subclass
of its own class that adds the check, rather than putting another layer in front of it.
"""

import functools
from typing import Any


def add_check(obj: Any, check: type) -> None:
    """Make ``obj`` an instance of a subclass of its class with the ``check`` mixin in front."""
    obj.__class__ = _with_check(type(obj), check)


@functools.cache
def _with_check(cls: type, check: type) -> type:
    # Same name and module as cachetools' class, so reprs and error messages don't change.
    return type(cls.__name__, (check, cls), {"__module__": cls.__module__, "__qualname__": cls.__qualname__})


class FunctionNotInClassBody:
    """Mixin for ``cached_function``'s wrapper: an error if it's placed directly in a class body.

    Python calls ``__set_name__`` on objects in a class body when the class is created, but not
    through ``staticmethod`` or ``classmethod``, so those (the right way to cache static and
    class methods) are unaffected.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        raise TypeError(
            f"{owner.__qualname__}.{name}: @cached_function can't cache a method: its one cache would be "
            "shared by all instances, with self in the key. Use @cached_method instead (a cache per instance). "
            "For a static or class method, put @staticmethod/@classmethod above @cached_function."
        )


def reject_static_or_class_method(func: Any, decorator: str, advice: str) -> None:
    """Error for ``@decorator`` placed above ``@staticmethod``/``@classmethod`` (it receives their object)."""
    if isinstance(func, (staticmethod, classmethod)):
        kind = type(func).__name__
        qualname = getattr(func, "__qualname__", repr(func))
        raise TypeError(f"{qualname}: @{decorator} is above @{kind}. {advice}")
