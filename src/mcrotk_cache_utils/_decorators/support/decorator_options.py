"""The decorators' options, defined once for type checkers.

The decorators' typed overloads take ``**options: Unpack[...]`` of these, so each option's name
and type is written in one place. Only type checkers use them; the defaults are in the
decorators' own signatures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from typing import Any, TypedDict

from ..._copying.copy_handling import OnCopy
from ..._keys.cache_keys import ArgRef, Passthrough


class _Options(TypedDict, total=False):
    """The options of both ``cached_method`` and ``cached_function``."""

    cache_factory: Callable[[], MutableMapping[Any, Any]]
    omit: ArgRef | Iterable[ArgRef]
    passthrough: Mapping[ArgRef, int | Passthrough] | None
    key_converters: Mapping[ArgRef, Callable[[Any], Any]] | None
    info: bool


class MethodOptions(_Options, total=False):
    """``cached_method``'s options: ``lock`` and ``condition`` are functions of ``self``."""

    lock: Callable[[Any], Any] | None
    condition: Callable[[Any], Any] | None
    on_copy: OnCopy | None


class FunctionOptions(_Options, total=False):
    """``cached_function``'s options: ``lock`` and ``condition`` are the objects themselves."""

    lock: Any
    condition: Any


class RngOmitMethodOptions(MethodOptions, total=False):
    """``rng_omit_cached_method``'s options: ``cached_method``'s, plus the generator argument ``rng_arg``."""

    rng_arg: ArgRef


class RngOmitFunctionOptions(FunctionOptions, total=False):
    """``rng_omit_cached_function``'s options: ``cached_function``'s, plus the generator argument ``rng_arg``."""

    rng_arg: ArgRef


class RngPassthroughMethodOptions(RngOmitMethodOptions, total=False):
    """``rng_passthrough_cached_method``'s options: the omit ones, plus the generator's position ``rng_output``."""

    rng_output: int


class RngPassthroughFunctionOptions(RngOmitFunctionOptions, total=False):
    """``rng_passthrough_cached_function``'s options: the omit ones, plus the generator's position ``rng_output``."""

    rng_output: int
