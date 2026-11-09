"""Caching decorators for methods and functions that take a random number generator argument.

Two styles of such functions are covered:

- **omit** (``rng_omit_cached_method``, ``rng_omit_cached_function``): the function takes the
  generator (or a seed) and draws from it, advancing the caller's generator in place, as in the
  recommended practice (Scientific Python SPEC 7). The generator is just left out of the key.
- **passthrough** (``rng_passthrough_cached_method``, ``rng_passthrough_cached_function``): the
  function also returns the generator with its result. On a hit, the caller's generator is
  returned in place of the one stored with the result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar, overload

from .cache_keys import ArgRef, Passthrough, _as_refs
from .decorator_options import (
    RngOmitFunctionOptions,
    RngOmitMethodOptions,
    RngPassthroughFunctionOptions,
    RngPassthroughMethodOptions,
)
from .function_caching import cached_function
from .method_caching import cached_method

if TYPE_CHECKING:
    from typing_extensions import Unpack

F = TypeVar("F", bound=Callable[..., Any])  # the decorated function: its type is kept as it is


# --- omit: functions that advance the caller's generator in place ---


@overload
def rng_omit_cached_method(func: F, /, **options: Unpack[RngOmitMethodOptions]) -> F: ...
@overload
def rng_omit_cached_method(func: None = None, /, **options: Unpack[RngOmitMethodOptions]) -> Callable[[F], F]: ...
def rng_omit_cached_method(
    func: Callable[..., Any] | None = None,
    /,
    *,
    rng_arg: ArgRef = "rng",
    omit: ArgRef | Iterable[ArgRef] = (),
    **kwargs: Any,
) -> Any:
    """Cache a method that takes a random number generator, leaving the generator out of the key.

    Use as ``@rng_omit_cached_method`` or ``@rng_omit_cached_method(...)``. The same as
    :func:`cached_method` with ``omit=rng_arg`` (other arguments can be added with ``omit=``);
    all other options are as for :func:`cached_method`.

    For methods that take the generator, or a seed, as an argument and draw from it::

        @rng_omit_cached_method
        def simulate(self, n, rng=None):
            rng = np.random.default_rng(rng)
            ...
            return result

    - Calls differing only in their generator share a cached result: the first call's random
      draws are reused.
    - On a miss, the method draws from the caller's generator, advancing it in place (when a
      ``Generator`` is passed). On a hit, nothing is drawn, so the generator is unchanged.

    Args:
        rng_arg: The generator argument, by name or position (positions count ``self``).
            Default ``'rng'``.
    """
    return cached_method(func, omit=(rng_arg, *_as_refs(omit)), **kwargs)


@overload
def rng_omit_cached_function(func: F, /, **options: Unpack[RngOmitFunctionOptions]) -> F: ...
@overload
def rng_omit_cached_function(func: None = None, /, **options: Unpack[RngOmitFunctionOptions]) -> Callable[[F], F]: ...
def rng_omit_cached_function(
    func: Callable[..., Any] | None = None,
    /,
    *,
    rng_arg: ArgRef = "rng",
    omit: ArgRef | Iterable[ArgRef] = (),
    **kwargs: Any,
) -> Any:
    """Cache a function that takes a random number generator, leaving the generator out of the key.

    The function version of :func:`rng_omit_cached_method`, built on :func:`cached_function`
    (whose options it takes). Also for static and class methods, with ``@staticmethod`` or
    ``@classmethod`` above it.

    Args:
        rng_arg: The generator argument, by name or position (as in the ``def`` line, from 0).
            Default ``'rng'``.
    """
    return cached_function(func, omit=(rng_arg, *_as_refs(omit)), **kwargs)


# --- passthrough: functions that also return the generator ---


@overload
def rng_passthrough_cached_method(func: F, /, **options: Unpack[RngPassthroughMethodOptions]) -> F: ...
@overload
def rng_passthrough_cached_method(
    func: None = None, /, **options: Unpack[RngPassthroughMethodOptions]
) -> Callable[[F], F]: ...
def rng_passthrough_cached_method(
    func: Callable[..., Any] | None = None,
    /,
    *,
    rng_arg: ArgRef = "rng",
    rng_output: int = -1,
    passthrough: Mapping[ArgRef, int | Passthrough] | None = None,
    **kwargs: Any,
) -> Any:
    """Cache a method that takes a random number generator and returns it, leaving it out of the key.

    Use as ``@rng_passthrough_cached_method`` or ``@rng_passthrough_cached_method(...)``. The
    same as :func:`cached_method` with
    ``passthrough={rng_arg: (rng_output, numpy.random.default_rng)}`` (other passthrough
    entries can be added with ``passthrough=``); all other options are as for
    :func:`cached_method`. Needs numpy.

    For methods that take the generator, or a seed, and also return the generator with their
    result, so that a caller who passed a seed or ``None`` can continue the stream::

        @rng_passthrough_cached_method
        def simulate(self, n, rng=None):
            rng = np.random.default_rng(rng)
            ...
            return result, rng

    - Calls differing only in their generator share a cached result: the first call's random
      draws are reused.
    - On a miss, the method's own outputs are returned, including the generator it advanced.
    - On a hit, the stored result is returned with the caller's generator passed through
      ``numpy.random.default_rng``: a ``Generator`` comes back unchanged (nothing was
      drawn), and a seed or ``None`` comes back as a new ``Generator`` made from it.

    For methods that don't return the generator, use :func:`rng_omit_cached_method`.

    Args:
        rng_arg: The generator argument, by name or position (positions count ``self``).
            Default ``'rng'``.
        rng_output: Position of the generator in the returned tuple. Default ``-1`` (last).
    """
    combined = _rng_passthrough("rng_passthrough_cached_method", rng_arg, rng_output, passthrough)
    return cached_method(func, passthrough=combined, **kwargs)


@overload
def rng_passthrough_cached_function(func: F, /, **options: Unpack[RngPassthroughFunctionOptions]) -> F: ...
@overload
def rng_passthrough_cached_function(
    func: None = None, /, **options: Unpack[RngPassthroughFunctionOptions]
) -> Callable[[F], F]: ...
def rng_passthrough_cached_function(
    func: Callable[..., Any] | None = None,
    /,
    *,
    rng_arg: ArgRef = "rng",
    rng_output: int = -1,
    passthrough: Mapping[ArgRef, int | Passthrough] | None = None,
    **kwargs: Any,
) -> Any:
    """Cache a function that takes a random number generator and returns it, leaving it out of the key.

    The function version of :func:`rng_passthrough_cached_method`, built on
    :func:`cached_function` (whose options it takes). Also for static and class methods, with
    ``@staticmethod`` or ``@classmethod`` above it. Needs numpy.

    Args:
        rng_arg: The generator argument, by name or position (as in the ``def`` line, from 0).
            Default ``'rng'``.
        rng_output: Position of the generator in the returned tuple. Default ``-1`` (last).
    """
    combined = _rng_passthrough("rng_passthrough_cached_function", rng_arg, rng_output, passthrough)
    return cached_function(func, passthrough=combined, **kwargs)


def _rng_passthrough(
    decorator: str, rng_arg: ArgRef, rng_output: int, passthrough: Mapping[ArgRef, int | Passthrough] | None
) -> dict[ArgRef, int | Passthrough]:
    """The ``passthrough`` option for a passthrough RNG decorator: the RNG entry plus any others given."""
    try:
        from numpy.random import default_rng
    except ImportError as e:
        raise ImportError(f"{decorator} needs numpy: pip install 'mcrotk-cache-utils[numpy]'") from e

    if passthrough is not None and rng_arg in passthrough:
        raise ValueError(f"{decorator}: {rng_arg!r} is set by rng_arg/rng_output; don't also put it in passthrough")
    return {rng_arg: (rng_output, default_rng), **(passthrough or {})}
