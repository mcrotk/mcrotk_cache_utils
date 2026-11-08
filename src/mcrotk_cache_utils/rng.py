"""Caching decorators for methods that take and return a numpy random number generator."""

from collections.abc import Callable, Mapping
from typing import Any

from ._spec import ArgRef, Passthrough
from .method import cached_method


def rng_cached_method(
    func: Callable[..., Any] | None = None,
    /,
    *,
    rng_arg: ArgRef = "rng",
    rng_output: int = -1,
    passthrough: Mapping[ArgRef, int | Passthrough] | None = None,
    **kwargs: Any,
) -> Any:
    """Cache a method that takes a random number generator and returns it, ignoring the RNG in the key.

    Use as ``@rng_cached_method`` or ``@rng_cached_method(...)``. The same as
    :func:`cached_method` with ``passthrough={rng_arg: (rng_output, numpy.random.default_rng)}``
    (other passthrough entries can be added with ``passthrough=``). All other options are
    as for :func:`cached_method`.

    For methods following numpy's advice to pass the RNG explicitly and return it::

        @rng_cached_method
        def simulate(self, n, rng=None):
            rng = np.random.default_rng(rng)
            ...
            return result, rng

    - The RNG is left out of the cache key, so calls differing only in their RNG share a
      cached result.
    - On a miss, the method's own outputs are returned, including the RNG it advanced.
    - On a hit, the stored result is returned with the caller's RNG passed through
      ``numpy.random.default_rng``: a ``Generator`` comes back unchanged (no numbers were
      drawn), and a seed or ``None`` comes back as a new ``Generator``, so the caller's next
      call continues from it rather than reseeding.

    Args:
        rng_arg: The RNG argument, by name or position (positions count ``self``). Default ``'rng'``.
        rng_output: Position of the RNG in the returned tuple. Default ``-1`` (last).
    """
    try:
        from numpy.random import default_rng
    except ImportError as e:
        raise ImportError("rng_cached_method needs numpy: pip install 'mcrotk-cache-utils[numpy]'") from e

    if passthrough is not None and rng_arg in passthrough:
        raise ValueError(
            f"rng_cached_method: {rng_arg!r} is set by rng_arg/rng_output; don't also put it in passthrough"
        )
    combined = {rng_arg: (rng_output, default_rng), **(passthrough or {})}
    return (
        cached_method(func, passthrough=combined, **kwargs)
        if func is not None
        else cached_method(passthrough=combined, **kwargs)
    )
