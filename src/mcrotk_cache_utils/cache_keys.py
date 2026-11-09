"""Resolving a decorator's argument options against a function's or method's signature, and building cache keys."""

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import cachetools.keys

from .key_helpers.arrays import array_key, is_array

ArgRef = str | int  # an argument, by name or by position in the def line (counting self or cls)
Passthrough = tuple[int, Callable[[Any], Any] | None]  # (output position, transform)

_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


class PassthroughKey(tuple):
    """A cache key that also carries the current call's passed-through argument values.

    The extra values don't affect hashing or equality, so this key finds the same
    cache entries as the plain tuple of its parts.
    """

    passthrough: dict[int, tuple[Any, Callable[[Any], Any] | None]]  # {output position: (argument value, transform)}
    _hash: int

    def __new__(cls, parts: Iterable[Any], passthrough: dict[int, tuple[Any, Callable[[Any], Any] | None]]):
        key = super().__new__(cls, parts)
        key.passthrough = passthrough
        return key

    def __hash__(self) -> int:
        try:
            return self._hash
        except AttributeError:
            self._hash = tuple.__hash__(self)
            return self._hash

    def plain(self) -> tuple:
        """The key without the passthrough values, for storing in the cache."""
        return tuple(self)


class ArgSpec:
    """A function's or method's signature plus the decorator's options, resolved and checked at decoration time.

    With ``has_self`` (methods), the first parameter is the instance: it's always left out of the
    key and the options can't refer to it.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        has_self: bool,
        omit: ArgRef | Iterable[ArgRef] = (),
        passthrough: Mapping[ArgRef, int | Passthrough] | None = None,
        key_converters: Mapping[ArgRef, Callable[[Any], Any]] | None = None,
    ):
        self.qualname = getattr(func, "__qualname__", repr(func))
        self.signature = inspect.signature(func)
        self._params = list(self.signature.parameters.values())
        self.self_name: str | None = None
        if has_self:
            if (
                not self._params
                or self._params[0].kind in _VAR_KINDS
                or self._params[0].kind is inspect.Parameter.KEYWORD_ONLY
            ):
                raise TypeError(f"{self.qualname}: a cached method needs a first positional parameter (self)")
            self.self_name = self._params[0].name

        omitted = {self._resolve(ref, "omit") for ref in _as_refs(omit)}
        self.passthrough = self._resolve_passthrough(passthrough or {})
        both = omitted & set(self.passthrough)
        if both:
            raise ValueError(f"{self.qualname}: argument(s) {sorted(both)} given in both omit and passthrough")
        self.converters = self._resolve_converters(key_converters or {}, omitted)

        left_out = {self.self_name} | omitted | set(self.passthrough)  # self_name None (functions) matches nothing
        self.key_names = tuple(p.name for p in self._params if p.name not in left_out)
        self._var_positional = next((p.name for p in self._params if p.kind is inspect.Parameter.VAR_POSITIONAL), None)
        self._var_keyword = next((p.name for p in self._params if p.kind is inspect.Parameter.VAR_KEYWORD), None)

    def make_key(self, /, *args: Any, **kwargs: Any) -> tuple:
        """Build the cache key for a call (for a method, ``args`` starts with the instance)."""
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        parts = [self._key_part(name, arguments[name]) for name in self.key_names]
        if self.passthrough:
            key = PassthroughKey(
                parts, {pos: (arguments[name], transform) for name, (pos, transform) in self.passthrough.items()}
            )
        else:
            key = cachetools.keys.hashkey(*parts)
        try:
            hash(key)
        except TypeError:
            self._raise_unhashable(parts)
        return key

    def _key_part(self, name: str, value: Any) -> Any:
        if name in self.converters:
            return self.converters[name](value)
        if name == self._var_positional:
            return tuple(self._value_key(name, v) for v in value)
        if name == self._var_keyword:
            return tuple(sorted((k, self._value_key(name, v)) for k, v in value.items()))
        return self._value_key(name, value)

    def _value_key(self, name: str, value: Any) -> Any:
        """A value as it goes in the key: numpy arrays by their contents, anything else as it is."""
        if not is_array(value):
            return value
        try:
            return array_key(value)
        except TypeError as e:
            raise TypeError(
                f"{self.qualname}: can't use argument {name!r} in the cache key: {e}. {_advice(name)}"
            ) from None

    def _raise_unhashable(self, parts: list[Any]) -> None:
        for name, part in zip(self.key_names, parts, strict=True):
            try:
                hash(part)
            except TypeError as e:
                what = f"the converted value of argument {name!r}" if name in self.converters else f"argument {name!r}"
                raise TypeError(
                    f"{self.qualname}: can't use {what} in the cache key: {type(part).__name__} is unhashable. "
                    + _advice(name)
                ) from e
        raise TypeError(f"{self.qualname}: cache key is unhashable")  # not expected: some part should fail above

    def _resolve(self, ref: ArgRef, option: str, *, allow_var: bool = True) -> str:
        """Turn a name or signature position into a parameter name, checking it's allowed."""
        if isinstance(ref, bool) or not isinstance(ref, (str, int)):
            raise TypeError(f"{self.qualname}: {option} arguments must be names or positions, got {ref!r}")
        if isinstance(ref, int):
            n = len(self._params)
            if not -n <= ref < n:
                counting = " (positions count self)" if self.self_name is not None else ""
                raise ValueError(
                    f"{self.qualname}: {option} position {ref} is out of range for {n} parameters{counting}"
                )
            param = self._params[ref]
        else:
            if ref not in self.signature.parameters:
                raise ValueError(f"{self.qualname}: {option} names {ref!r}, which is not a parameter")
            param = self.signature.parameters[ref]
        if self.self_name is not None and param.name == self.self_name:
            raise ValueError(
                f"{self.qualname}: {option} can't refer to {self.self_name!r} (it is always left out of the key)"
            )
        if not allow_var and param.kind in _VAR_KINDS:
            raise ValueError(f"{self.qualname}: {option} can't refer to *{param.name} or **{param.name}")
        return param.name

    def _resolve_passthrough(self, passthrough: Mapping[ArgRef, int | Passthrough]) -> dict[str, Passthrough]:
        if not isinstance(passthrough, Mapping):
            raise TypeError(f"{self.qualname}: passthrough must be a mapping {{argument: output position}}")
        resolved: dict[str, Passthrough] = {}
        for ref, spec in passthrough.items():
            name = self._resolve(ref, "passthrough", allow_var=False)
            if name in resolved:
                raise ValueError(f"{self.qualname}: passthrough refers to {name!r} twice")
            if isinstance(spec, tuple):
                if len(spec) != 2:
                    raise TypeError(
                        f"{self.qualname}: passthrough for {name!r} must be a position or (position, transform)"
                    )
                position, transform = spec
            else:
                position, transform = spec, None
            if isinstance(position, bool) or not isinstance(position, int):
                raise TypeError(
                    f"{self.qualname}: passthrough output position for {name!r} must be an int, got {position!r}"
                )
            if transform is not None and not callable(transform):
                raise TypeError(f"{self.qualname}: passthrough transform for {name!r} must be callable")
            if any(position == p for p, _ in resolved.values()):
                raise ValueError(f"{self.qualname}: two passthrough arguments use output position {position}")
            resolved[name] = (position, transform)
        return resolved

    def _resolve_converters(
        self, converters: Mapping[ArgRef, Callable[[Any], Any]], omitted: set[str]
    ) -> dict[str, Callable[[Any], Any]]:
        if not isinstance(converters, Mapping):
            raise TypeError(f"{self.qualname}: key_converters must be a mapping {{argument: function}}")
        resolved: dict[str, Callable[[Any], Any]] = {}
        for ref, converter in converters.items():
            name = self._resolve(ref, "key_converters")
            if name in omitted or name in self.passthrough:
                raise ValueError(
                    f"{self.qualname}: key_converters has {name!r}, which is not in the key (omitted or passed through)"
                )
            if name in resolved:
                raise ValueError(f"{self.qualname}: key_converters refers to {name!r} twice")
            if not callable(converter):
                raise TypeError(f"{self.qualname}: key converter for {name!r} must be callable")
            resolved[name] = converter
        return resolved


def _advice(name: str) -> str:
    return (
        f"Make it hashable with key_converters={{{name!r}: ...}} (e.g. a function returning a tuple or bytes), "
        f"or, if it doesn't affect the result, leave it out with omit=[{name!r}]."
    )


def _as_refs(refs: ArgRef | Iterable[ArgRef]) -> tuple[ArgRef, ...]:
    """Allow a single name or position as well as a collection of them."""
    if isinstance(refs, (str, int)):
        return (refs,)
    return tuple(refs)
