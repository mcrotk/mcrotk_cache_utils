"""Resolving a decorator's argument options against a function's or method's signature, and building cache keys."""

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import cachetools.keys

from .converters.arrays import array_key, is_array
from .converters.containers import container_key
from .passthrough import PassthroughKey

ArgRef = str | int  # an argument, by name or by position in the def line (counting self or cls)
Passthrough = tuple[int, Callable[[Any], Any] | None]  # (output position, transform)

_VAR_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
_CONTAINERS = (list, tuple, dict, set, frozenset)  # converted automatically when unhashable


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
        """An argument's part of the key: its converter's result, or its value(s) as ``_value_key`` gives them."""
        if name in self.converters:
            return self.converters[name](value)
        if name == self._var_positional:
            return tuple(self._value_key(name, v) for v in value)
        if name == self._var_keyword:
            return tuple(sorted((k, self._value_key(name, v)) for k, v in value.items()))
        return self._value_key(name, value)

    def _value_key(self, name: str, value: Any) -> Any:
        """A value as it goes in the key: numpy arrays by their contents, unhashable containers converted
        (by ``container_key`` without type tags), anything else as it is."""
        try:
            if is_array(value):
                return array_key(value)
            if isinstance(value, _CONTAINERS) and not _is_hashable(value):
                converted = container_key(value, type_tags=False)
                hash(converted)  # something unhashable inside: report it here, naming the argument
                return converted
        except TypeError as e:
            raise TypeError(
                f"{self.qualname}: can't use argument {name!r} in the cache key: {e}. {_advice(name)}"
            ) from None
        return value

    def _raise_unhashable(self, parts: list[Any]) -> None:
        """Raise a ``TypeError`` naming the first argument whose part of the key is unhashable."""
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
        """Check ``passthrough`` and key it by parameter name: ``{name: (output position, transform)}``."""
        if not isinstance(passthrough, Mapping):
            raise TypeError(f"{self.qualname}: passthrough must be a mapping {{argument: output position}}")
        resolved: dict[str, Passthrough] = {}
        for ref, entry in passthrough.items():
            name = self._resolve(ref, "passthrough", allow_var=False)
            if name in resolved:
                raise ValueError(f"{self.qualname}: passthrough refers to {name!r} twice")
            position, transform = self._passthrough_entry(name, entry)
            if any(position == p for p, _ in resolved.values()):
                raise ValueError(f"{self.qualname}: two passthrough arguments use output position {position}")
            resolved[name] = (position, transform)
        return resolved

    def _passthrough_entry(self, name: str, entry: int | Passthrough) -> Passthrough:
        """One passthrough value, ``position`` or ``(position, transform)``, checked, as ``(position, transform)``."""
        if isinstance(entry, tuple):
            if len(entry) != 2:
                raise TypeError(
                    f"{self.qualname}: passthrough for {name!r} must be a position or (position, transform)"
                )
            position, transform = entry
        else:
            position, transform = entry, None
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError(
                f"{self.qualname}: passthrough output position for {name!r} must be an int, got {position!r}"
            )
        if transform is not None and not callable(transform):
            raise TypeError(f"{self.qualname}: passthrough transform for {name!r} must be callable")
        return position, transform

    def _resolve_converters(
        self, converters: Mapping[ArgRef, Callable[[Any], Any]], omitted: set[str]
    ) -> dict[str, Callable[[Any], Any]]:
        """Check ``key_converters`` (none for omitted or passed-through arguments) and key it by parameter name."""
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


def _is_hashable(value: Any) -> bool:
    """Whether ``hash(value)`` works."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _advice(name: str) -> str:
    """The fixes to suggest for an unhashable argument ``name``."""
    return (
        f"Make it hashable with key_converters={{{name!r}: ...}} (e.g. a function returning a tuple or bytes), "
        f"or, if it doesn't affect the result, leave it out with omit=[{name!r}]."
    )


def _as_refs(refs: ArgRef | Iterable[ArgRef]) -> tuple[ArgRef, ...]:
    """Allow a single name or position as well as a collection of them."""
    if isinstance(refs, (str, int)):
        return (refs,)
    return tuple(refs)
