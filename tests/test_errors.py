"""Errors raised when the decorator is applied (bad options) and when a call can't be cached."""

import numpy as np
import pytest

from mcrotk_cache_utils import cached_method, rng_cached_method


def decorate(**options):
    def f(self, a, b=1, *args, rng=None, **kwargs):
        return a, rng

    return cached_method(**options)(f)


@pytest.mark.parametrize(
    "options, error, match",
    [
        ({"omit": "nope"}, ValueError, "not a parameter"),
        ({"omit": "self"}, ValueError, "always left out"),
        ({"omit": 0}, ValueError, "always left out"),
        ({"omit": 9}, ValueError, "out of range"),
        ({"omit": True}, TypeError, "names or positions"),
        ({"passthrough": {"rng": -1}, "omit": "rng"}, ValueError, "both omit and passthrough"),
        ({"passthrough": {"args": 0}}, ValueError, r"\*args"),
        ({"passthrough": {"kwargs": 0}}, ValueError, r"\*\*kwargs"),
        ({"passthrough": {"rng": -1, "a": -1}}, ValueError, "same output position|use output position"),
        ({"passthrough": {"a": 0, 1: 1}}, ValueError, "twice"),
        ({"passthrough": {"rng": "last"}}, TypeError, "must be an int"),
        ({"passthrough": {"rng": (-1, "x")}}, TypeError, "must be callable"),
        ({"passthrough": {"rng": (-1,)}}, TypeError, "position or"),
        ({"passthrough": ["rng"]}, TypeError, "mapping"),
        ({"key_converters": {"a": 3}}, TypeError, "must be callable"),
        ({"key_converters": {"rng": tuple}, "passthrough": {"rng": -1}}, ValueError, "not in the key"),
        ({"key_converters": {"b": tuple}, "omit": "b"}, ValueError, "not in the key"),
    ],
)
def test_bad_options_fail_at_decoration(options, error, match):
    with pytest.raises(error, match=match):
        decorate(**options)


def test_positions_count_self():
    calls = []  # in f below, position 0 is self and position 1 is a

    def f(self, a, b=1):
        calls.append(a)
        return a

    g = cached_method(omit=1)(f)

    class C:
        m = g

    c = C()
    c.m(1)
    c.m(2)
    assert calls == [1]


def test_needs_self_parameter():
    with pytest.raises(TypeError, match="first positional parameter"):
        cached_method(lambda *args: None)


class C:
    @cached_method
    def f(self, arr):
        return 1

    @cached_method(key_converters={"arr": lambda a: [a]})
    def g(self, arr):
        return 1

    @cached_method(passthrough={"t": -1})
    def not_tuple(self, x, t):
        return x

    @cached_method(passthrough={"t": 5})
    def short(self, x, t):
        return x, t

    @cached_method(passthrough={"t": -1, "u": 1})
    def overlap(self, x, t, u):
        return x, t


def test_unhashable_argument_message():
    with pytest.raises(TypeError) as e:
        C().f(np.zeros(3))
    msg = str(e.value)
    assert "argument 'arr'" in msg and "ndarray is unhashable" in msg
    assert "key_converters={'arr': ...}" in msg and "omit=['arr']" in msg


def test_unhashable_converted_value_message():
    with pytest.raises(TypeError, match="converted value of argument 'arr'"):
        C().g(1)


def test_return_value_must_be_tuple_with_valid_positions():
    with pytest.raises(TypeError, match="must return a tuple"):
        C().not_tuple(1, "t")
    with pytest.raises(TypeError, match="out of range"):
        C().short(1, "t")
    with pytest.raises(TypeError, match="overlap"):
        C().overlap(1, "t", "u")


def test_rng_arg_also_in_passthrough():
    with pytest.raises(ValueError, match="rng_arg"):
        rng_cached_method(passthrough={"rng": 0})


def test_existing_non_cache_attribute():
    class D:
        @cached_method
        def f(self):
            return 1

    d = D()
    d._D_f_cache = 5
    with pytest.raises(TypeError, match="already used"):
        d.f()


def test_unhashable_argument_in_passthrough_method():
    class P:
        @cached_method(passthrough={"t": -1})
        def f(self, arr, t):
            return arr.sum(), t

    with pytest.raises(TypeError, match="argument 'arr'.*ndarray is unhashable"):
        P().f(np.zeros(2), "t")
