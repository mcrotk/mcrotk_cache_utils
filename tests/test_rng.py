import numpy as np

from mcrotk_cache_utils import rng_cached_method


class Sim:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.runs = 0

    @rng_cached_method
    def draw(self, n, rng=None):
        self.runs += 1
        rng = np.random.default_rng(rng)
        return self.scale * rng.normal(size=n).sum(), rng

    @rng_cached_method
    def nested(self, n, rng=None):
        a, rng = self.draw(n, rng=rng)
        b, rng = self.draw(n + 1, rng=rng)
        return a + b, rng

    @rng_cached_method(rng_arg="gen", rng_output=0)
    def first(self, n, gen=None):
        gen = np.random.default_rng(gen)
        return gen, gen.random(n).sum()


def test_miss_returns_functions_own_advanced_rng():
    s = Sim()
    g = np.random.default_rng(1)
    state_before = g.bit_generator.state["state"]["state"]
    _, r = s.draw(3, rng=g)
    assert r is g and g.bit_generator.state["state"]["state"] != state_before


def test_hit_returns_cached_value_and_callers_generator_unchanged():
    s = Sim()
    v1, _ = s.draw(3, rng=1)
    g = np.random.default_rng(99)
    state_before = g.bit_generator.state
    v2, r = s.draw(3, rng=g)
    assert v2 == v1 and r is g and g.bit_generator.state == state_before
    assert s.runs == 1


def test_hit_turns_seed_or_none_into_generator():
    s = Sim()
    s.draw(3, rng=1)
    _, r_seed = s.draw(3, rng=42)
    _, r_none = s.draw(3)
    assert isinstance(r_seed, np.random.Generator) and isinstance(r_none, np.random.Generator)
    # the seed was used to make the Generator, so continuing from it doesn't repeat the seed's stream
    assert r_seed.random() == np.random.default_rng(42).random()


def test_continuing_after_hit_does_not_repeat_stream():
    s = Sim()
    s.draw(3, rng=1)
    _, rng = s.draw(3, rng=7)  # hit
    a = rng.random()
    b = rng.random()
    assert a != b


def test_nested_cached_calls():
    s = Sim()
    t1, _ = s.nested(5, rng=7)
    runs = s.runs
    t2, _ = s.nested(5, rng=8)
    assert t1 == t2 and s.runs == runs


def test_custom_arg_and_output_position():
    s = Sim()
    g1, v1 = s.first(4, gen=1)
    g2, v2 = s.first(4, gen=2)
    assert v1 == v2 and isinstance(g1, np.random.Generator) and isinstance(g2, np.random.Generator)


def test_rng_not_stored_in_cache():
    s = Sim()
    s.draw(3, rng=np.random.default_rng(1))
    cache = s.__dict__["_Sim_draw_cache"]
    (key,) = cache
    (stored,) = cache.values()
    assert type(key) is tuple and key == (3,)
    assert not any(isinstance(v, np.random.Generator) for v in stored.values)
