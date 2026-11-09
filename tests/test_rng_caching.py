import pytest

from mcrotk_cache_utils import cache_controls, rng_passthrough_cached_function, rng_passthrough_cached_method

np = pytest.importorskip("numpy")


class Sim:
    def __init__(self, scale=1.0):
        self.scale = scale
        self.runs = 0

    @rng_passthrough_cached_method
    def draw(self, n, rng=None):
        self.runs += 1
        rng = np.random.default_rng(rng)
        return self.scale * rng.normal(size=n).sum(), rng

    @rng_passthrough_cached_method
    def nested(self, n, rng=None):
        a, rng = self.draw(n, rng=rng)
        b, rng = self.draw(n + 1, rng=rng)
        return a + b, rng

    @rng_passthrough_cached_method(rng_arg="gen", rng_output=0)
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


# Every kind of RNG argument numpy.random.default_rng accepts. Each entry makes a fresh one,
# since Generators and bit generators are consumed when used.
RNG_KINDS = {
    "none": lambda: None,
    "int": lambda: 42,
    "int_array": lambda: [1, 2, 3],
    "seed_sequence": lambda: np.random.SeedSequence(5),
    "bit_generator": lambda: np.random.PCG64(6),
    "generator": lambda: np.random.default_rng(7),
}
DETERMINISTIC = [k for k in RNG_KINDS if k != "none"]


@pytest.mark.parametrize("kind", RNG_KINDS)
def test_miss_with_each_rng_kind(kind):
    s = Sim()
    value, rng = s.draw(3, rng=RNG_KINDS[kind]())
    assert isinstance(rng, np.random.Generator)
    if kind != "none":
        expected = np.random.default_rng(RNG_KINDS[kind]()).normal(size=3).sum()
        assert value == expected  # computed with the caller's RNG, as if not cached


@pytest.mark.parametrize("kind", RNG_KINDS)
def test_hit_with_each_rng_kind(kind):
    s = Sim()
    cached_value, _ = s.draw(3, rng=0)
    arg = RNG_KINDS[kind]()
    value, rng = s.draw(3, rng=arg)
    assert value == cached_value and s.runs == 1
    assert isinstance(rng, np.random.Generator)
    if kind == "generator":
        assert rng is arg
    elif kind == "bit_generator":
        assert rng.bit_generator is arg
    if kind in DETERMINISTIC:
        # unused: the returned RNG continues exactly as a fresh one from the caller's argument
        assert rng.random() == np.random.default_rng(RNG_KINDS[kind]()).random()


def test_rng_passed_positionally():
    s = Sim()
    v1, _ = s.draw(3, 1)
    g = np.random.default_rng(2)
    v2, r = s.draw(3, g)
    assert v2 == v1 and r is g and s.runs == 1


def test_info_with_rng_passthrough():
    class C:
        @rng_passthrough_cached_method(info=True)
        def f(self, n, rng=None):
            rng = np.random.default_rng(rng)
            return rng.random(n).sum(), rng

    c = C()
    c.f(2, rng=1)
    c.f(2, rng=2)
    info = cache_controls(c.f).cache_info()
    assert (info.hits, info.misses, info.currsize) == (1, 1, 1)


def make_draw():
    """A fresh passthrough-cached function and the list of its runs (a function's cache lasts as long as it does)."""
    runs = []

    @rng_passthrough_cached_function
    def draw(n, rng=None):
        runs.append(n)
        rng = np.random.default_rng(rng)
        return rng.normal(size=n).sum(), rng

    return draw, runs


def test_function_miss_returns_functions_own_advanced_rng():
    draw, _ = make_draw()
    g = np.random.default_rng(1)
    state_before = g.bit_generator.state["state"]["state"]
    _, r = draw(3, rng=g)
    assert r is g and g.bit_generator.state["state"]["state"] != state_before


@pytest.mark.parametrize("kind", RNG_KINDS)
def test_function_hit_with_each_rng_kind(kind):
    draw, runs = make_draw()
    cached_value, _ = draw(3, rng=0)
    arg = RNG_KINDS[kind]()
    value, rng = draw(3, rng=arg)
    assert value == cached_value and runs == [3]
    assert isinstance(rng, np.random.Generator)
    if kind == "generator":
        assert rng is arg
    if kind in DETERMINISTIC:
        assert rng.random() == np.random.default_rng(RNG_KINDS[kind]()).random()


def test_function_rng_not_stored_and_positional_rng():
    draw, runs = make_draw()
    draw(3, 1)
    g = np.random.default_rng(2)
    _, r = draw(3, g)
    assert r is g and runs == [3]
    ((key, stored),) = cache_controls(draw).cache.items()
    assert key == (3,) and not any(isinstance(v, np.random.Generator) for v in stored.values)


def test_function_custom_arg_and_output_position():
    @rng_passthrough_cached_function(rng_arg=0, rng_output=0)  # the first parameter, by position
    def first(gen, n):
        gen = np.random.default_rng(gen)
        return gen, gen.random(n).sum()

    g1, v1 = first(1, 4)
    g2, v2 = first(2, 4)
    assert v1 == v2 and isinstance(g2, np.random.Generator)
    assert g2.random() == np.random.default_rng(2).random()


def test_rng_static_and_class_methods():
    class S:
        runs = 0

        @staticmethod
        @rng_passthrough_cached_function
        def draw(n, rng=None):
            S.runs += 1
            rng = np.random.default_rng(rng)
            return rng.random(n).sum(), rng

        @classmethod
        @rng_passthrough_cached_function(info=True)
        def scaled(cls, n, rng=None):
            value, rng = cls.draw(n, rng=rng)
            return 2 * value, rng

    a, _ = S.scaled(3, rng=1)
    g = np.random.default_rng(5)
    b, r = S().scaled(3, rng=g)
    assert a == b and r is g and S.runs == 1
    assert cache_controls(S.scaled).cache_info().hits == 1
