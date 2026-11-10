# mcrotk-cache-utils

Caching decorators for Python methods, built on [cachetools](https://github.com/tkem/cachetools): a cache for each instance, with nothing to set up, and simple options to leave an argument out of the key or pass it through to the result. Functions, static methods and class methods get the same handling of keys and arguments.

## What it's for

The standard tools (`functools.cache` and `lru_cache`, and cachetools) leave several common cases awkward or impossible:

- **A cache for each instance**
  - `functools.cache` and `lru_cache` keep one cache for the whole class, keyed on `self`. Instances must be hashable (a regular dataclass isn't), and every instance stays in memory as long as the cache does.
  - `cachetools.cachedmethod` needs each instance to create and hold its own cache: on a dataclass, a field to keep out of `==` and `repr`, and on a frozen dataclass, a workaround to set it.
  - A cache held by the instance is shared by `copy.copy`, so a copy that's then changed gets the original's results. With `cachetools.cachedmethod`, a copy made after a call even runs the method on the original.
- **Choosing what goes in the key**, for methods and functions alike
  - Arguments are keyed as passed, so `f(1)` and `f(x=1)` are separate entries.
  - Leaving an argument out of the key (e.g., a `verbose` flag) isn't an option in `functools`. In cachetools it means writing a key function that repeats the method's signature, including its defaults, handles the argument whether it's passed by position or by name, and has to be kept in step whenever the method changes.
  - Arguments returned with the result (a state, a counter) can't simply be left out of the key: a hit would return the value stored by an earlier call, not the one passed in this call.
  - Random number generators: the recommended practice ([SPEC 7](https://scientific-python.org/specs/spec-0007/)) is to pass a numpy `Generator` (or a seed) into each function that needs random numbers, which advances it in place. As part of a key, a generator is compared by identity, not by its state, so calls with different generators never share a cached result.
- **Unhashable arguments:** numpy arrays, lists and dicts can't be used at all.

What this library does:

- **[A cache per instance](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#caching-methods)**, with nothing to declare. `@cached_method` creates it on first use, outside the fields, and it goes with the instance. Instances needn't be hashable, and `==`, `repr` and frozen dataclasses are unaffected.
- **[Keys from the bound arguments](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#cache-keys).** `f(1)` and `f(x=1)` share an entry, with defaults filled in.
- **[Omitting arguments](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#cache-keys)** with one option: `omit="verbose"`, by name or position, however the argument is passed. Use it for arguments that don't affect the result, or whose effect shouldn't produce separate cache entries.
- **[Passthrough](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#passing-arguments-through)** for any argument returned with the result. On a hit, the stored result comes back with the value passed in this call, not the one stored with it.
- **[Random number generators](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#random-number-generators):** a common case of omission, and the original motivation for this library, with its own decorator. `@rng_omit_cached_method` leaves the generator out of the key, so a hit draws nothing. `@rng_passthrough_cached_method` caches methods written to also return the generator. On a hit, it returns the generator passed in this call in place of the stored one.
- **[Unhashable arguments](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#unhashable-arguments).** numpy arrays (keyed by their contents), lists, dicts and sets work automatically. Converters handle anything else, with ready-made ones for pandas objects.
- **[Safe copies](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#copying-and-pickling).** `copy.copy` and `copy.deepcopy` give the copy its own cache, empty or copied, and its cached methods run on the copy.
- **[Functions too](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#functions-static-methods-and-class-methods).** `@cached_function` and function versions of the RNG decorators, with the same keys and options, also for static and class methods.
- **[cachetools underneath](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#cache-types-sizes-and-statistics):** its cache types (`LRUCache`, `TTLCache`, ...), locks and hit/miss statistics.
- **[Typed](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#type-checking).** Decorated methods and functions keep their signatures for pyright, mypy and editors.

## Installation

Not on PyPI yet; until it is, install from GitHub:

```sh
pip install "mcrotk-cache-utils @ git+https://github.com/mcrotk/mcrotk_cache_utils"
pip install "mcrotk-cache-utils[numpy] @ git+https://github.com/mcrotk/mcrotk_cache_utils"         # for the RNG passthrough decorators
pip install "mcrotk-cache-utils[pandas] @ git+https://github.com/mcrotk/mcrotk_cache_utils"        # for the pandas key helpers
pip install "mcrotk-cache-utils[numpy,pandas] @ git+https://github.com/mcrotk/mcrotk_cache_utils"  # both
```

Requires Python 3.11+ and cachetools 8+ (installed with it); numpy 1.23.2+ and pandas 1.5+ only for the features that use them.

## Examples

### Omitting arguments

```python
from dataclasses import dataclass

from mcrotk_cache_utils import cached_method


@dataclass
class Portfolio:
    weights: list[float]

    @cached_method(omit="verbose")
    def risk(self, horizon: int, verbose: bool = False) -> float:
        if verbose:
            print("computing...")
        return sum(w * w for w in self.weights) * horizon


p = Portfolio([0.5, 0.5])
p.risk(10)  # computed
p.risk(horizon=10)  # cached: the same call
p.risk(10, verbose=True)  # cached: verbose isn't part of the key
Portfolio([0.5, 0.5]).risk(10)  # computed: another instance, its own cache
```

### Omitting a random number generator

A special case of omission, common enough to get its own decorator: `@rng_omit_cached_method` is `@cached_method(omit="rng")`.

```python
import numpy as np

from mcrotk_cache_utils import rng_omit_cached_method


class Simulation:
    def __init__(self, n_paths: int):
        self.n_paths = n_paths

    @rng_omit_cached_method
    def mean_path(self, horizon: int, rng=None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        paths = rng.normal(size=(self.n_paths, horizon)).cumsum(axis=1)
        return paths.mean(axis=0)


sim = Simulation(10_000)
rng = np.random.default_rng(0)  # one generator, made once and passed to every call
short = sim.mean_path(10, rng)  # computed: the draws advance rng
long = sim.mean_path(250, rng)  # computed: continues the same stream
short = sim.mean_path(10, rng)  # cached: nothing is drawn, rng is unchanged
```

### Passing arguments through

For an argument returned with the result. Here, a count of evaluations, which a hit shouldn't add to:

```python
from dataclasses import dataclass

from mcrotk_cache_utils import cached_method


@dataclass
class Model:
    data: list[float]

    @cached_method(passthrough={"n_evals": -1})
    def loss(self, center: float, n_evals: int = 0) -> tuple[float, int]:
        value = sum((x - center) ** 2 for x in self.data)  # stands in for an expensive computation
        return value, n_evals + 1


model = Model([1.0, 2.0, 4.0])
loss, n_evals = model.loss(2.0)  # computed: n_evals == 1
loss, n_evals = model.loss(2.5, n_evals)  # computed: n_evals == 2
loss, n_evals = model.loss(2.0, n_evals)  # cached: n_evals == 2, as passed in
```

Left in the key, `n_evals` would make every call a miss; with `omit=`, the last call would return the stored count, 1.

### Array arguments, and functions

`@cached_function` for functions; numpy arrays are keyed by their contents:

```python
import numpy as np

from mcrotk_cache_utils import cached_function


@cached_function
def normalized(values: np.ndarray) -> np.ndarray:
    return values / values.sum()


normalized(np.arange(5.0))  # computed
normalized(np.arange(5.0))  # cached: an equal array
```

## Documentation

- **[User guide](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md)**: how each part works, with examples and caveats.
- **[API reference](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/reference.md)**: every function and option.

## License

MIT; see [LICENSE](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/LICENSE).
