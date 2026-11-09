# mcrotk-cache-utils

Caching decorators for Python methods and functions, built on [cachetools](https://github.com/tkem/cachetools), that handle the details which usually make caching a method awkward.

## What it's for

Caching a method's results should be one line. With the standard tools it isn't:

- **`functools.cache` and `lru_cache`** keep one cache for the whole class, keyed on `self`. Instances must be hashable (a regular dataclass isn't), and every instance stays in memory as long as the cache does.
- **`cachetools.cachedmethod`** needs each instance to create and hold its own cache: on a dataclass, a field to keep out of `==` and `repr`, and on a frozen dataclass, a workaround to set it.
- **Keys** are built from the arguments as passed, so `f(1)` and `f(x=1)` are separate entries. Arguments that don't affect the result (a `verbose` flag) can't be left out, and arrays, lists and dicts can't be used at all.
- **Random number generators.** The recommended practice ([SPEC 7](https://scientific-python.org/specs/spec-0007/)) is to pass a numpy `Generator` (or a seed) into each function that needs random numbers, which advances it in place. As part of a key, a generator is compared by identity, not by its state, so calls with different generators never share a cached result.

With `@cached_method`, each instance gets its own cache, created on first use and gone with the instance. Keys come from the arguments as bound to the signature. Arguments can be left out of the key, numpy arrays, lists and dicts work as arguments, and `@rng_omit_cached_method` leaves a generator argument out of the key (with `@rng_passthrough_cached_method` for methods that also return the generator).

## Installation

```sh
pip install mcrotk-cache-utils
pip install 'mcrotk-cache-utils[numpy]'    # for the RNG passthrough decorators
pip install 'mcrotk-cache-utils[pandas]'   # for the pandas key helpers
pip install 'mcrotk-cache-utils[numpy,pandas]'   # both
```

The latest development version: `pip install "mcrotk-cache-utils @ git+https://github.com/mcrotk/mcrotk_cache_utils"`.

Requires Python 3.11+ and cachetools 8+ (installed with it); numpy 1.23.2+ and pandas 1.5+ only for the features that use them.

## Examples

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

Methods that take a random number generator, and draw from it:

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
rng = np.random.default_rng(0)
mean = sim.mean_path(250, rng)  # computed: the draws advance rng
mean = sim.mean_path(250, rng)  # cached: nothing is drawn, rng is unchanged
```

Methods that also return the generator with their result:

```python
import numpy as np

from mcrotk_cache_utils import rng_passthrough_cached_method


class Simulation:
    def __init__(self, n_paths: int):
        self.n_paths = n_paths

    @rng_passthrough_cached_method
    def mean_path(self, horizon: int, rng=None) -> tuple[np.ndarray, np.random.Generator]:
        rng = np.random.default_rng(rng)
        paths = rng.normal(size=(self.n_paths, horizon)).cumsum(axis=1)
        return paths.mean(axis=0), rng


sim = Simulation(10_000)
mean, rng = sim.mean_path(250, rng=0)  # computed; rng is the generator made from seed 0, advanced
mean, rng = sim.mean_path(250, rng=rng)  # cached; rng comes back as it was passed in
```

On a hit, the caller's generator is returned in place of the one stored with the result (a seed or `None` comes back as a generator made from it).

Functions, with arrays as arguments:

```python
import numpy as np

from mcrotk_cache_utils import cached_function


@cached_function
def normalized(values: np.ndarray) -> np.ndarray:
    return values / values.sum()


normalized(np.arange(5.0))  # computed
normalized(np.arange(5.0))  # cached: an equal array
```

## Features

- **[A cache per instance](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#caching-methods)**, with nothing to declare: it isn't a field, so `==`, `repr` and frozen dataclasses are unaffected.
- **[Keys from the bound arguments](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#cache-keys)**, with `omit=` for arguments that don't affect the result.
- **[Unhashable arguments](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#unhashable-arguments)**: numpy arrays are keyed by their contents, and lists, dicts and sets are converted, automatically. Converters for anything else, with ready-made ones for pandas objects.
- **[Random number generators](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#random-number-generators)**: `rng_omit_cached_method` leaves a generator argument out of the key, and `rng_passthrough_cached_method` (built on the general **[passthrough](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#passing-arguments-through)** option) handles methods that also return it. Both have function versions.
- **[Functions, static methods and class methods](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#functions-static-methods-and-class-methods)** (`cached_function`).
- **[Safe copying](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#copying-and-pickling)**: `copy.copy` and `copy.deepcopy` give copies whose caches are emptied or copied, as you choose.
- **[cachetools' cache types and options](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#cache-types-sizes-and-statistics)**: `LRUCache`, `TTLCache`, ..., locks and hit/miss statistics.
- **[Type checking](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md#type-checking)**: decorated functions keep their signatures for pyright, mypy and editors.

## Documentation

- **[User guide](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/guide.md)**: how each part works, with examples and caveats.
- **[API reference](https://github.com/mcrotk/mcrotk_cache_utils/blob/main/docs/reference.md)**: every function and option.
