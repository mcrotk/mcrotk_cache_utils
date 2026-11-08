# mcrotk-cache-utils

Caching decorators for methods, built on [cachetools](https://github.com/tkem/cachetools):

- **A separate cache per instance**, created on first use: no cache field to declare, nothing added to `==` or `repr`, and it works on frozen dataclasses.
- **A key built from the bound arguments**: `f(1)`, `f(x=1)` and `f()` with default `x=1` are the same call; `self` is left out.
- **`omit=`**: leave arguments that don't affect the result out of the key.
- **`key_converters=`**: make unhashable arguments (lists, arrays, ...) usable in the key.
- **`passthrough=`**: for methods that take an argument and return it, e.g. a random number generator. On a cache hit, the caller's current argument is returned in its place, not the one from when the result was cached.
- All of cachetools' cache types (`LRUCache`, `TTLCache`, ...) and its `lock`, `condition` and `info` options.

## Example: methods that take and return an RNG

```python
import numpy as np
from mcrotk_cache_utils import rng_cached_method

class Simulation:
    def __init__(self, n_paths):
        self.n_paths = n_paths

    @rng_cached_method
    def mean_path(self, horizon, rng=None):
        rng = np.random.default_rng(rng)
        paths = rng.normal(size=(self.n_paths, horizon)).cumsum(axis=1)
        return paths.mean(axis=0), rng

sim = Simulation(10_000)
rng = np.random.default_rng(0)
mean, rng = sim.mean_path(250, rng=rng)  # computed; rng has advanced
mean, rng = sim.mean_path(250, rng=rng)  # cached; rng returned unchanged
```

The RNG is left out of the cache key. On a hit it comes back unused (a seed or `None` comes back as a `Generator` made from it), so the caller's random stream continues correctly.

## General form

```python
from mcrotk_cache_utils import cached_method, clear_caches

class Model:
    @cached_method(omit="verbose", key_converters={"weights": tuple})
    def fit(self, data_id, weights, verbose=False):
        ...

    @cached_method(passthrough={"state": (-1, normalize_state)})
    def step(self, x, state):
        ...
        return result, state

clear_caches(model)          # all cached methods
clear_caches(model, "fit")   # just one
```

Positions can be used instead of names: they count `self`, as in the `def` line, and negative positions count from the end.

## Caveats

- Leaving an argument out of the key (`omit`, `passthrough`) is only right if it doesn't affect the result.
- Instance attributes aren't part of the key: after changing one that affects results, call `clear_caches(obj)`.
- Cached objects are returned as stored, not copied.
- Copy with `copy.deepcopy` or `dataclasses.replace`, not `copy.copy` (see the `cached_method` docstring).

Requires Python 3.11+ and cachetools 8+; `rng_cached_method` also needs numpy (`pip install 'mcrotk-cache-utils[numpy]'`).
