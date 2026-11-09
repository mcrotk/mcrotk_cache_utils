# mcrotk-cache-utils

Caching decorators for methods and functions, built on [cachetools](https://github.com/tkem/cachetools):

- **For methods, a separate cache per instance**, created on first use: no cache field to declare, nothing added to `==` or `repr`, and it works on frozen dataclasses.
- **A key built from the bound arguments**: `f(1)`, `f(x=1)` and `f()` with default `x=1` are the same call; a method's `self` is left out.
- **`omit=`**: leave arguments that don't affect the result out of the key.
- **numpy arrays as arguments**: keyed by their contents (shape, dtype and a hash of the data), so equal arrays share a result and an array changed in place doesn't get a stale one.
- **`key_converters=`**: make other unhashable arguments (lists, ...) usable in the key.
- **`passthrough=`**: for methods that take an argument and return it, e.g. a random number generator. On a cache hit, the caller's current argument is returned in its place, not the one from when the result was cached.
- **Safe copying**: `copy.copy` and `copy.deepcopy` give copies whose cached methods run on the copy, with caches emptied or copied as you choose (`on_copy=`).
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
    def fit(self, data_id, weights, verbose=False): ...

    @cached_method(passthrough={"state": (-1, normalize_state)})
    def step(self, x, state):
        ...
        return result, state


clear_caches(model)  # all cached methods
clear_caches(model, "fit")  # just one
```

Positions can be used instead of names, counting the parameters as in the `def` line from 0 (so a method's `self` is 0), with negative positions counting from the end. Names are clearer.

## Functions, static methods and class methods

```python
import numpy as np

from mcrotk_cache_utils import cached_function, rng_cached_function


@cached_function(key_converters={"weights": tuple})
def fit(data_id, weights): ...


@rng_cached_function
def simulate(n, rng=None):
    rng = np.random.default_rng(rng)
    ...
    return result, rng


class Model:
    @staticmethod
    @cached_function
    def kernel(n): ...

    @classmethod
    @cached_function
    def defaults(cls, n): ...  # cls is in the key: subclasses get their own entries


fit.cache_clear()
```

The options are the same as for methods. A function has one cache, which lives as long as the program: the default is unbounded, so pass e.g. `cache_factory=lambda: cachetools.LRUCache(maxsize=128)` to bound it. Using `cached_function` directly on an ordinary method is an error (its cache would be shared by all instances); use `cached_method`.

## Copying instances

`@cached_method` gives the class a `__copy__` and `__deepcopy__`, so `copy.copy(obj)` and `copy.deepcopy(obj)` work as usual for the other attributes, and each cached method's cache in the copy is set by its `on_copy` option:

- `"empty"` (default): the copy starts with an empty cache. Never stale, even if you then change the copy.
- `"copy"`: the copy keeps the cached results (`copy.copy` shares the result objects with the original, `deepcopy` duplicates them). Stale if you then change the copy, until `clear_caches(copy)`.
- `"manual"`: the class's own `__copy__` handles it.

For a different choice for one copy, use `copy_instance(obj, caches="empty" | "copy", deep=False)`. It also makes a correct copy inside a class's own `__copy__` or `__deepcopy__`: if the class already defines one, it's kept, and you get a `CacheCopyWarning` unless the cached methods use `on_copy="manual"`. `dataclasses.replace` and `copy.replace` start with empty caches; pickling carries the caches along.

## Caveats

- Leaving an argument out of the key (`omit`, `passthrough`) is only right if it doesn't affect the result.
- numpy arrays are hashed on every call, hit or miss: cheap next to most computations on them, but it reads the whole array. Object arrays and array subclasses (masked arrays, ...) aren't handled: use `key_converters`.
- Instance attributes aren't part of the key: after changing one that affects results, call `clear_caches(obj)`.
- Cached objects are returned as stored, not copied.
- Threads: the default `dict` cache is safe to use from several threads. cachetools' caches (`LRUCache`, ...) aren't: with them, also pass `lock=` (and `condition=` to stop two threads from computing the same result).
- Processes: a function's cache isn't shared between processes; worker processes start with empty caches. Pickling an instance carries its method caches along.

Requires Python 3.11+ and cachetools 8+; the RNG decorators also need numpy (`pip install 'mcrotk-cache-utils[numpy]'`).
