# mcrotk-cache-utils

Caching decorators for methods and functions, built on [cachetools](https://github.com/tkem/cachetools):

- **For methods, a separate cache per instance**, created on first use: no cache field to declare, nothing added to `==` or `repr`, and it works on frozen dataclasses.
- **A key built from the bound arguments**: `f(1)`, `f(x=1)` and `f()` with default `x=1` are the same call; a method's `self` is left out.
- **`omit=`**: leave arguments that don't affect the result out of the key.
- **Unhashable arguments handled**: numpy arrays are keyed by their contents (shape, dtype and a hash of the data), so equal arrays share a result and an array changed in place doesn't get a stale one; lists, dicts and sets are converted to hashable equivalents.
- **`key_converters=`**: make other arguments usable in the key, with ready-made converters such as `container_key` and, where nothing simpler will do, pandas objects (`dataframe_key`, `series_key`, `index_key`).
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
    @cached_method(omit="verbose", key_converters={"estimator": lambda e: e.name})
    def fit(self, data_id, estimator, verbose=False): ...

    @cached_method(passthrough={"state": (-1, normalize_state)})
    def step(self, x, state):
        ...
        return result, state


clear_caches(model)  # all cached methods
clear_caches(model, "fit")  # just one
```

Positions can be used instead of names, counting the parameters as in the `def` line from 0 (so a method's `self` is 0), with negative positions counting from the end. Names are clearer.

## Unhashable arguments

Cache keys must be hashable. Two kinds of unhashable arguments are converted automatically (also inside `*args` and `**kwargs`):

- **numpy arrays**, by their contents: shape, dtype and a hash of the data (`array_key`).
- **lists, dicts and sets**, and tuples and frozensets containing them, by `container_key(value, type_tags=False)`: lists become tuples, sets become frozensets and dicts become frozensets of their `(key, value)` pairs, recursively. So `f([1, 2])` and `f((1, 2))` share a result.

For anything else, or to change how an argument is keyed, give a converter:

```python
from mcrotk_cache_utils import cached_method, container_key, dataframe_key


class Model:
    @cached_method(key_converters={"shape": container_key})
    def reshaped(self, shape): ...  # [2, 3] and (2, 3) get separate entries

    @cached_method(key_converters={"data": dataframe_key})
    def fit(self, data): ...
```

- `container_key(value, type_tags=True)`: as above, but by default each container's exact type is kept, so `[2, 3]` and `(2, 3)` get separate entries, as do a NamedTuple and a plain tuple. Use it when the result depends on which was passed, e.g. when it has the same type as the argument.
- `array_key(arr)`: what's applied to arrays automatically, for use in your own converters.
- `dataframe_key(df, index=True)`: the column labels, dtypes and a hash of each column's values (pandas' `hash_pandas_object`), plus the index's values, dtypes and names. Use `index=False` only if the result doesn't depend on the index: a result that carries the index (e.g. `df["x"] * 2`) would come back with the old index for a DataFrame with the same values under different labels.
- `series_key(s, name=True, index=True)` and `index_key(idx)`: the same for a Series (with its name) and an Index or MultiIndex.

Use the pandas helpers sparingly. Keying a pandas object by its contents reads all of it on every call (slowly for object columns), and equality between pandas objects has many corners (dtypes, labels, order, exact floats) for a key to get wrong. An argument that identifies the data, such as a path, an ID or the parameters it was made from, is usually a better key.

## Functions, static methods and class methods

```python
import numpy as np

from mcrotk_cache_utils import cached_function, rng_cached_function


@cached_function
def fit(data_id, weights): ...  # weights can be a list


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
- numpy arrays are hashed, and lists, dicts and sets converted, on every call, hit or miss: cheap next to most computations on them, but it reads all of each one. Object arrays and array subclasses (masked arrays, ...) aren't handled: use `key_converters`.
- `[1, 2]` and `(1, 2)` share a cached result. If the result depends on which was passed, use `key_converters={"x": container_key}`.
- Instance attributes aren't part of the key: after changing one that affects results, call `clear_caches(obj)`.
- Cached objects are returned as stored, not copied.
- Threads: the default `dict` cache is safe to use from several threads. cachetools' caches (`LRUCache`, ...) aren't: with them, also pass `lock=` (and `condition=` to stop two threads from computing the same result).
- Processes: a function's cache isn't shared between processes; worker processes start with empty caches. Pickling an instance carries its method caches along.

Requires Python 3.11+ and cachetools 8+; the RNG decorators also need numpy (`pip install 'mcrotk-cache-utils[numpy]'`), and the pandas helpers need pandas 1.5+ (`[pandas]`).
