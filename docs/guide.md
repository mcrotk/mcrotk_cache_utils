# User guide

How each part of mcrotk-cache-utils works. For every option in one place, see the [API reference](reference.md).

- [Caching methods](#caching-methods)
- [Cache keys](#cache-keys)
- [Unhashable arguments](#unhashable-arguments)
- [Passing arguments through](#passing-arguments-through)
- [Random number generators](#random-number-generators)
- [Functions, static methods and class methods](#functions-static-methods-and-class-methods)
- [Cache types, sizes and statistics](#cache-types-sizes-and-statistics)
- [Copying and pickling](#copying-and-pickling)
- [Threads and processes](#threads-and-processes)
- [Type checking](#type-checking)
- [Errors](#errors)
- [Caveats](#caveats)

## Caching methods

```python
from mcrotk_cache_utils import cached_method, clear_caches


class Model:
    def __init__(self, data):
        self.data = data

    @cached_method
    def fit(self, alpha=1.0): ...
```

Each instance gets its own cache, made on its first call to `fit`. It's stored in the instance's `__dict__` (as `_Model_fit_cache`), so it disappears with the instance. Nothing has to be declared, and since it isn't a dataclass field, it doesn't affect `==` or `repr`, and works on frozen dataclasses. The class needs a `__dict__`, so classes with only `__slots__` aren't supported.

Each cached method has its own cache, named after the class it's defined in. A cached override and the parent method it calls with `super()` keep separate caches.

**Instance attributes aren't part of the key.** If you change an attribute that results depend on, empty the caches:

```python
clear_caches(model)  # every cached method of model
clear_caches(model, "fit")  # just fit
```

Other instances are unaffected. Frozen dataclasses avoid the question, since their attributes can't change.

## Cache keys

The key is made from the call's arguments **bound to the method's signature, with defaults filled in**, so `f(1)`, `f(x=1)` and `f()` (with default `x=1`) are the same call. A method's `self` is never part of the key: each instance has its own cache anyway.

**`omit=`** leaves arguments out of the key, for arguments that don't affect the result:

```python
@cached_method(omit=["verbose", "n_jobs"])
def fit(self, alpha, verbose=False, n_jobs=1): ...
```

Calls that differ only in omitted arguments get the same cached result, so only omit arguments that really don't change it.

**`key_converters=`** replaces an argument's value in the key by a function of it, for example to key a model object by its name:

```python
@cached_method(key_converters={"estimator": lambda e: e.name})
def fit(self, estimator): ...
```

Options can refer to arguments by name, or by position in the `def` line counting from 0 (so a method's `self` is position 0) with negative positions counting from the end. Names are clearer. `*args` and `**kwargs` are part of the key too: their values in order, and their names and values (in sorted order).

Option errors, such as an unknown argument name or an argument both omitted and passed through, are raised when the decorator is applied, not at the first call.

## Unhashable arguments

Cache keys must be hashable. Two kinds of unhashable arguments are handled automatically, also inside `*args` and `**kwargs`:

- **numpy arrays** (`ndarray`, `memmap`) are keyed by their contents: shape, dtype and a hash of the data. Equal arrays share a result (in any memory layout), and an array changed in place gets a new key rather than a stale result. The hash reads the whole array on every call, hit or miss: usually cheap next to the computation being cached.
- **lists, dicts and sets**, and tuples and frozensets containing them, are converted to hashable equivalents, recursively: lists to tuples, sets to frozensets and dicts to frozensets of their `(key, value)` pairs. Arrays inside them are keyed by content. So `f([1, 2])` and `f((1, 2))` share a result, as do dicts with the same items in a different order.

Anything else unhashable raises a `TypeError` naming the argument and suggesting `key_converters` or `omit`. That includes object arrays (whose data is pointers), array subclasses such as masked arrays (which carry more than their data), and pandas objects.

### Ready-made converters

For `key_converters`:

- **`container_key(value, type_tags=True)`**: the container conversion above, but by default keeping each container's exact type, so `[2, 3]` and `(2, 3)` get separate entries, as do a NamedTuple and a plain tuple. Use it when the result depends on which was passed, for example when it returns the same type as the argument:

  ```python
  @cached_method(key_converters={"shape": container_key})
  def zeros(self, shape): ...
  ```

- **`array_key(arr)`**: what arrays get automatically, for use in your own converters.
- **`dataframe_key(df, index=True)`**, **`series_key(s, name=True, index=True)`** and **`index_key(idx)`**: pandas objects, by their labels, dtypes and values.

### pandas objects

Use the pandas helpers only when nothing simpler will do. Keying a pandas object by its contents reads all of it on every call (slowly for columns of mixed Python objects), and equality between pandas objects has many corners for a key to get wrong. An argument that identifies the data, such as a path, an ID or the parameters it was made from, is usually a better key.

When you do use them:

- Two objects get the same key if their labels, dtypes and values match exactly, in the same order (rows and columns). Floats are compared exactly; NaNs in the same places match.
- **`index=True`** (the default) includes the index: its values, dtypes and names. Leave it out (`index=False`) only if the result doesn't depend on it. A result that carries the index, such as `df["x"] * 2`, would otherwise come back with the old index for a DataFrame with the same values under different labels.
- **`name=True`** (Series only) includes the Series' name. Results often carry it too.
- Values are hashed with pandas' `hash_pandas_object`, except in object columns holding anything other than strings, where pandas would hash values by their text (so `1` and `"1"` would match). Those values go in the key themselves, with their types.

## Passing arguments through

Some functions take an argument and return it, possibly updated, as part of their result: a state, a counter, a random number generator. Such an argument usually shouldn't be part of the key, but on a cache hit the caller still expects to get *their* argument back, not the one stored from an earlier call.

**`passthrough=`** handles this: `{argument: output_position}`.

```python
@cached_method(passthrough={"state": -1})
def step(self, x, state):
    ...
    return result, state
```

- The argument is left out of the key.
- On a miss, the method's own result is returned (with the state as the method updated it).
- On a hit, the stored result is returned with the caller's current argument at `output_position`.

The method must return a tuple; a NamedTuple is rebuilt as the same type. `{argument: (output_position, transform)}` applies `transform` to the caller's argument on a hit, which is how the RNG passthrough decorators turn a seed into a generator.

## Random number generators

The recommended practice for numpy's random numbers ([SPEC 7](https://scientific-python.org/specs/spec-0007/)) is to pass a generator, or a seed, into each function that needs random numbers, rather than relying on global state: create one `Generator` where the computation starts and pass it explicitly to everything that needs it. Functions normalize the argument with `np.random.default_rng(rng)`, which returns a `Generator` it's given unchanged, so their draws advance the caller's generator in place.

A generator can't usefully be part of a cache key: it's compared by identity, not by its state, so calls with different generators would never share a result.

### Functions that take a generator

```python
import numpy as np

from mcrotk_cache_utils import rng_omit_cached_method


class Simulation:
    @rng_omit_cached_method
    def simulate(self, n, rng=None):
        rng = np.random.default_rng(rng)  # a Generator, a seed or None
        ...
        return result


rng = np.random.default_rng(seed)
a = sim.simulate(1000, rng)  # computed: the draws advance rng
b = sim.simulate(1000, rng)  # cached: nothing is drawn, rng is unchanged
```

`rng_omit_cached_method` is `cached_method` with `omit="rng"`:

- **The generator isn't part of the key**, so calls differing only in their generator (or seed) share a result.
- **On a miss**, the method draws from the caller's generator, advancing it in place.
- **On a hit**, nothing is drawn, and the generator is unchanged.

It doesn't need numpy itself, since it only leaves the argument out of the key.

### Functions that also return the generator

Some code also returns the generator with its result (`return result, rng`). That isn't part of the recommended practice, and when a `Generator` is passed in it adds nothing, since the caller's generator has already advanced in place. But it lets a caller who passed a seed or `None` continue the stream the function created.

For such functions, leaving the generator out of the key isn't enough: on a hit, the stored result would hand back the generator from the call that computed it, not the caller's. `rng_passthrough_cached_method` handles this. It's `cached_method` with `passthrough={"rng": (-1, np.random.default_rng)}` (see [Passing arguments through](#passing-arguments-through)):

```python
from mcrotk_cache_utils import rng_passthrough_cached_method


class Simulation:
    @rng_passthrough_cached_method
    def simulate(self, n, rng=None):
        rng = np.random.default_rng(rng)
        ...
        return result, rng
```

- **The generator isn't part of the key**, as above.
- **On a miss**, the method's own outputs are returned, including the generator it advanced.
- **On a hit**, the stored result is returned with the caller's generator passed through `np.random.default_rng`. A `Generator` comes back unchanged (nothing was drawn), and a seed or `None` comes back as a new `Generator` made from it, so the caller's next call continues from it rather than reseeding.

It needs numpy (`pip install 'mcrotk-cache-utils[numpy]'`).

### Either way

Caching treats a result as determined by the other arguments: the first call's draws are reused by every later call with the same arguments. That's the point when the random draws are an implementation detail (a simulation estimate you want once per set of parameters), but not when callers want fresh draws each time. And since a hit draws nothing, the numbers drawn after a cached call differ from those of an uncached run. A run is still reproducible from its seed, as long as it starts with the same cached results (for example, none).

For work in several processes, give each process its own generator, from `rng.spawn(n)` (NumPy 1.25+) or `np.random.SeedSequence(seed).spawn(n)`, rather than copies of one, which would produce the same numbers.

Options: `rng_arg=` names the generator argument (default `"rng"`), and for the passthrough decorators `rng_output=` gives its position in the returned tuple (default `-1`, the last). All of `cached_method`'s options apply too, including further `omit=` or `passthrough=` entries. `rng_omit_cached_function` and `rng_passthrough_cached_function` are the same for functions.

## Functions, static methods and class methods

```python
from mcrotk_cache_utils import cached_function


@cached_function
def fit(data_id, weights): ...


class Model:
    @staticmethod
    @cached_function
    def kernel(n): ...

    @classmethod
    @cached_function
    def defaults(cls, n): ...
```

`cached_function` takes the same options as `cached_method`, except that `lock` and `condition` are objects rather than functions of `self`. A function has one cache, made when it's decorated, which lives as long as the function does, usually the whole program. The default cache is unbounded, so for functions called with many different arguments, pass a bounded one (see below).

For static and class methods, put `@staticmethod` or `@classmethod` above `@cached_function`. A class method's `cls` is part of the key, so subclasses get their own entries in the one shared cache; `omit="cls"` makes them share results.

Using the wrong decorator is an error: `cached_function` directly on an ordinary method (its cache would be shared by every instance), or `cached_method` on a function, static method or class method.

## Cache types, sizes and statistics

**`cache_factory=`** makes the cache: any `MutableMapping`, including all of cachetools' cache types. The default is `dict`: unbounded, and safe to use from several threads.

```python
import cachetools


@cached_method(cache_factory=lambda: cachetools.LRUCache(maxsize=128))
def fit(self, alpha): ...
```

For methods, the factory is called once per instance; for functions, once.

**`info=True`** counts hits and misses. **`lock=`** and **`condition=`** are for threads (see below). The cache's attributes are reached through `cache_controls`:

```python
from mcrotk_cache_utils import cache_controls

cache_controls(model.fit).cache_info()  # CacheInfo(hits=..., misses=..., maxsize=..., currsize=...), with info=True
cache_controls(model.fit).cache  # the cache itself
cache_controls(fit).cache_clear()  # empties a function's cache
```

`model.fit.cache_info()` and `fit.cache_clear()` also work, but type checkers don't know about them (see [Type checking](#type-checking)).

## Copying and pickling

`copy.copy` and `copy.deepcopy` of an instance with cached methods work as usual for its other attributes. Each cached method's cache in the copy follows its **`on_copy=`** option:

- **`"empty"`** (the default): the copy starts with an empty cache. It's never stale, even if you then change the copy.
- **`"copy"`**: the copy keeps the cached results. `copy.copy` shares the result objects with the original, and `copy.deepcopy` duplicates them. They're stale if you then change the copy, until `clear_caches(copy)`.
- **`"manual"`**: the class's own `__copy__` handles it (see below).

Either way, the copy's cached methods run on the copy. (With plain `cachetools.cachedmethod`, a copy made after a cached method was called runs that method on the *original*.)

**`copy_instance(obj, caches=None, deep=False)`** makes a copy with a different choice for every cache (`"empty"` or `"copy"`):

```python
from mcrotk_cache_utils import copy_instance

copy_instance(model, "copy")  # keep the cached results this time
```

**If the class defines its own `__copy__` or `__deepcopy__`**, it's kept, and it decides what the copy gets. `copy_instance` makes a correct copy inside it:

```python
class Model:
    @cached_method(on_copy="manual")
    def fit(self, alpha): ...

    def __copy__(self):
        new = copy_instance(self)
        ...  # anything else the copy needs
        return new
```

Without `on_copy="manual"`, such a class gets a `CacheCopyWarning` (once per class) as a reminder to check that its copy method handles the caches. Setting `on_copy="empty"` or `"copy"` on it is an error, since the class's own method would ignore it.

**Other ways of making instances**: `dataclasses.replace` and `copy.replace` make new instances, which start with empty caches. **Pickling** carries the caches along, so an unpickled instance (for example in a worker process) has the original's cached results.

## Threads and processes

**Threads.** The default `dict` cache is safe to use from several threads: at worst, two threads compute the same result at the same time. cachetools' caches (`LRUCache`, `TTLCache`, ...) aren't, so pass a lock with them. For methods, `lock` is a function of the instance returning its lock:

```python
import threading


class Model:
    def __init__(self):
        self.lock = threading.Lock()

    @cached_method(cache_factory=lambda: cachetools.LRUCache(maxsize=128), lock=lambda self: self.lock)
    def fit(self, alpha): ...
```

For functions, `lock` is the lock itself: `@cached_function(lock=threading.Lock(), ...)`. `condition=` (a `threading.Condition`, or for methods a function returning one) also stops two threads from computing the same result. Copying an instance with `on_copy="copy"` while another thread calls its cached methods isn't safe.

**Processes.** Each process has its own function caches: worker processes start with empty ones, and their results don't come back to the parent. Pickling an instance to send it to a worker carries its method caches along.

## Type checking

The decorators keep the decorated function's own type, so type checkers (pyright, mypy) and editors see its signature and return type, check calls against it, and allow a subclass to override a cached method with a plain one, or a plain one with a cached one.

The attributes cachetools adds (`cache_clear()`, `cache_info()`, `cache`, `cache_key`, `cache_lock`, `cache_condition`) aren't part of that type: they work at runtime, but type checkers report them as unknown. `cache_controls` gives typed access to them:

```python
cache_controls(fit).cache_clear()
cache_controls(model.fit).cache_info()
```

For the original undecorated function, use `inspect.unwrap(fit)`.

## Errors

Mistakes are reported as early as possible.

When the decorator is applied:

- An unknown argument name or an out-of-range position in `omit`, `passthrough` or `key_converters`, or an argument given to two options that conflict (such as both `omit` and `passthrough`).
- Either decorator placed *above* `@staticmethod` or `@classmethod` (put those on top).

When the class is created:

- `cached_function` directly on an ordinary method (use `cached_method`).
- `on_copy="empty"` or `"copy"` on a class with its own `__copy__`/`__deepcopy__`, or `on_copy="manual"` without its own `__copy__`.

When it's called:

- `cached_method` on a function outside a class, or under `@staticmethod` or `@classmethod` (use `cached_function`).
- An unhashable value the key can't use: a `TypeError` naming the argument.
- A `passthrough` method that doesn't return a tuple long enough for its output positions.

## Caveats

- **Only leave out of the key** (with `omit` or `passthrough`) arguments that don't affect the result.
- **Instance attributes aren't part of the key**: after changing one that affects results, call `clear_caches(obj)`.
- **Results are returned as stored**, not copied: changing a returned object in place changes the cached result.
- **`[1, 2]` and `(1, 2)` share a result.** If the result depends on which was passed, use `key_converters={"x": container_key}`.
- **Arrays and containers are hashed or converted on every call**, hit or miss.
- **Function caches are unbounded by default** and last as long as the program: bound them with `cache_factory` if they could grow large.
- **With `passthrough`, cached entries record the type of the returned tuple.** If that's a NamedTuple defined inside a function, an instance with cached results can't be pickled.
