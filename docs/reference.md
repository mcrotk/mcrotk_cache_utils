# API reference

Everything is imported from the package: `from mcrotk_cache_utils import cached_method, ...`. For explanations and examples, see the [user guide](guide.md).

**Decorators**
- [`cached_method`](#cached_method): cache a method, per instance.
- [`cached_function`](#cached_function): cache a function, static method or class method.
- [`rng_cached_method`](#rng_cached_method), [`rng_cached_function`](#rng_cached_function): the same, for methods and functions that take and return a random number generator.

**Instances and caches**
- [`clear_caches`](#clear_caches): empty an instance's caches.
- [`copy_instance`](#copy_instance): copy an instance, choosing what the copy gets in its caches.
- [`cache_controls`](#cache_controls): typed access to a cache's attributes (`cache_clear`, `cache_info`, ...).
- [`CacheCopyWarning`](#cachecopywarning)

**Key converters**
- [`container_key`](#container_key): lists, tuples, dicts and sets.
- [`array_key`](#array_key): numpy arrays.
- [`dataframe_key`](#dataframe_key), [`series_key`](#series_key), [`index_key`](#index_key): pandas objects.

Every decorator can be used bare (`@cached_method`) or with options (`@cached_method(omit="verbose")`). All options are keyword-only.

---

## `cached_method`

```python
cached_method(*, cache_factory=dict, omit=(), passthrough=None, key_converters=None,
              lock=None, condition=None, info=False, on_copy=None)
```

Cache a method's results, with a separate cache for each instance, created on its first call and stored in its `__dict__`. For ordinary methods defined in a class body; for functions, static methods and class methods, use [`cached_function`](#cached_function). Built on `cachetools.cachedmethod`.

The key is made from the arguments bound to the signature with defaults filled in, without `self`. numpy arrays and lists, dicts and sets in it are converted automatically ([guide](guide.md#unhashable-arguments)).

| Option | Default | Description |
|---|---|---|
| `cache_factory` | `dict` | Makes each instance's cache: any `MutableMapping`, e.g. `lambda: cachetools.LRUCache(maxsize=128)`. The default is unbounded. |
| `omit` | `()` | Arguments to leave out of the key: a name or position, or a list of them. Positions count from 0 in the `def` line (`self` is 0); negative ones count from the end. Only for arguments that don't affect the result. |
| `passthrough` | `None` | `{argument: output_position}` or `{argument: (output_position, transform)}`, for methods that take an argument and return it in a tuple. The argument is left out of the key; on a hit, the stored result is returned with the caller's argument (after `transform`) at `output_position`. The method must return a tuple. ([guide](guide.md#passing-arguments-through)) |
| `key_converters` | `None` | `{argument: function}`: the function's result goes in the key in place of the argument. Replaces the automatic handling of arrays and containers for that argument. |
| `lock` | `None` | A function of `self` returning a lock (e.g. `lambda self: self.lock`), held while reading and writing the cache. Needed for cachetools' caches used from several threads. |
| `condition` | `None` | A function of `self` returning a `threading.Condition`, which also stops two threads from computing the same result. |
| `info` | `False` | Count hits and misses, for `cache_controls(obj.method).cache_info()`. |
| `on_copy` | `"empty"` | What a copy of the instance gets in this method's cache: `"empty"`, `"copy"` (the cached results) or `"manual"` (the class's own `__copy__` decides). ([guide](guide.md#copying-and-pickling)) |

Option errors are raised when the decorator is applied; an unhashable argument raises `TypeError` naming it. The decorator also gives the class `__copy__` and `__deepcopy__` methods that apply `on_copy`, unless the class has its own.

## `cached_function`

```python
cached_function(*, cache_factory=dict, omit=(), passthrough=None, key_converters=None,
                lock=None, condition=None, info=False)
```

Cache a function's results in one cache, made when the function is decorated and kept as long as the function exists. Built on `cachetools.cached`. For static and class methods, put `@staticmethod` or `@classmethod` above it; a class method's `cls` is part of the key (`omit="cls"` to share results between subclasses).

Options as for [`cached_method`](#cached_method), except:

| Option | Description |
|---|---|
| `lock` | A lock object (e.g. `threading.Lock()`), not a function. |
| `condition` | A `threading.Condition`, not a function. |

There's no `on_copy`. The cache is reached with `cache_controls(f)`.

## `rng_cached_method`

```python
rng_cached_method(*, rng_arg="rng", rng_output=-1, passthrough=None, **options)
```

Cache a method that takes a numpy random number generator and returns it, leaving the generator out of the key. The same as `cached_method(passthrough={rng_arg: (rng_output, numpy.random.default_rng)})`: on a hit, the stored result is returned with the caller's generator, or a generator made from the caller's seed or `None`. Needs numpy. ([guide](guide.md#random-number-generators))

| Option | Default | Description |
|---|---|---|
| `rng_arg` | `"rng"` | The generator argument, by name or position. |
| `rng_output` | `-1` | Its position in the returned tuple. |
| `passthrough` | `None` | Other arguments to pass through (not the generator). |
| `**options` | | Any of [`cached_method`](#cached_method)'s options. |

## `rng_cached_function`

```python
rng_cached_function(*, rng_arg="rng", rng_output=-1, passthrough=None, **options)
```

[`rng_cached_method`](#rng_cached_method) for functions, static methods and class methods, built on [`cached_function`](#cached_function) (whose options it takes).

---

## `clear_caches`

```python
clear_caches(obj, *methods: str) -> None
```

Empty `obj`'s caches for the named cached methods, or for all of them if none are named. Only this library's caches are touched, and only `obj`'s. Raises `ValueError` for a name that isn't a cached method.

## `copy_instance`

```python
copy_instance(obj, caches=None, *, deep=False, memo=None)
```

Copy an instance with cached methods, like `copy.copy` (or `copy.deepcopy` with `deep=True`), choosing what the copy gets in its caches. It never calls the class's own `__copy__`/`__deepcopy__`, so it can be used inside them. Returns the copy, of the same type.

| Parameter | Description |
|---|---|
| `caches` | `"empty"` or `"copy"` for every cached method, or `None` (default) to follow each method's `on_copy` (`"manual"` ones emptied). |
| `deep` | Make a deep copy. |
| `memo` | Inside your own `__deepcopy__(self, memo)`, pass its `memo` (implies `deep=True`). |

## `cache_controls`

```python
cache_controls(cached) -> CacheControls
```

The cache attributes of a cached function (`cache_controls(fit)`, including static and class methods accessed on the class) or of a cached method accessed on an instance (`cache_controls(obj.fit)`), typed for type checkers. Returns the same object. Raises `TypeError` for anything else.

| Attribute | Description |
|---|---|
| `cache_clear()` | Empty the cache (and reset the hit/miss counts). |
| `cache_info()` | `CacheInfo(hits, misses, maxsize, currsize)`. Only with `info=True`; otherwise `AttributeError`. |
| `cache` | The cache itself. |
| `cache_key` | The function that makes the keys. |
| `cache_lock`, `cache_condition` | The lock and condition, if given. |

The same attributes can be read directly (`fit.cache_clear()`), but type checkers report them as unknown.

## `CacheCopyWarning`

A `UserWarning`, given once per class when a class with cached methods defines its own `__copy__` or `__deepcopy__`, so the decorator's copy handling isn't used. Silence it with `on_copy="manual"` on the cached methods (after checking the class's method handles the caches, e.g. with `copy_instance`), or with `warnings.filterwarnings("ignore", category=CacheCopyWarning)`.

---

## `container_key`

```python
container_key(value, *, type_tags=True)
```

A hashable equivalent of `value`: lists and tuples become tuples, sets and frozensets become frozensets, and dicts become frozensets of their `(key, value)` pairs, recursively. numpy arrays inside are keyed as by [`array_key`](#array_key); anything else is left as it is.

With `type_tags=True` (the default), each container's exact type is kept, so `[2, 3]` and `(2, 3)` get different keys, as do a NamedTuple and a plain tuple. With `False`, containers of the same kind share keys, and so do a dict and the set of its `(key, value)` pairs. Unhashable containers in arguments get `type_tags=False` automatically.

## `array_key`

```python
array_key(array)
```

A hashable stand-in for a numpy array: its shape, dtype and a hash of its data (BLAKE2b, in C order, so equal values in any memory layout get the same key). Applied automatically to `ndarray` and `memmap` arguments; use it in `key_converters` for other array types whose data alone determines the result. Raises `TypeError` for object arrays.

## `dataframe_key`

```python
dataframe_key(df, *, index=True)
```

A hashable stand-in for a pandas DataFrame: its column labels, dtypes and values, and with `index=True` its index (as [`index_key`](#index_key)). Two DataFrames get the same key if these match exactly, in the same row and column order. Use sparingly ([guide](guide.md#pandas-objects)). Needs pandas.

## `series_key`

```python
series_key(s, *, name=True, index=True)
```

The same for a pandas Series: its dtype and values, with `name=True` its name, and with `index=True` its index.

Values are hashed with `pandas.util.hash_pandas_object`, except in object columns holding anything other than strings: pandas would hash those by their text (so `1` and `"1"` would match), so they go in the key themselves, with their types.

## `index_key`

```python
index_key(idx)
```

The same for a pandas Index or MultiIndex: its level names, dtypes and values, and whether it's a MultiIndex.
