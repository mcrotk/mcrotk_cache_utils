"""The library without its optional dependencies, numpy and pandas: run in a subprocess with them blocked.

CI also runs the whole suite with neither installed (tests needing them are skipped there).
"""

import subprocess
import sys
import textwrap

import pytest

_BLOCK = """
import sys
for name in {blocked!r}:
    sys.modules[name] = None  # makes "import name" raise ImportError, as if it weren't installed
"""


def run(code, blocked=("numpy", "pandas")):
    """Run ``code`` in a new Python with ``blocked`` modules unavailable; return its output."""
    script = _BLOCK.format(blocked=list(blocked)) + textwrap.dedent(code)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_core_works_without_numpy_and_pandas():
    output = run("""
        import copy
        from mcrotk_cache_utils import cached_function, cached_method, clear_caches, container_key, copy_instance

        class C:
            def __init__(self):
                self.calls = 0

            @cached_method(key_converters={"xs": container_key})
            def f(self, xs, *args, **kwargs):
                self.calls += 1
                return len(xs)

        c = C()
        c.f([1, [2]], 3, k=(1, 2))
        c.f([1, [2]], 3, k=(1, 2))
        assert c.calls == 1
        assert copy.copy(c).f([1, [2]], 3, k=(1, 2)) == 2
        assert copy_instance(c, "copy").f([1, [2]], 3, k=(1, 2)) == 2
        clear_caches(c)

        calls = []

        @cached_function
        def g(x, *args, **kwargs):
            calls.append(1)
            return x

        g(1, (2, 3), y="a")
        g(1, (2, 3), y="a")
        assert len(calls) == 1
        assert container_key({"a": [1]}) == container_key({"a": [1]}) != container_key({"a": (1,)})
        print("ok")
    """)
    assert output == "ok"


@pytest.mark.parametrize(
    "code, message",
    [
        ("rng_cached_method(lambda self, rng=None: (1, rng))", "rng_cached_method needs numpy"),
        ("rng_cached_function(lambda rng=None: (1, rng))", "rng_cached_function needs numpy"),
        ("array_key([1])", "array_key needs numpy"),
        ("dataframe_key({'x': [1]})", "dataframe_key needs pandas"),
    ],
)
def test_helpful_errors_without_numpy_and_pandas(code, message):
    output = run(f"""
        from mcrotk_cache_utils import array_key, dataframe_key, rng_cached_function, rng_cached_method

        try:
            {code}
        except ImportError as e:
            print(e)
    """)
    assert output.startswith(message) and "pip install 'mcrotk-cache-utils[" in output


def test_numpy_without_pandas():
    pytest.importorskip("numpy")
    output = run(
        """
        import numpy as np
        from mcrotk_cache_utils import cached_function, dataframe_key, rng_cached_function

        calls = []

        @cached_function
        def total(arr):
            calls.append(1)
            return arr.sum()

        total(np.arange(3))
        total(np.arange(3))
        assert len(calls) == 1

        @rng_cached_function
        def draw(n, rng=None):
            rng = np.random.default_rng(rng)
            return rng.random(n), rng

        draw(2, rng=0)
        try:
            dataframe_key(None)
        except ImportError as e:
            print(e)
        """,
        blocked=("pandas",),
    )
    assert output.startswith("dataframe_key needs pandas")
