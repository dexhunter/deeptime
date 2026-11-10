"""Allocation-count baseline for a dense Markov correlation propagation.

Prints one integer: the number of malloc/calloc/realloc calls of at least
8192 bytes during a single time_correlation_direct_by_mtx_vec_prod call.
The counter is a small LD_PRELOAD shim compiled from alloccount.c next to
this file. Counting is enabled only around the call.

Workload: one 1500-state row-stochastic matrix propagated for 2500 steps.
Each matrix-vector product allocates a fresh length-1500 vector.
"""

import ctypes
import gc
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

_MIN_BYTES = 8192
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "alloccount.c"
_SO = Path("/tmp/ossp-intake-deeptime-alloccount.so")


def _ensure_preloaded() -> None:
    if os.environ.get("OSSPR_ALLOC_PRELOADED") == "1":
        return
    subprocess.check_call(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", str(_SO), str(_SRC), "-ldl"],
        stdout=subprocess.DEVNULL,
    )
    env = os.environ.copy()
    env["OSSPR_ALLOC_PRELOADED"] = "1"
    env["LD_PRELOAD"] = str(_SO)
    env["ALLOC_COUNT_MIN"] = str(_MIN_BYTES)
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], env)


def _count() -> int:
    lib = ctypes.CDLL(str(_SO))
    lib.alloc_counter_get.restype = ctypes.c_ulonglong
    from deeptime.markov.tools.analysis.dense._correlations import (
        time_correlation_direct_by_mtx_vec_prod,
    )

    rng = np.random.default_rng(5)
    n_states = 1500
    n_steps = 2500
    transition = rng.random((n_states, n_states))
    transition /= transition.sum(axis=1, keepdims=True)
    stationary = np.full(n_states, 1.0 / n_states)
    obs1 = rng.standard_normal(n_states)
    obs2 = rng.standard_normal(n_states)
    gc.collect()
    lib.alloc_counter_reset()
    lib.alloc_counter_enable(1)
    corr = time_correlation_direct_by_mtx_vec_prod(
        transition, stationary, obs1, obs2, time=n_steps
    )
    lib.alloc_counter_enable(0)
    if not np.isfinite(corr):
        raise RuntimeError("correlation is not finite")
    return int(lib.alloc_counter_get())


def main() -> None:
    _ensure_preloaded()
    with threadpool_limits(limits=1):
        print(_count())


if __name__ == "__main__":
    main()
