"""Wall-time baseline for _pcca_connected_isa.

Prints one float: milliseconds for one inner-simplex representative search,
after a warmup call. BLAS/OpenMP threads are pinned to 1.

Workload: the first 10 eigenvectors of a 6000-state discretization (the
stationary eigenvector is the required constant column; the other columns are
the slow modes handed to PCCA+). This is the representative-selection step
inside PCCA+ before the membership rotation is optimized.
"""

import time

import numpy as np
from threadpoolctl import threadpool_limits

from deeptime.markov.tools.analysis.dense._pcca import _pcca_connected_isa


def _eigenvectors(n_states: int, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=(n_states, n_clusters))
    vectors[:, 0] = 1.0
    return vectors


def main() -> None:
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(0)
        n_states = 6000
        n_clusters = 10
        eigenvectors = _eigenvectors(n_states, n_clusters, rng)

        def run():
            return _pcca_connected_isa(eigenvectors, n_clusters)

        run()
        started = time.perf_counter()
        chi, rotation = run()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if chi.shape != (n_states, n_clusters) or rotation.shape != (n_clusters, n_clusters):
            raise RuntimeError(f"unexpected PCCA shapes {chi.shape} {rotation.shape}")
        print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
