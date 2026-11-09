"""Wall-time baseline for expectation_sensitivity.

Prints one float: milliseconds for one call on a 100-state dense transition
matrix, after a warmup call in the same process. BLAS/OpenMP threads are
pinned to 1.

Workload: sensitivity of a random observable's expectation value to every
entry of a 100-state row-stochastic transition matrix. That is the dense
MSM analysis path (how an expectation moves when the transition matrix is
perturbed) at a state count used for small discretized systems.
"""

import time

import numpy as np
from threadpoolctl import threadpool_limits

from deeptime.markov.tools.analysis import expectation_sensitivity


def _transition_matrix(n_states: int, rng: np.random.Generator) -> np.ndarray:
    weights = rng.random((n_states, n_states)) + 2.0 * np.eye(n_states)
    return weights / weights.sum(axis=1, keepdims=True)


def main() -> None:
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(0)
        n_states = 100
        transition = _transition_matrix(n_states, rng)
        observable = rng.normal(size=n_states)

        def run():
            return expectation_sensitivity(transition, observable)

        run()
        started = time.perf_counter()
        result = run()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result.shape != (n_states, n_states):
            raise RuntimeError(f"unexpected sensitivity shape {result.shape}")
        print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
