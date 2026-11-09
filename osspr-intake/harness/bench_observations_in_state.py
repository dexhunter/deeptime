"""Wall-time baseline for observations_in_state.

Prints one float: milliseconds to gather the observations assigned to one
hidden state, after a warmup call.

Workload: 2000 trajectories of 500 frames, four hidden states. Bayesian HMM
sampling calls this once per hidden state on every Gibbs update of the output
model, and multi-trajectory MD sets of this size are ordinary.
"""

import time

import numpy as np

from deeptime.markov.hmm._util import observations_in_state


def main() -> None:
    rng = np.random.default_rng(0)
    n_trajectories = 2000
    length = 500
    n_hidden = 4
    hidden = [rng.integers(0, n_hidden, size=length) for _ in range(n_trajectories)]
    observed = [rng.normal(size=length) for _ in range(n_trajectories)]
    state = 1

    def run():
        return observations_in_state(hidden, observed, state)

    run()
    started = time.perf_counter()
    gathered = run()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if gathered.ndim != 1 or gathered.size == 0:
        raise RuntimeError(f"unexpected gathered shape {gathered.shape}")
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
