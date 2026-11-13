"""Dense flux coarse-graining on a few hundred microstates.

Transition-path theory coarse-grains a state-to-state flux into a handful of
sets (reactants, intermediates, products). The nested Python loop in
``coarsegrain`` rebuilds each block with fancy indexing.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from deeptime.markov.tools.flux import coarsegrain


def setup():
    rng = np.random.default_rng(0)
    n_states = 900
    width = 15
    n_sets = 60
    flux = rng.random((n_states, n_states))
    # Net fluxes are skew-symmetric; a symmetric non-negative matrix is the
    # gross-flux shape this routine is called with.
    flux = np.ascontiguousarray(0.5 * (flux + flux.T))
    sets = [np.arange(i * width, (i + 1) * width) for i in range(n_sets)]
    return flux, sets


def workload(flux, sets):
    return coarsegrain(flux, sets)


def main():
    flux, sets = setup()
    coarse = workload(flux, sets)
    # Consume the result so the call cannot be skipped.
    print(f"{coarse.shape[0]} {float(coarse.sum()):.8e}")


if __name__ == "__main__":
    main()
