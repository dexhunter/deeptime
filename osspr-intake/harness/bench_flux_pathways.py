"""Dominant pathways on a grid flux network.

Transition-path theory decomposes the net flux from one corner of a
two-dimensional grid to the opposite corner. ``pathways`` repeatedly
rebuilds the residual graph and searches it for the next bottleneck.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import csr_matrix

from deeptime.markov.msm import MarkovStateModel
from deeptime.markov.tools.flux.pathways import pathways


def _grid_transition_matrix(side, stay=0.6):
    n_states = side * side
    transition = np.zeros((n_states, n_states))

    def index(i, j):
        return i * side + j

    for i in range(side):
        for j in range(side):
            neighbors = []
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < side and 0 <= nj < side:
                    neighbors.append(index(ni, nj))
            state = index(i, j)
            transition[state, state] = stay
            weight = (1.0 - stay) / len(neighbors)
            for neighbor in neighbors:
                transition[state, neighbor] = weight
    return transition


def setup():
    side = 10
    transition = _grid_transition_matrix(side)
    model = MarkovStateModel(transition, reversible=True)
    # Source is the first cell, target is the opposite corner.
    flux = model.reactive_flux([0], [side * side - 1]).net_flux
    return csr_matrix(flux), [0], [side * side - 1]


def workload(flux, source, target):
    # Half of the total flux, capped, matches a typical pathway report.
    return pathways(flux, source, target, fraction=0.5, maxiter=30)


def main():
    flux, source, target = setup()
    paths, capacities = workload(flux, source, target)
    print(f"{len(paths)} {float(np.sum(capacities)):.8e}")


if __name__ == "__main__":
    main()
