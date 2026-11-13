"""Element-wise confidence intervals on posterior samples.

Bayesian MSM and HMM summaries, and augmented-MSM expectation checks, pass a
stack of samples into ``confidence_interval``. For a matrix-valued quantity
the implementation sorts every column from Python.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from deeptime.util.stats import confidence_interval


def setup():
    # 80 posterior draws of a 60-by-60 observable (transition matrix or
    # correlation block on a modest MSM).
    rng = np.random.default_rng(1)
    samples = rng.normal(loc=0.0, scale=1.0, size=(80, 60, 60))
    return samples


def workload(samples):
    return confidence_interval(samples, conf=0.95)


def main():
    samples = setup()
    lower, upper = workload(samples)
    print(f"{lower.shape} {float(lower.mean()):.8e} {float(upper.mean()):.8e}")


if __name__ == "__main__":
    main()
