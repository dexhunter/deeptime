"""Sliding windows along a trajectory.

``sliding_window`` is the public helper for fixed-radius views of a time
series (feature windows, finite-difference stencils, local statistics).
Each step copies the deque into a new ndarray.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from deeptime.util.data import sliding_window


def setup():
    # One continuous trajectory, about 1.2e5 frames, window radius 3
    # (7 frames), the size used when scanning a feature time series.
    return np.linspace(0.0, 1.0, 120_000)


def workload(sequence):
    total = 0.0
    count = 0
    for window in sliding_window(sequence, radius=3, fixed_width=True):
        total += float(window[0] + window[-1])
        count += 1
    return total, count


def main():
    sequence = setup()
    total, count = workload(sequence)
    print(f"{count} {total:.8e}")


if __name__ == "__main__":
    main()
