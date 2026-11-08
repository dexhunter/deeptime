"""Frozen baselines for osspr intake. Prints one JSON object.

Run from the repository root with deeptime importable:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        NUMEXPR_NUM_THREADS=1 taskset -c 0 \\
        python osspr-intake/harness/bench.py <target>
"""
import json
import statistics
import sys
import time

import numpy as np

WARMUP = 3
REPEATS = 11


def _fn(target):
    if target == "polynomial_kernel_apply":
        from deeptime.kernels import PolynomialKernel

        rng = np.random.default_rng(0)
        data_1 = rng.standard_normal((900, 24))
        data_2 = rng.standard_normal((900, 24))
        kernel = PolynomialKernel(3, 1.0)

        def fn():
            return kernel.apply(data_1, data_2)

        return fn

    if target == "kernel_apply":
        from deeptime.kernels import GaussianKernel, PolynomialKernel

        rng = np.random.default_rng(1)
        data_1 = rng.standard_normal((180, 8))
        data_2 = rng.standard_normal((180, 8))
        # ProductKernel does not override apply, so this is Kernel.apply.
        kernel = PolynomialKernel(3, 1.0) * GaussianKernel(1.5)

        def fn():
            return kernel.apply(data_1, data_2)

        return fn

    if target == "split_sequences_singletraj":
        from deeptime.markov.tools.estimation.sparse.effective_counts import (
            _split_sequences_singletraj,
        )

        rng = np.random.default_rng(2)
        dtraj = rng.integers(0, 40, size=1_500_000)

        def fn():
            return _split_sequences_singletraj(dtraj, 40, 1)

        return fn

    if target == "finite_difference_operator_midpoints":
        from deeptime.util.diff import finite_difference_operator_midpoints

        xs = np.linspace(0.0, 10.0, 8000)

        def fn():
            return finite_difference_operator_midpoints(xs, k=1, window_radius=5)

        return fn

    if target == "error_perturbation_cov":
        from deeptime.markov.tools.estimation.dense.covariance import error_perturbation_cov

        rng = np.random.default_rng(3)
        counts = rng.random((100, 100)) + np.eye(100)
        sensitivity = rng.standard_normal((6, 100, 100))

        def fn():
            return error_perturbation_cov(counts, sensitivity)

        return fn

    if target == "statistical_inefficiency":
        from deeptime.util.stats import statistical_inefficiency

        rng = np.random.default_rng(4)
        series = rng.standard_normal(12000)

        def fn():
            return statistical_inefficiency([series], truncate_acf=False)

        return fn

    raise SystemExit(f"unknown target {target}")


def main(argv):
    if len(argv) != 2:
        raise SystemExit("usage: bench.py <target>")
    target = argv[1]
    fn = _fn(target)
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    median = statistics.median(samples)
    noise = 100.0 * statistics.pstdev(samples) / median
    json.dump(
        {
            "target": target,
            "median_ms": median,
            "noise_pct": noise,
            "samples_ms": samples,
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
