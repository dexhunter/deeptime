"""Peak-resident-memory baselines for deeptime kernel, covariance, and dense Markov paths.

Each invocation prints one integer: the rise in VmHWM, in bytes, across a single
call. Inputs are allocated before the measurement. Run the process once per
sample; VmHWM never goes down inside a process.

Targets
-------
polynomial_kernel
    PolynomialKernel.apply on two 700 x 48 blocks. The broadcast product
    ri * rj is a 700 x 700 x 48 temporary.
moments_xx
    moments_XX on 200_000 x 80 frames with per-frame weights, dense mode.
tmatrix_cov
    tmatrix_cov on a 200 x 200 count matrix (the full (200, 200, 200) tensor).
"""

import gc
import sys

import numpy as np
from threadpoolctl import threadpool_limits


def _vmhwm_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM missing from /proc/self/status")


def _measure(fn):
    gc.collect()
    before = _vmhwm_bytes()
    out = fn()
    after = _vmhwm_bytes()
    return after - before, out


def _polynomial_kernel():
    from deeptime.kernels import PolynomialKernel

    rng = np.random.default_rng(0)
    n_frames = 700
    n_features = 48
    data_1 = np.ascontiguousarray(rng.standard_normal((n_frames, n_features)))
    data_2 = np.ascontiguousarray(rng.standard_normal((n_frames, n_features)))
    kernel = PolynomialKernel(degree=3, inhomogeneity=1.0)
    delta, gram = _measure(lambda: kernel.apply(data_1, data_2))
    if gram.shape != (n_frames, n_frames):
        raise RuntimeError(f"unexpected Gram shape {gram.shape}")
    if not np.isfinite(gram).all():
        raise RuntimeError("polynomial Gram is not finite")
    return delta


def _moments_xx():
    from deeptime.covariance.util import moments_XX

    rng = np.random.default_rng(1)
    n_frames = 200_000
    n_features = 80
    data = np.ascontiguousarray(rng.standard_normal((n_frames, n_features)))
    weights = np.ascontiguousarray(rng.random(n_frames) + 0.1)
    delta, moments = _measure(
        lambda: moments_XX(
            data,
            remove_mean=False,
            modify_data=False,
            weights=weights,
            sparse_mode="dense",
        )
    )
    weight, column_sum, second = moments
    if second.shape != (n_features, n_features):
        raise RuntimeError(f"unexpected second-moment shape {second.shape}")
    if column_sum.shape != (n_features,) or not np.isfinite(weight):
        raise RuntimeError("moments_XX returned a malformed sum")
    if not np.isfinite(second).all():
        raise RuntimeError("second moment is not finite")
    return delta


def _tmatrix_cov():
    from deeptime.markov.tools.estimation.dense.covariance import tmatrix_cov

    rng = np.random.default_rng(2)
    n_states = 200
    counts = np.ascontiguousarray(rng.random((n_states, n_states)))
    delta, cov = _measure(lambda: tmatrix_cov(counts))
    if cov.shape != (n_states, n_states, n_states):
        raise RuntimeError(f"unexpected covariance shape {cov.shape}")
    if not np.isfinite(cov).all():
        raise RuntimeError("tmatrix_cov is not finite")
    return delta


_TARGETS = {
    "polynomial_kernel": _polynomial_kernel,
    "moments_xx": _moments_xx,
    "tmatrix_cov": _tmatrix_cov,
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in _TARGETS:
        names = ", ".join(sorted(_TARGETS))
        raise SystemExit(f"usage: python osspr-intake/harness/peak_mem.py <{names}>")
    with threadpool_limits(limits=1):
        print(_TARGETS[sys.argv[1]]())


if __name__ == "__main__":
    main()
