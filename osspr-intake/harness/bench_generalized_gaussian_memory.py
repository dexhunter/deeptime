"""Peak-memory baseline for GeneralizedGaussianKernel.apply.

Prints one integer: the extra peak resident memory, in bytes, of a single
apply() call. The figure is the rise in VmHWM (Linux high-water mark) across
that call, after the input blocks are already allocated. A later change that
keeps the same Gram matrix without the (n_frames, n_frames, n_features)
broadcast temporary should move this number.

Workload: two blocks of 900 frames in 24 dimensions, the size of a kernel
Gram chunk on a featurized trajectory (distances or contacts), with a
positive per-coordinate bandwidth.
"""

import gc

import numpy as np
from threadpoolctl import threadpool_limits

from deeptime.kernels import GeneralizedGaussianKernel


def _vmhwm_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmHWM missing from /proc/self/status")


def main() -> None:
    # One BLAS/OpenMP thread so the peak is not mixed with per-thread workspaces.
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(0)
        n_frames = 900
        n_features = 24
        data_1 = np.ascontiguousarray(rng.standard_normal((n_frames, n_features)), dtype=np.float64)
        data_2 = np.ascontiguousarray(rng.standard_normal((n_frames, n_features)), dtype=np.float64)
        bandwidths = np.linspace(0.4, 1.6, n_features)
        kernel = GeneralizedGaussianKernel(bandwidths)
        gc.collect()
        before = _vmhwm_bytes()
        gram = kernel.apply(data_1, data_2)
        # Keep the result alive until the high-water mark is read.
        after = _vmhwm_bytes()
        if gram.shape != (n_frames, n_frames):
            raise RuntimeError(f"unexpected Gram shape {gram.shape}")
        print(after - before)


if __name__ == "__main__":
    main()
