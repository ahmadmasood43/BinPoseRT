"""Process pool for CPU stages that parallelise over scenes (D12).

Each spawned worker gets ``OMP_NUM_THREADS=1`` (and the BLAS equivalents): Open3D and numpy
otherwise start a full thread pool per process — measured 15 workers × ~90 threads on a 16-core
machine, load average > 200 — and the stage runs several times slower than single-threaded
workers. The variables are set in the parent only while the pool is being created (spawned
children inherit the environment at start-up) and restored afterwards.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing.pool import Pool

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def scene_pool(n_workers: int) -> Iterator[Pool]:
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    for k in THREAD_VARS:
        os.environ[k] = "1"
    try:
        pool = mp.get_context("spawn").Pool(n_workers)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    with pool:
        yield pool
