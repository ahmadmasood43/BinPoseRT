"""Process pool for CPU stages that parallelise over scenes (D12).

Each spawned worker gets ``OMP_NUM_THREADS=1`` (and the BLAS equivalents): Open3D and numpy
otherwise start a full thread pool per process — measured 15 workers × ~90 threads on a 16-core
machine, load average > 200 — and the stage runs several times slower than single-threaded
workers. The variables are set in the parent only while the pool is being created (spawned
children inherit the environment at start-up) and restored afterwards.

:func:`map_scenes` is what stages call. It uses ``concurrent.futures`` rather than
``multiprocessing.Pool``: a worker that dies from a signal (Open3D's raycast corruption can
segfault, Gamma G15) makes ``Pool.starmap`` wait forever for the lost task, whereas the executor
raises ``BrokenProcessPool``, after which the scenes that did not finish are resubmitted to a fresh
pool (up to ``retries`` times) and only then does the stage fail. A job that raises
``RenderCorrupted`` (the renderer validated and retried its cast three times and still saw a
flipped bit in an index — G21's hardware suspicion) is treated the same way: its worker's memory,
not the job, is what failed, so it is re-run in a fresh process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from multiprocessing.pool import Pool
from typing import Any

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# exceptions (by name, so this module imports nothing heavy) that mean the *process* is
# corrupted rather than the job wrong: the job is resubmitted to a fresh pool
RETRY_EXCEPTIONS = ("RenderCorrupted",)
log = logging.getLogger("binposert.pipeline")


@contextmanager
def _single_thread_env() -> Iterator[None]:
    saved = {k: os.environ.get(k) for k in THREAD_VARS}
    for k in THREAD_VARS:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def scene_pool(n_workers: int) -> Iterator[Pool]:
    """A plain ``multiprocessing.Pool`` of single-threaded spawned workers (kept for callers that
    need the Pool API; prefer :func:`map_scenes`, which survives a dying worker)."""
    with _single_thread_env():
        pool = mp.get_context("spawn").Pool(n_workers)
    with pool:
        yield pool


def map_scenes(
    fn: Callable[..., Any],
    jobs: Sequence[tuple[Any, ...]],
    n_workers: int,
    retries: int = 3,
) -> list[Any]:
    """``[fn(*job) for job in jobs]`` over single-threaded spawned workers, in job order.

    With ``n_workers <= 1`` or a single job the work runs in this process. A job whose worker
    dies is resubmitted (fresh pool) up to ``retries`` times; jobs that finished are not redone.
    """
    results: list[Any] = [None] * len(jobs)
    pending = list(range(len(jobs)))
    if n_workers <= 1 or len(jobs) <= 1:
        return [fn(*job) for job in jobs]
    for attempt in range(retries):
        if not pending:
            break
        # the executor spawns its workers lazily, on the first submit: the single-thread
        # environment must therefore cover the submits, not only the constructor
        broken = False
        with _single_thread_env():
            executor = ProcessPoolExecutor(
                max_workers=min(n_workers, len(pending)), mp_context=mp.get_context("spawn")
            )
            try:
                futures = {executor.submit(fn, *jobs[i]): i for i in pending}
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        corrupted = 0
        try:
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except BrokenProcessPool:
                    broken = True
                    continue
                except Exception as e:  # noqa: BLE001 — only the named corruption is retried
                    if type(e).__name__ not in RETRY_EXCEPTIONS:
                        raise
                    # the worker survived but its memory did not (Gamma G15 / G21): the job is
                    # re-run in a fresh process instead of failing the stage
                    corrupted += 1
                    broken = True
                    continue
                pending.remove(i)
        finally:
            executor.shutdown(wait=not broken, cancel_futures=True)
        if broken and pending:
            log.warning(
                "%s; resubmitting %d unfinished job(s) in a fresh pool (attempt %d/%d)",
                f"{corrupted} job(s) hit corrupted render output"
                if corrupted
                else "a worker process died",
                len(pending),
                attempt + 2,
                retries,
            )
    if pending:
        raise RuntimeError(f"{len(pending)} scene job(s) failed after {retries} attempts")
    return results
