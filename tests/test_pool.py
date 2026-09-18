"""The scene process pool survives a worker that dies mid-task (Gamma G15/G20)."""

import os
import signal
from pathlib import Path

import pytest

from binposert.pipeline.pool import map_scenes


def _square(x: int) -> int:
    return x * x


def _square_or_die(x: int, flag_dir: str) -> int:
    """Kills its own worker the first time it sees ``x == 3`` (a segfault stand-in)."""
    flag = Path(flag_dir) / "died"
    if x == 3 and not flag.exists():
        flag.write_text("once")
        os.kill(os.getpid(), signal.SIGKILL)
    return x * x


def test_map_scenes_runs_in_order_and_inline_for_one_worker():
    jobs = [(i,) for i in range(6)]
    assert map_scenes(_square, jobs, n_workers=1) == [i * i for i in range(6)]
    assert map_scenes(_square, jobs, n_workers=3) == [i * i for i in range(6)]


def test_map_scenes_resubmits_after_a_worker_is_killed(tmp_path):
    jobs = [(i, str(tmp_path)) for i in range(6)]
    assert map_scenes(_square_or_die, jobs, n_workers=3) == [i * i for i in range(6)]
    assert (tmp_path / "died").exists()


class RenderCorrupted(RuntimeError):  # matched by name, as the pool does
    pass


def _square_or_corrupt(x: int, flag_dir: str) -> int:
    """Reports corrupted render output the first time it sees ``x == 2``."""
    flag = Path(flag_dir) / "corrupted"
    if x == 2 and not flag.exists():
        flag.write_text("once")
        raise RenderCorrupted("render failed 3 times: index 132151 is out of bounds")
    return x * x


def _fail(x: int) -> int:
    raise ValueError("a real bug in the job")


def test_map_scenes_resubmits_a_job_that_hit_corrupted_render_output(tmp_path):
    jobs = [(i, str(tmp_path)) for i in range(6)]
    assert map_scenes(_square_or_corrupt, jobs, n_workers=3) == [i * i for i in range(6)]
    assert (tmp_path / "corrupted").exists()
    # any other exception is the job's own and propagates
    with pytest.raises(ValueError, match="a real bug"):
        map_scenes(_fail, [(1,), (2,)], n_workers=2)


def test_map_scenes_gives_up_after_retries():
    with pytest.raises(RuntimeError, match="failed after"):
        map_scenes(_die_always, [(1,), (2,)], n_workers=2, retries=2)


def _die_always(x: int) -> int:
    os.kill(os.getpid(), signal.SIGKILL)
    return x


def _thread_env(_: int) -> str:
    return os.environ.get("OMP_NUM_THREADS", "unset")


def test_map_scenes_workers_are_single_threaded(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    assert map_scenes(_thread_env, [(0,), (1,)], n_workers=2) == ["1", "1"]
    assert os.environ["OMP_NUM_THREADS"] == "8"  # the parent's environment is restored
