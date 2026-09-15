"""The adapters' shared plumbing runs on CPU; test it against the mini fixture."""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from adapters.common import (  # noqa: E402
    AdapterArgs,
    bbox_xywh,
    ensure_symlink,
    rgb_path,
    target_images,
    target_objects,
    write_adapter_info,
)

FIX = REPO / "tests" / "fixtures" / "mini_bop"


def _args(tmp_path: Path, **kw) -> AdapterArgs:
    base = dict(
        dataset_root=FIX,
        split="test",
        out=tmp_path / "out",
        params={"a": 1},
        targets=None,
        detections=None,
        upstream=tmp_path / "upstream",
        checkpoints=tmp_path / "ckpt",
        work=tmp_path / "work",
        scenes=None,
        max_images=None,
        device="cpu",
    )
    base.update(kw)
    return AdapterArgs(**base)


def test_target_images_without_targets_enumerates_the_split(tmp_path):
    args = _args(tmp_path)
    assert target_images(args) == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert target_images(_args(tmp_path, scenes=[2])) == [(2, 0), (2, 1)]
    assert target_images(_args(tmp_path, max_images=3)) == [(1, 0), (1, 1), (2, 0)]
    assert target_objects(args) is None
    assert rgb_path(args, 1, 0).name == "000000.png"


def test_target_images_honour_targets_file(tmp_path):
    targets = [
        {"scene_id": 2, "im_id": 1, "obj_id": 5, "inst_count": 1},
        {"scene_id": 2, "im_id": 1, "obj_id": 1, "inst_count": 2},
        {"scene_id": 1, "im_id": 0, "obj_id": 1, "inst_count": 2},
    ]
    tf = tmp_path / "targets.json"
    tf.write_text(json.dumps(targets))
    args = _args(tmp_path, targets=tf)
    assert target_images(args) == [(1, 0), (2, 1)]
    assert target_objects(args) == {(2, 1): {5: 1, 1: 2}, (1, 0): {1: 2}}


def test_symlink_and_adapter_info(tmp_path):
    target = tmp_path / "real.txt"
    target.write_text("x")
    link = tmp_path / "links" / "a.txt"
    ensure_symlink(link, target)
    ensure_symlink(link, target)  # idempotent
    assert link.is_symlink() and link.read_text() == "x"
    other = tmp_path / "other.txt"
    other.write_text("y")
    ensure_symlink(link, other)  # re-pointed
    assert link.read_text() == "y"

    args = _args(tmp_path)
    write_adapter_info(args, "stub", {"n": 3})
    info = json.loads((args.out / "adapter.json").read_text())
    assert info["adapter"] == "stub" and info["n"] == 3 and info["params"] == {"a": 1}
    assert info["upstream_commit"] == "unknown"  # not a git checkout


def test_bbox_xywh():
    m = np.zeros((10, 12), bool)
    m[2:5, 3:9] = True
    assert bbox_xywh(m) == (3, 2, 6, 3)
    assert bbox_xywh(np.zeros((4, 4), bool)) == (0, 0, 0, 0)
