#!/usr/bin/env bash
# Minimal source patches applied to the MegaPose checkout (idempotent).
#  1. The panda3d render workers are forked children that hand renders back through
#     torch.multiprocessing queues as shared-memory tensors; on this stack (torch 2.5, python 3.10)
#     share_memory_() deadlocks in the child. Pass plain numpy arrays through the pipe instead and
#     build the tensors on the parent side (a few MB per render, negligible next to the GPU work).
set -euo pipefail
src="${1:?upstream dir}"
python3 - "$src" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "src/megapose/panda3d_renderer/panda3d_batch_renderer.py"
s = p.read_text()
marker = "# binposert: numpy through the queue"
if marker in s:
    print("already patched", p)
    sys.exit(0)
old_w = """        output = RenderOutput(
            data_id=render_args.data_id,
            rgb=torch.tensor(renderings_.rgb).share_memory_(),
            normals=torch.tensor(renderings_.normals).share_memory_()
            if render_args.render_normals
            else None,
            depth=torch.tensor(renderings_.depth).share_memory_()
            if render_args.render_depth
            else None,
        )"""
new_w = """        output = RenderOutput(  # binposert: numpy through the queue
            data_id=render_args.data_id,
            rgb=np.ascontiguousarray(renderings_.rgb),
            normals=np.ascontiguousarray(renderings_.normals)
            if render_args.render_normals
            else None,
            depth=np.ascontiguousarray(renderings_.depth)
            if render_args.render_depth
            else None,
        )"""
old_p = """            list_rgbs[data_id] = renders.rgb
            if render_depth:
                list_depths[data_id] = renders.depth
            if render_normals:
                list_normals[data_id] = renders.normals"""
new_p = """            list_rgbs[data_id] = torch.from_numpy(renders.rgb)
            if render_depth:
                list_depths[data_id] = torch.from_numpy(renders.depth)
            if render_normals:
                list_normals[data_id] = torch.from_numpy(renders.normals)"""
assert old_w in s and old_p in s, "upstream source changed; patch needs updating"
p.write_text(s.replace(old_w, new_w).replace(old_p, new_p))
print("patched", p)
PY
