#!/usr/bin/env bash
# Minimal source patches applied to the FoundPose checkout (idempotent).
#  1. faiss k-means on CPU: faiss-cpu has no GPU k-means, and the adapter keeps samples on CPU.
set -euo pipefail
src="${1:?upstream dir}"
python3 - "$src" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "utils" / "cluster_util.py"
s = p.read_text()
old = 'gpu = True if samples.device.type == "cuda" else False'
new = 'gpu = False  # binposert: faiss-cpu build (docker/foundpose/patch.sh)'
if old in s:
    p.write_text(s.replace(old, new))
    print("patched", p)
else:
    print("already patched", p)
PY

#  2. XYZ-IBD in the vendored bop_toolkit's hard-coded dataset table (object ids with gaps, the
#     symmetric ids from models_info.json, 1440 x 1080 `xyz` camera, val object depth 600-830 mm;
#     the `xyzibd_xyz` alias is the per-camera symlink view tools/bop_camera_view.py builds).
python3 - "$src" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "external" / "bop_toolkit" / "bop_toolkit_lib" / "dataset_params.py"
s = p.read_text()
marker = '"xyzibd"'
if marker in s:
    print("already patched", p); sys.exit()
ids = "[1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]"
sym = "[1, 2, 5, 8, 9, 11, 12, 16, 17]"
old1 = '        "handal": list(range(1, 41)),\n    }[dataset_name]'
new1 = f'        "handal": list(range(1, 41)),\n        "xyzibd": {ids},\n        "xyzibd_xyz": {ids},\n    }}[dataset_name]'
old2 = '        "handal": [26, 35, 36, 37, 38, 39, 40],\n    }[dataset_name]'
new2 = f'        "handal": [26, 35, 36, 37, 38, 39, 40],\n        "xyzibd": {sym},\n        "xyzibd_xyz": {sym},\n    }}[dataset_name]'
old3 = '    else:\n        raise ValueError("Unknown BOP dataset ({}).".format(dataset_name))'
new3 = '''    elif dataset_name in ["xyzibd", "xyzibd_xyz"]:
        # binposert patch: XYZ-IBD `xyz` camera (BOP 2025), val split object distances
        p["scene_ids"] = list(range(0, 75, 5)) if split == "val" else list(range(1, 50))
        p["im_size"] = (1440, 1080)
        if split in ["val", "test"]:
            p["depth_range"] = (603.6, 827.7)
            p["azimuth_range"] = (0, 2 * math.pi)
            p["elev_range"] = (-0.5 * math.pi, 0.5 * math.pi)

    else:
        raise ValueError("Unknown BOP dataset ({}).".format(dataset_name))'''
for old, new in ((old1, new1), (old2, new2), (old3, new3)):
    if old not in s:
        sys.exit(f"dataset_params.py: expected block not found:\n{old}")
    s = s.replace(old, new)
p.write_text(s); print("patched", p)
PY

#  3. scripts/gen_templates.py: a per-object view-sphere radius. The template camera must see the
#     whole model, but the radius is sampled from the dataset's object-distance range regardless of
#     the model: XYZ-IBD object 15 is a 296 mm bar whose origin sits at one end (269 mm reach), which
#     at 716 mm spans 677 px from the principal point of the 1440 x 1080 / f = 1802 `xyz` camera and
#     touches the border ("The model does not fit the viewport"). Objects whose reach does not fit
#     at the sampled radius are rendered from further away (templates are cropped and resized anyway).
python3 - "$src" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "scripts" / "gen_templates.py"
s = p.read_text()
old = """        # Add the model to the renderer.
        model_path = bop_model_props["model_tpath"].format(obj_id=object_lid)
        renderer.add_object_model(obj_id=object_lid, model_path=model_path, debug=True)
"""
new = """        # Add the model to the renderer.
        model_path = bop_model_props["model_tpath"].format(obj_id=object_lid)
        renderer.add_object_model(obj_id=object_lid, model_path=model_path, debug=True)

        # binposert patch: push the view sphere out for models that cannot fit the viewport at
        # the sampled radius (reach from the model origin vs the shorter half-extent of the image).
        _pts = inout.load_ply(model_path)["pts"]
        _reach = float(np.linalg.norm(_pts, axis=1).max())
        _half = 0.9 * min(render_camera_model.c[0], render_camera_model.c[1])
        _f = float(np.mean(render_camera_model.f))
        _scale = max(1.0, _reach * _f / _half / min(np.linalg.norm(v["t"]) for v in views))
        obj_views = views
        if _scale > 1.0:
            logger.info(f"Object {object_lid}: reach {_reach:.0f} mm, view radius scaled by {_scale:.2f}")
            obj_views = [{"R": v["R"], "t": v["t"] * _scale} for v in views]
"""
old_loop = "        for view_id, view in enumerate(views):\n"
new_loop = "        for view_id, view in enumerate(obj_views):\n"
if new in s:
    print("gen_templates.py already patched")
elif old in s and s.count(old_loop) == 1:
    p.write_text(s.replace(old, new).replace(old_loop, new_loop)); print("patched gen_templates.py")
else:
    sys.exit("gen_templates.py: expected block not found; upstream changed?")
PY

#  4. scripts/gen_templates.py: a view whose rendered mask still touches the image border is
#     re-rendered from 25 % further away (up to five times) instead of aborting the whole
#     onboarding; the per-object radius of patch 3 is the first line of defence, this the second.
python3 - "$src" <<'PY'
import sys, pathlib, re
p = pathlib.Path(sys.argv[1]) / "scripts" / "gen_templates.py"
s = p.read_text()
start = s.index("                # Transformation from model to camera.\n")
end_marker = '                    raise ValueError("The model does not fit the viewport.")\n'
end = s.index(end_marker) + len(end_marker)
block = s[start:end]
if "_attempt" in s:
    print("gen_templates.py retry already patched"); sys.exit()
body = block.replace('trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])',
                     'trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"] * (1.25 ** _attempt))')
body = body.replace(end_marker,
    '                    if _attempt < 5:\n'
    '                        logger.info(f"Object {object_lid} view {view_id}: touches the border, re-rendering further away")\n'
    '                        continue\n'
    '                    raise ValueError("The model does not fit the viewport.")\n'
    '                break\n')
indented = "".join(("    " + line if line.strip() else line) for line in body.splitlines(True))
new = "                for _attempt in range(6):\n" + indented
p.write_text(s[:start] + new + s[end:]); print("patched gen_templates.py (retry)")
PY
