#!/usr/bin/env bash
# Patches applied to the CNOS checkout (tools/setup_estimator_env.sh runs this after cloning).
#
# 1. src/poses/pyrender.py: the template renderer re-centres the model with its bounding-box
#    centroid computed in the mesh's native units (mm) and applies that translation to the mesh
#    *after* scaling it to metres — an 8 mm off-centre CAD model is moved 8 m out of the frame and
#    every template comes out empty. T-LESS models are centred, XYZ-IBD's are not (13 of 15 objects
#    rendered blank). The centroid is now taken after the unit change.
set -euo pipefail
src="${1:?usage: patch.sh <cnos checkout>}"
python3 - "$src/src/poses/pyrender.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = """    # re-center objects at the origin
    re_center_transform = np.eye(4)
    re_center_transform[:3, 3] = -mesh.bounding_box.centroid
    print(f"Object center at {mesh.bounding_box.centroid}")

    diameter = get_obj_diameter(mesh)
    if not is_hot3d and diameter > 100: # object is in mm
        mesh.apply_scale(0.001)
"""
new = """    diameter = get_obj_diameter(mesh)
    if not is_hot3d and diameter > 100: # object is in mm
        mesh.apply_scale(0.001)

    # re-center objects at the origin (binposert patch: centroid taken *after* the unit change,
    # otherwise a mm centroid is applied as a metre offset and off-centre models render blank)
    re_center_transform = np.eye(4)
    re_center_transform[:3, 3] = -mesh.bounding_box.centroid
    print(f"Object center at {mesh.bounding_box.centroid}")
"""
if new in s:
    print("pyrender.py already patched")
elif old in s:
    p.write_text(s.replace(old, new)); print("patched pyrender.py")
else:
    sys.exit("pyrender.py: expected block not found; upstream changed?")
PY

# 2. src/model/dinov2.py: process_rgb_proposals replicates the full-resolution image once per
#    proposal (N x 3 x H x W float32) before cropping — 1.9 GB for 100 proposals on a 1440 x 1080
#    frame, which is a CUDA OOM on a 12 GB card. Same computation in chunks of 16 proposals.
python3 - "$src/src/model/dinov2.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = """        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        rgbs = rgb.unsqueeze(0).repeat(num_proposals, 1, 1, 1)
        masked_rgbs = rgbs * masks.unsqueeze(1)
        processed_masked_rgbs = self.rgb_proposal_processor(
            masked_rgbs, boxes
        )  # [N, 3, target_size, target_size]
        return processed_masked_rgbs
"""
new = """        num_proposals = len(masks)
        rgb = self.rgb_normalize(image_np).to(masks.device).float()
        # binposert patch: chunked so the peak is 16 full-resolution copies, not N
        step = 16
        outs = []
        for i in range(0, num_proposals, step):
            m, b = masks[i : i + step], boxes[i : i + step]
            rgbs = rgb.unsqueeze(0).repeat(len(m), 1, 1, 1)
            outs.append(self.rgb_proposal_processor(rgbs * m.unsqueeze(1), b))
        processed_masked_rgbs = torch.cat(outs, dim=0)  # [N, 3, target_size, target_size]
        return processed_masked_rgbs
"""
if new in s:
    print("dinov2.py already patched")
elif old in s:
    p.write_text(s.replace(old, new)); print("patched dinov2.py")
else:
    sys.exit("dinov2.py: expected block not found; upstream changed?")
PY
