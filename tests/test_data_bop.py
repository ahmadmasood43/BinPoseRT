import numpy as np

from binposert import transforms as tf
from binposert.data import BopDataset, camera_id_for
from binposert.symmetry import sym_aware_rotation_distance_deg


def test_enumeration(mini_bop: BopDataset):
    assert mini_bop.scene_ids == [1, 2]
    assert mini_bop.image_ids(1) == [0, 1]
    assert mini_bop.object_ids == [1, 5]
    assert mini_bop.name == "mini_bop"


def test_models_carry_symmetry_from_models_info(mini_bop: BopDataset):
    m1 = mini_bop.load_model(1)
    m5 = mini_bop.load_model(5)
    assert m1.diameter > 0 and m5.diameter > 0
    assert len(m1.symmetry) == 36  # continuous z axis, sampled
    assert len(m5.symmetry) == 2  # one discrete 180 deg symmetry
    assert m1.vertices.shape[1] == 3 and m1.faces.shape[1] == 3
    assert mini_bop.load_model(1) is m1  # cached


def test_view_and_ground_truth_loading(mini_bop: BopDataset):
    view, gts = mini_bop.load_view(1, 0)
    assert view.camera_id == camera_id_for(0) == "000000"
    assert view.image_size == (240, 320)
    assert view.rgb is not None and view.rgb.shape == (240, 320, 3)
    assert view.depth is not None and view.depth.shape == (240, 320)
    assert view.depth[view.depth > 0].min() > 100  # mm, not raw uint16 units
    assert tf.is_rigid(view.T_world_camera)
    assert [g.object_id for g in gts] == [1, 1, 5]
    assert all(0 < g.visible_fraction <= 1 for g in gts)
    mask = mini_bop.gt_mask(1, 0, gt_index=2, visible_only=True)
    assert mask.shape == (240, 320) and mask.dtype == bool and mask.any()
    full = mini_bop.gt_mask(1, 0, gt_index=2, visible_only=False)
    assert full.sum() >= mask.sum()


def test_gt_mask_matches_depth_support(mini_bop: BopDataset):
    view, _ = mini_bop.load_view(2, 1)
    union = np.zeros(view.image_size, dtype=bool)
    for i in range(3):
        union |= mini_bop.gt_mask(2, 1, i)
    assert view.depth is not None
    np.testing.assert_array_equal(union, view.depth > 0)


def test_multi_view_ground_truth_agrees_in_world_frame(mini_bop: BopDataset):
    """The invariant multi-view fusion depends on: T_world_camera_i @ T_camera_object_i is the same
    physical pose from every camera (up to the object's symmetry)."""
    scene = mini_bop.load_scene(1, load_rgb=False, load_depth=False)
    assert scene.is_multi_view and len(scene.views) == 2
    a, b = scene.views
    for gt_a, gt_b in zip(
        scene.ground_truth[a.camera_id], scene.ground_truth[b.camera_id], strict=True
    ):
        assert gt_a.object_id == gt_b.object_id
        T_wo_a = a.T_world_camera @ gt_a.T_camera_object
        T_wo_b = b.T_world_camera @ gt_b.T_camera_object
        sym = mini_bop.load_model(gt_a.object_id).symmetry
        assert tf.translation_distance(T_wo_a, T_wo_b) < 1e-6
        assert sym_aware_rotation_distance_deg(T_wo_a, T_wo_b, sym) < 1e-6


def test_scene_subset_of_views(mini_bop: BopDataset):
    scene = mini_bop.load_scene(2, image_ids=[1], load_rgb=False)
    assert [v.image_id for v in scene.views] == [1]
    assert scene.view("000001").scene_id == 2
