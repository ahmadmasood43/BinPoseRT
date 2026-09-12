# BinPoseRT

Model-based 6DoF pose estimation of textureless industrial objects in cluttered bins, from one or more
calibrated RGB-D views, with a calibrated confidence and a decision attached to every output pose.

## Language

### Inputs

**Scene**:
One bin state observed by one or more calibrated Views at approximately the same time. A BOP scene is a Scene.
_Avoid_: frame set, capture, sample

**View**:
One camera capture of a Scene: RGB image, optional depth image, intrinsics `K`, extrinsics `T_world_camera`, and a `camera_id`.
_Avoid_: frame, image, camera (when meaning the capture)

**ObjectModel**:
A CAD mesh with a stable `object_id`, a diameter, and its SymmetryGroup. Onboarded once; the same model may appear many times in a Scene.
_Avoid_: mesh, CAD, class, category

**SymmetryGroup**:
The finite list of rigid transforms under which an ObjectModel is indistinguishable, always containing the identity. Continuous symmetries are represented by evenly spaced samples.
_Avoid_: symmetry list, sym

### Per-view results

**Detection**:
One mask for one `object_id` in one View, with a segmentation score. Carries no pose.
_Avoid_: instance, proposal, segment

**PoseHypothesis**:
One candidate `T_camera_object` for one Detection, tagged with a `stage` (`coarse` or `refined`) and its QualitySignals. An estimator may produce several per Detection.
_Avoid_: pose (alone), estimate, candidate, prediction

**QualitySignals**:
The measurable evidence about a PoseHypothesis or FusedPose — segmentation score, estimator score, reprojection error, registration fitness and RMSE, depth coverage, visible fraction, silhouette error, multi-view residual, hypothesis dispersion.
_Avoid_: confidence, score (alone), features

**Refinement**:
A stage that maps a coarse PoseHypothesis to a refined one and may reject, returning the input unchanged with a rejection reason.
_Avoid_: ICP (as the stage name), polish

### Cross-view results

**Association**:
Assigning the PoseHypotheses of several Views to ObjectTracks, one-to-one per View, gated by geometric compatibility; a hypothesis may remain unassigned.
_Avoid_: matching, tracking, linking

**ObjectTrack**:
One physical object in a Scene, expressed in the world frame, owning the PoseHypotheses associated to it across Views.
_Avoid_: instance, object (alone), track (alone)

**Fusion**:
Combining an ObjectTrack's hypotheses into one FusedPose after symmetry alignment.
_Avoid_: averaging, merging, aggregation

**FusedPose**:
The final `T_world_object` of an ObjectTrack together with its Confidence and Verdict.
_Avoid_: final pose, result, output pose

### Decisions

**Confidence**:
A calibrated probability that a FusedPose (or PoseHypothesis) satisfies the success criterion `MSSD < 0.1 · diameter`. Produced only by a ConfidenceModel.
_Avoid_: score, certainty, probability (alone)

**ConfidenceModel**:
A fitted model mapping QualitySignals to a Confidence. Model H scores PoseHypotheses; Model F scores FusedPoses.
_Avoid_: classifier, failure detector

**Verdict**:
The decision attached to a FusedPose: `accept`, `reject`, or `request_view`.
_Avoid_: status, flag, outcome

**Candidate Viewpoint**:
A View of the Scene that has not yet been used and that the next-best-view selector may unlock.
_Avoid_: NBV, camera pose, virtual camera

**Grasp Pose**:
`T_object_gripper` authored per ObjectModel; combined with a FusedPose it yields `T_robot_gripper`. The perception system emits the transform and stops there.
_Avoid_: pick pose, grasp point

### Frames

**Transform naming**:
`T_a_b` maps points expressed in frame `b` into frame `a`, so `T_world_object = T_world_camera @ T_camera_object`. Frames in use: `world`, `camera`, `object`, `robot`, `gripper`.
_Avoid_: pose matrix, RT, extrinsics (for anything other than `T_world_camera`)
