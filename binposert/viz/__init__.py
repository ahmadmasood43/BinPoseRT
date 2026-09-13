"""Overlays and failure galleries (D18). Pure CPU: Open3D raycasting + OpenCV drawing."""

from binposert.viz.gallery import failure_gallery
from binposert.viz.overlay import crop_around, draw_pose_contour, overlay_pose

__all__ = ["crop_around", "draw_pose_contour", "failure_gallery", "overlay_pose"]
