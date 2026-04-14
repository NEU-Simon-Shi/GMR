"""Dataset-specific motion adapters."""

from .beat2 import (
    Beat2Motion,
    beat2_to_axis_angle,
    beat2_to_fgd_rot6d,
    canonicalize_beat2_npz,
    flatten_axis_angle_sample,
    read_beat2_npz,
)
from .gmr_3d import flatten_joint_position_sample, gmr_3d_to_joint_positions, load_gmr_3d_npz
from .pkl_motion import flatten_axis_angle_proxy, load_robot_pkl, pkl_to_axis_angle_proxy, pkl_to_fgd_rot6d_proxy

__all__ = [
    "Beat2Motion",
    "beat2_to_axis_angle",
    "beat2_to_fgd_rot6d",
    "canonicalize_beat2_npz",
    "flatten_axis_angle_sample",
    "flatten_axis_angle_proxy",
    "flatten_joint_position_sample",
    "gmr_3d_to_joint_positions",
    "load_robot_pkl",
    "load_gmr_3d_npz",
    "pkl_to_axis_angle_proxy",
    "pkl_to_fgd_rot6d_proxy",
    "read_beat2_npz",
]
