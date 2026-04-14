from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from evaluation.formats import MotionRepresentation, MotionSample

SMPLX_JOINT_COUNT = 55
SMPLX_AXIS_ANGLE_DIMS = SMPLX_JOINT_COUNT * 3
SMPLX_ROT6D_DIMS = SMPLX_JOINT_COUNT * 6


@dataclass(slots=True)
class Beat2Motion:
    """Canonical BEAT2 motion container before metric-specific conversion."""

    poses_axis_angle: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    gender: str
    fps: float
    source_path: str | None = None
    has_full_smplx_pose: bool = False

    @property
    def num_frames(self) -> int:
        return int(self.poses_axis_angle.shape[0])


def read_beat2_npz(npz_path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def canonicalize_beat2_npz(npz_path: str | Path) -> Beat2Motion:
    """
    Convert a BEAT2-like npz file into a canonical representation.

    Two cases are supported:
    1. Raw BEAT2-like files with `poses` and `trans`.
    2. GMR's AMASS-compatible files with `pose_body`, `root_orient`, `trans`.

    For FGD we need the full SMPL-X pose (55 joints * 3 axis-angle dims = 165).
    Files that only keep `root_orient + pose_body` are marked as partial.
    """

    raw = read_beat2_npz(npz_path)
    keys = set(raw.keys())

    if {"poses", "trans"}.issubset(keys):
        poses = np.asarray(raw["poses"], dtype=np.float32)
        if poses.ndim != 2:
            raise ValueError(f"`poses` must be 2D, got shape {poses.shape}.")
        pose_dims = poses.shape[1]
        has_full_smplx_pose = pose_dims >= SMPLX_AXIS_ANGLE_DIMS
        canonical_poses = poses[:, :SMPLX_AXIS_ANGLE_DIMS]
        if canonical_poses.shape[1] < 66:
            raise ValueError(
                f"Expected at least 66 pose dims for root+body, got {canonical_poses.shape[1]}."
            )
        trans = np.asarray(raw["trans"], dtype=np.float32)
        betas = _canonicalize_betas(raw.get("betas"))
        gender = _canonicalize_gender(raw.get("gender"))
        fps = _canonicalize_fps(raw.get("mocap_frame_rate"))
    elif {"pose_body", "root_orient", "trans"}.issubset(keys):
        root_orient = np.asarray(raw["root_orient"], dtype=np.float32)
        pose_body = np.asarray(raw["pose_body"], dtype=np.float32)
        trans = np.asarray(raw["trans"], dtype=np.float32)
        if root_orient.ndim != 2 or root_orient.shape[1] != 3:
            raise ValueError(
                f"`root_orient` must have shape (T, 3), got {root_orient.shape}."
            )
        if pose_body.ndim != 2 or pose_body.shape[1] != 63:
            raise ValueError(f"`pose_body` must have shape (T, 63), got {pose_body.shape}.")

        canonical_poses = np.zeros((pose_body.shape[0], SMPLX_AXIS_ANGLE_DIMS), dtype=np.float32)
        canonical_poses[:, :3] = root_orient
        canonical_poses[:, 3:66] = pose_body
        betas = _canonicalize_betas(raw.get("betas"))
        gender = _canonicalize_gender(raw.get("gender"))
        fps = _canonicalize_fps(raw.get("mocap_frame_rate"))
        has_full_smplx_pose = False
    else:
        raise ValueError(
            f"{npz_path} is not a supported BEAT2/GMR motion file. Keys: {sorted(keys)}"
        )

    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"`trans` must have shape (T, 3), got {trans.shape}.")
    if trans.shape[0] != canonical_poses.shape[0]:
        raise ValueError(
            f"Frame count mismatch between poses ({canonical_poses.shape[0]}) and trans ({trans.shape[0]})."
        )

    return Beat2Motion(
        poses_axis_angle=canonical_poses,
        trans=trans,
        betas=betas,
        gender=gender,
        fps=fps,
        source_path=str(Path(npz_path)),
        has_full_smplx_pose=has_full_smplx_pose,
    )


def beat2_to_fgd_rot6d(npz_path: str | Path, require_full_pose: bool = True) -> MotionSample:
    """
    Convert BEAT2 motion into the 55-joint SMPL-X rot6d format expected by FGD.
    """

    motion = canonicalize_beat2_npz(npz_path)
    if require_full_pose and not motion.has_full_smplx_pose:
        raise ValueError(
            "FGD needs the full SMPL-X pose from the original BEAT2 `poses` array. "
            "This file only contains root+body and pads the remaining joints with zeros."
        )

    rot6d = axis_angle_to_rot6d(motion.poses_axis_angle)
    sample = MotionSample(
        motion=rot6d,
        fps=motion.fps,
        representation=MotionRepresentation.ROT6D,
        source_path=motion.source_path,
    )
    sample.validate()
    return sample


def beat2_to_axis_angle(npz_path: str | Path) -> MotionSample:
    """
    Convert BEAT2 motion into SMPL-X axis-angle format with shape (T, 55, 3).

    This follows GMR's current conversion chain: when only root+body are available,
    missing joints are zero-padded.
    """

    motion = canonicalize_beat2_npz(npz_path)
    axis_angle = motion.poses_axis_angle.reshape(motion.num_frames, SMPLX_JOINT_COUNT, 3)
    sample = MotionSample(
        motion=axis_angle.astype(np.float32),
        fps=motion.fps,
        representation=MotionRepresentation.AXIS_ANGLE,
        source_path=motion.source_path,
    )
    sample.validate()
    return sample


def flatten_axis_angle_sample(sample: MotionSample) -> np.ndarray:
    if sample.representation != MotionRepresentation.AXIS_ANGLE:
        raise ValueError(f"Expected axis-angle sample, got {sample.representation}.")
    sample.validate()
    if sample.motion.shape[-2:] != (SMPLX_JOINT_COUNT, 3):
        raise ValueError(
            f"Expected axis-angle shape (T, {SMPLX_JOINT_COUNT}, 3), got {sample.motion.shape}."
        )
    return sample.motion.reshape(sample.motion.shape[0], -1).astype(np.float32)


def axis_angle_to_rot6d(poses_axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert SMPL-X axis-angle poses from (T, 165) to (T, 55, 6).
    """

    poses_axis_angle = np.asarray(poses_axis_angle, dtype=np.float32)
    if poses_axis_angle.ndim != 2 or poses_axis_angle.shape[1] != SMPLX_AXIS_ANGLE_DIMS:
        raise ValueError(
            f"Expected axis-angle poses with shape (T, {SMPLX_AXIS_ANGLE_DIMS}), got {poses_axis_angle.shape}."
        )

    num_frames = poses_axis_angle.shape[0]
    rotations = R.from_rotvec(poses_axis_angle.reshape(-1, 3)).as_matrix()
    rot6d = rotations[:, :, :2].transpose(0, 2, 1).reshape(num_frames, SMPLX_JOINT_COUNT, 6)
    return rot6d.astype(np.float32)


def _canonicalize_betas(betas: np.ndarray | None) -> np.ndarray:
    if betas is None:
        return np.zeros(16, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    result = np.zeros(16, dtype=np.float32)
    result[: min(16, betas.shape[0])] = betas[:16]
    return result


def _canonicalize_gender(gender: np.ndarray | str | None) -> str:
    if gender is None:
        return "neutral"
    if isinstance(gender, np.ndarray):
        if gender.ndim == 0:
            return str(gender.item())
        if gender.size == 1:
            return str(gender.reshape(-1)[0])
    return str(gender)


def _canonicalize_fps(fps: np.ndarray | float | int | None) -> float:
    if fps is None:
        return 30.0
    if isinstance(fps, np.ndarray):
        if fps.ndim == 0:
            return float(fps.item())
        if fps.size == 1:
            return float(fps.reshape(-1)[0])
    return float(fps)
