from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from evaluation.adapters.beat2 import SMPLX_AXIS_ANGLE_DIMS, SMPLX_JOINT_COUNT, axis_angle_to_rot6d
from evaluation.formats import MotionRepresentation, MotionSample


def load_robot_pkl(pkl_path: str | Path) -> dict:
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported pkl content in {pkl_path}, expected dict.")
    required = {"fps", "root_pos", "root_rot", "dof_pos"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"{pkl_path} missing required keys for motion conversion: {missing}")
    return data


def pkl_to_axis_angle_proxy(pkl_path: str | Path) -> MotionSample:
    """
    Convert robot retargeting pkl into a proxy human-like axis-angle tensor.

    The mapping is deterministic and keeps the original temporal structure:
    - root position (3)
    - root rotation converted from xyzw quaternion to rotvec (3)
    - robot dof positions (remaining dims)
    Then truncate/pad to 55*3 dims and reshape to (T, 55, 3).
    """

    data = load_robot_pkl(pkl_path)
    fps = float(np.asarray(data["fps"]).item())
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"`root_pos` must have shape (T, 3), got {root_pos.shape}")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"`root_rot` must have shape (T, 4) in xyzw, got {root_rot_xyzw.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"`dof_pos` must have shape (T, D), got {dof_pos.shape}")
    if root_pos.shape[0] != root_rot_xyzw.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError("Frame count mismatch among root_pos, root_rot, and dof_pos.")

    root_rotvec = R.from_quat(root_rot_xyzw).as_rotvec().astype(np.float32)
    state_vec = np.concatenate([root_pos, root_rotvec, dof_pos], axis=1)
    proxy_flat = np.zeros((state_vec.shape[0], SMPLX_AXIS_ANGLE_DIMS), dtype=np.float32)
    dim = min(state_vec.shape[1], SMPLX_AXIS_ANGLE_DIMS)
    proxy_flat[:, :dim] = state_vec[:, :dim]
    proxy_axis_angle = proxy_flat.reshape(proxy_flat.shape[0], SMPLX_JOINT_COUNT, 3)

    sample = MotionSample(
        motion=proxy_axis_angle,
        fps=fps,
        representation=MotionRepresentation.AXIS_ANGLE,
        source_path=str(Path(pkl_path)),
    )
    sample.validate()
    return sample


def pkl_to_fgd_rot6d_proxy(pkl_path: str | Path) -> MotionSample:
    axis_sample = pkl_to_axis_angle_proxy(pkl_path)
    rot6d = axis_angle_to_rot6d(axis_sample.motion.reshape(axis_sample.motion.shape[0], -1))
    sample = MotionSample(
        motion=rot6d,
        fps=axis_sample.fps,
        representation=MotionRepresentation.ROT6D,
        source_path=axis_sample.source_path,
    )
    sample.validate()
    return sample


def flatten_axis_angle_proxy(sample: MotionSample) -> np.ndarray:
    if sample.representation != MotionRepresentation.AXIS_ANGLE:
        raise ValueError(f"Expected axis-angle proxy sample, got {sample.representation}.")
    if sample.motion.shape[-2:] != (SMPLX_JOINT_COUNT, 3):
        raise ValueError(
            f"Expected axis-angle shape (T, {SMPLX_JOINT_COUNT}, 3), got {sample.motion.shape}."
        )
    return sample.motion.reshape(sample.motion.shape[0], -1).astype(np.float32)


def resolve_gmr_3d_body_names(
        use_only_position_weighted_joints: bool = True,
) -> list[str]:
    config_path = Path(__file__).resolve().parents[
                      2] / "general_motion_retargeting" / "ik_configs" / "smplx_to_nao.json"
    with config_path.open("r", encoding="utf-8") as f:
        ik_config = json.load(f)

    body_names: list[str] = []
    for body_name, entry in ik_config["ik_match_table1"].items():
        _human_name, pos_weight, _rot_weight, _pos_offset, _rot_offset = entry
        if use_only_position_weighted_joints and float(pos_weight) <= 0:
            continue
        body_names.append(body_name)

    if not body_names:
        raise ValueError(
            "No body names selected from general_motion_retargeting/ik_configs/smplx_to_nao.json."
        )
    return body_names


def pkl_to_gmr_3d(
        pkl_path: str | Path,
        use_only_position_weighted_joints: bool = True,
) -> dict[str, object]:
    import mujoco as mj

    from general_motion_retargeting.params import ROBOT_XML_DICT

    pkl_path = Path(pkl_path)
    data = load_robot_pkl(pkl_path)
    body_names = resolve_gmr_3d_body_names(
        use_only_position_weighted_joints=use_only_position_weighted_joints,
    )

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["nao"]))
    mj_data = mj.MjData(model)
    body_ids = [model.body(body_name).id for body_name in body_names]

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    fps = float(np.asarray(data["fps"]).item())

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"`root_pos` must have shape (T, 3), got {root_pos.shape}")
    if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
        raise ValueError(f"`root_rot` must have shape (T, 4) in xyzw, got {root_rot_xyzw.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"`dof_pos` must have shape (T, D), got {dof_pos.shape}")
    if root_pos.shape[0] != root_rot_xyzw.shape[0] or root_pos.shape[0] != dof_pos.shape[0]:
        raise ValueError("Frame count mismatch among root_pos, root_rot, and dof_pos.")

    positions: list[np.ndarray] = []
    qpos_all: list[np.ndarray] = []
    for frame_idx in range(root_pos.shape[0]):
        qpos = np.concatenate([root_pos[frame_idx], root_rot_wxyz[frame_idx], dof_pos[frame_idx]], axis=0)
        if qpos.shape[0] != model.nq:
            raise ValueError(
                f"qpos dim mismatch for {pkl_path}: got {qpos.shape[0]}, expected {model.nq}."
            )
        mj_data.qpos[:] = qpos
        mj.mj_forward(model, mj_data)
        qpos_all.append(qpos.astype(np.float32).copy())
        positions.append(np.asarray(mj_data.xpos[body_ids], dtype=np.float32).copy())

    return {
        "positions": np.stack(positions, axis=0).astype(np.float32),
        "body_names": np.asarray(body_names, dtype=object),
        "fps": np.asarray(fps, dtype=np.float32),
        "source_pkl": np.asarray(str(pkl_path), dtype=object),
        "robot": np.asarray("nao", dtype=object),
        "ik_table": np.asarray("ik_match_table1", dtype=object),
        "use_only_position_weighted_joints": np.asarray(bool(use_only_position_weighted_joints)),
        "qpos": np.stack(qpos_all, axis=0).astype(np.float32),
    }
