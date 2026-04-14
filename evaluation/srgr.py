from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R

from evaluation.adapters.beat2 import canonicalize_beat2_npz, read_beat2_npz
from evaluation.adapters.gmr_3d import gmr_3d_to_joint_positions
from evaluation.fgd import collect_motion_pairs
from evaluation.formats import MotionRepresentation, MotionSample

SRGR_BODY_NAMES = (
    "torso",
    "l_ankle",
    "r_ankle",
    "LBicep",
    "LForeArm",
    "l_gripper",
    "RBicep",
    "RForeArm",
    "r_gripper",
)

SRGR_HUMAN_BODY_MAP = {
    "torso": "pelvis",
    "l_ankle": "left_foot",
    "r_ankle": "right_foot",
    "LBicep": "left_shoulder",
    "LForeArm": "left_elbow",
    "l_gripper": "left_wrist",
    "RBicep": "right_shoulder",
    "RForeArm": "right_elbow",
    "r_gripper": "right_wrist",
}

SRGR_GLOBAL_SCALE_ALPHA = 0.32
SRGR_THRESHOLD_METERS = 0.08  # 0.1 -> SRGR = 0.9451; 0.05 -> SRGR = 0.3960


@dataclass(frozen=True, slots=True)
class SRGRConfig:
    """Runtime config for SRGR on BEAT2 / 3D_GMR motions."""

    threshold: float = SRGR_THRESHOLD_METERS
    global_scale_alpha: float = SRGR_GLOBAL_SCALE_ALPHA
    smplx_model_root: str = "assets/body_models"
    raw_beat2_up_axis: str = "auto"
    body_names: tuple[str, ...] = SRGR_BODY_NAMES


def _import_srgr():
    from evaluation.emage_evaltools.mertic import SRGR as EmageSRGR

    return EmageSRGR


def _resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_raw_beat2_npz(npz_path: str | Path) -> bool:
    keys = set(read_beat2_npz(npz_path).keys())
    return {"poses", "trans"}.issubset(keys)


def _convert_up_axis_to_z_up(
        root_orient: np.ndarray,
        trans: np.ndarray,
        source_up_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    if source_up_axis == "z":
        return root_orient, trans
    if source_up_axis != "y":
        raise ValueError(f"Unsupported source_up_axis: {source_up_axis}")

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    rotation_world = R.from_matrix(rotation_matrix)
    root_rot = R.from_rotvec(root_orient)
    root_orient_out = (rotation_world * root_rot).as_rotvec().astype(np.float32)
    trans_out = (trans @ rotation_matrix.T).astype(np.float32)
    return root_orient_out, trans_out


def _load_smplx_frames(
        npz_path: str | Path,
        smplx_model_root: str | Path,
        config: SRGRConfig,
):
    from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast

    npz_path = Path(npz_path)
    smplx_model_root = Path(smplx_model_root)
    canonical = canonicalize_beat2_npz(npz_path)
    root_orient = canonical.poses_axis_angle[:, :3].astype(np.float32)
    trans = canonical.trans.astype(np.float32)
    if _is_raw_beat2_npz(npz_path):
        source_up_axis = "y" if config.raw_beat2_up_axis == "auto" else config.raw_beat2_up_axis
        root_orient, trans = _convert_up_axis_to_z_up(root_orient, trans, source_up_axis)

    smplx_data = {
        "pose_body": canonical.poses_axis_angle[:, 3:66].astype(np.float32),
        "root_orient": root_orient,
        "trans": trans,
        "betas": canonical.betas.astype(np.float32),
        "gender": np.array(canonical.gender),
        "mocap_frame_rate": np.array(canonical.fps, dtype=np.float32),
    }

    body_model = smplx.create(
        str(smplx_model_root),
        "smplx",
        gender=canonical.gender,
        use_pca=False,
    )
    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    frames, fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=30)
    return frames, float(fps)


def _gt_npz_to_body_positions(npz_path: str | Path, config: SRGRConfig) -> MotionSample:
    smplx_model_root = Path(config.smplx_model_root)
    if not smplx_model_root.is_absolute():
        smplx_model_root = _resolve_project_root() / smplx_model_root

    frames, fps = _load_smplx_frames(npz_path, smplx_model_root, config)
    positions: list[np.ndarray] = []
    for frame in frames:
        frame_positions = []
        for body_name in config.body_names:
            human_body_name = SRGR_HUMAN_BODY_MAP[body_name]
            if human_body_name not in frame:
                raise ValueError(
                    f"GT frame from {npz_path} missing required body `{human_body_name}` for SRGR."
                )
            pos = np.asarray(frame[human_body_name][0], dtype=np.float32)
            frame_positions.append(pos)
        positions.append(np.stack(frame_positions, axis=0))

    sample = MotionSample(
        motion=np.stack(positions, axis=0).astype(np.float32),
        fps=fps,
        representation=MotionRepresentation.JOINT_POSITIONS,
        source_path=str(Path(npz_path)),
    )
    sample.validate()
    return sample


def _pred_gmr_to_body_positions(npz_path: str | Path, config: SRGRConfig) -> MotionSample:
    sample, body_names = gmr_3d_to_joint_positions(npz_path)
    body_index = {name: idx for idx, name in enumerate(body_names)}
    missing = [name for name in config.body_names if name not in body_index]
    if missing:
        raise ValueError(
            f"3D_GMR motion {npz_path} missing required SRGR body names: {missing}. "
            f"Available names: {body_names}"
        )
    ordered = np.stack([sample.motion[:, body_index[name], :] for name in config.body_names], axis=1).astype(np.float32)
    result = MotionSample(
        motion=ordered,
        fps=sample.fps,
        representation=MotionRepresentation.JOINT_POSITIONS,
        source_path=sample.source_path,
    )
    result.validate()
    return result


def _torso_relative_and_scale(sample: MotionSample, alpha: float) -> MotionSample:
    motion = np.asarray(sample.motion, dtype=np.float32).copy()
    torso = motion[:, 0:1, :]
    motion = (motion - torso) * float(alpha)
    result = MotionSample(
        motion=motion.astype(np.float32),
        fps=sample.fps,
        representation=MotionRepresentation.JOINT_POSITIONS,
        source_path=sample.source_path,
    )
    result.validate()
    return result


def _prepare_gt_srgr_sample(gt_motion: str | Path, config: SRGRConfig) -> MotionSample:
    return _torso_relative_and_scale(_gt_npz_to_body_positions(gt_motion, config), config.global_scale_alpha)


def _prepare_pred_srgr_sample(pred_motion: str | Path, config: SRGRConfig) -> MotionSample:
    return _torso_relative_and_scale(_pred_gmr_to_body_positions(pred_motion, config), 1.0)


def _resolve_csv_score_key(fieldnames: list[str]) -> str:
    normalized = {name.lower().strip().replace(" ", "").replace("_", ""): name for name in fieldnames}
    candidates = (
        "semantic",
        "semanticscore",
        "semanticrelevance",
        "score",
        "value",
        "weight",
        "mean",
        "avg",
    )
    for key in candidates:
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not find a semantic score column in CSV headers: {fieldnames}")


def _resolve_csv_time_keys(fieldnames: list[str]) -> tuple[str | None, str | None]:
    normalized = {name.lower().strip().replace(" ", "").replace("_", ""): name for name in fieldnames}
    start_candidates = ("start", "startframe", "starttime", "begin", "from")
    end_candidates = ("end", "endframe", "endtime", "stop", "to")
    start_key = next((normalized[key] for key in start_candidates if key in normalized), None)
    end_key = next((normalized[key] for key in end_candidates if key in normalized), None)
    return start_key, end_key


def _load_semantic_csv(path: Path, num_frames: int, fps: float) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return np.ones(num_frames, dtype=np.float32)

    score_key = _resolve_csv_score_key(reader.fieldnames or [])
    start_key, end_key = _resolve_csv_time_keys(reader.fieldnames or [])
    if start_key is None or end_key is None:
        scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float32)
        return _fit_semantic_length(scores, num_frames)

    scores = np.ones(num_frames, dtype=np.float32)
    max_seconds = num_frames / float(fps)
    for row in rows:
        score = float(row[score_key])
        start_raw = float(row[start_key])
        end_raw = float(row[end_key])
        if end_raw <= max_seconds + 1.0:
            start_idx = int(round(start_raw * fps))
            end_idx = int(round(end_raw * fps))
        else:
            start_idx = int(round(start_raw))
            end_idx = int(round(end_raw))
        start_idx = max(0, min(num_frames, start_idx))
        end_idx = max(start_idx + 1, min(num_frames, end_idx))
        scores[start_idx:end_idx] = score
    return scores.astype(np.float32)


def _fit_semantic_length(values: np.ndarray, num_frames: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.ones(num_frames, dtype=np.float32)
    if values.size >= num_frames:
        return values[:num_frames].astype(np.float32)
    pad_value = float(values[-1])
    return np.pad(values, (0, num_frames - values.size), constant_values=pad_value).astype(np.float32)


def load_semantic_weights(semantic_path: str | Path | None, num_frames: int, fps: float) -> np.ndarray:
    if semantic_path is None:
        return np.ones(num_frames, dtype=np.float32)

    path = Path(semantic_path)
    if not path.exists():
        raise FileNotFoundError(f"Semantic file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
        return _fit_semantic_length(np.asarray(values, dtype=np.float32).reshape(-1), num_frames)
    if suffix in {".csv", ".txt"}:
        return _load_semantic_csv(path, num_frames, fps)
    raise ValueError(f"Unsupported semantic file format: {path}")


def resolve_semantic_path(
        semantic: str | Path | None,
        semantic_dir: str | Path | None,
        gt_motion: str | Path,
        gt_root: str | Path | None = None,
) -> Path | None:
    if semantic is not None:
        return Path(semantic)
    if semantic_dir is None:
        return None

    semantic_dir = Path(semantic_dir)
    gt_motion = Path(gt_motion)
    rel = gt_motion.name if gt_root is None else str(gt_motion.relative_to(Path(gt_root)))
    rel_path = Path(rel)
    for suffix in (".npy", ".csv", ".txt"):
        candidate = (semantic_dir / rel_path).with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve semantic file for {gt_motion} under {semantic_dir}. "
        "Expected same relative path with .npy/.csv/.txt suffix."
    )


def compute_srgr_for_pair(
        gt_motion: str | Path,
        pred_motion: str | Path,
        config: SRGRConfig | None = None,
        semantic_path: str | Path | None = None,
) -> float:
    cfg = config or SRGRConfig()
    gt_sample = _prepare_gt_srgr_sample(gt_motion, cfg)
    pred_sample = _prepare_pred_srgr_sample(pred_motion, cfg)
    return _compute_srgr_from_samples(gt_sample, pred_sample, cfg, semantic_path)


def compute_srgr_for_pairs(
        pairs: list[tuple[Path, Path]],
        config: SRGRConfig | None = None,
        semantic_dir: str | Path | None = None,
        semantic_file_map: dict[str, Path] | None = None,
        gt_root: str | Path | None = None,
) -> float:
    cfg = config or SRGRConfig()
    EmageSRGR = _import_srgr()
    metric = EmageSRGR(threshold=cfg.threshold, joints=len(cfg.body_names), joint_dim=3)
    frame_counter = 0
    weighted_sum = 0.0

    for gt_motion, pred_motion in pairs:
        gt_sample = _prepare_gt_srgr_sample(gt_motion, cfg)
        pred_sample = _prepare_pred_srgr_sample(pred_motion, cfg)
        semantic_path = None
        if semantic_file_map is not None:
            semantic_path = semantic_file_map.get(str(gt_motion))
        elif semantic_dir is not None:
            semantic_path = resolve_semantic_path(None, semantic_dir, gt_motion, gt_root)

        rate, length = _run_srgr(metric, gt_sample, pred_sample, semantic_path)
        weighted_sum += rate * length
        frame_counter += length

    if frame_counter == 0:
        return 0.0
    return float(weighted_sum / frame_counter)


def _compute_srgr_from_samples(
        gt_sample: MotionSample,
        pred_sample: MotionSample,
        cfg: SRGRConfig,
        semantic_path: str | Path | None,
) -> float:
    EmageSRGR = _import_srgr()
    metric = EmageSRGR(threshold=cfg.threshold, joints=gt_sample.motion.shape[1], joint_dim=3)
    rate, _length = _run_srgr(metric, gt_sample, pred_sample, semantic_path)
    return float(rate)


def _run_srgr(
        metric,
        gt_sample: MotionSample,
        pred_sample: MotionSample,
        semantic_path: str | Path | None,
) -> tuple[float, int]:
    if gt_sample.representation != MotionRepresentation.JOINT_POSITIONS:
        raise ValueError(f"GT sample must be joint positions, got {gt_sample.representation}")
    if pred_sample.representation != MotionRepresentation.JOINT_POSITIONS:
        raise ValueError(f"Pred sample must be joint positions, got {pred_sample.representation}")

    frame_count = min(gt_sample.motion.shape[0], pred_sample.motion.shape[0])
    if frame_count <= 0:
        raise ValueError("SRGR requires at least one shared frame.")

    gt_motion = gt_sample.motion[:frame_count]
    pred_motion = pred_sample.motion[:frame_count]
    semantic = None if semantic_path is None else load_semantic_weights(semantic_path, frame_count, gt_sample.fps)
    metric.pose_dimes = gt_motion.shape[1]
    metric.joint_dim = gt_motion.shape[2]
    rate = metric.run(pred_motion, gt_motion, semantic=semantic)
    return float(rate), int(frame_count)


__all__ = [
    "SRGR_BODY_NAMES",
    "SRGR_GLOBAL_SCALE_ALPHA",
    "SRGR_THRESHOLD_METERS",
    "SRGRConfig",
    "collect_motion_pairs",
    "compute_srgr_for_pair",
    "compute_srgr_for_pairs",
    "load_semantic_weights",
    "resolve_semantic_path",
]
