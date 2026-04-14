from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import argrelextrema

from evaluation.adapters.beat2 import beat2_to_axis_angle, flatten_axis_angle_sample
from evaluation.adapters.gmr_3d import flatten_joint_position_sample, gmr_3d_to_joint_positions
from evaluation.emage_assets import validate_assets_for_bc
from evaluation.fgd import format_fgd_download_commands

BCMotionFormat = Literal["beat", "gmr"]
GMR_UPPER_BODY_NAMES = (
    "torso",
    "LBicep",
    "LForeArm",
    "l_gripper",
    "RBicep",
    "RForeArm",
    "r_gripper",
)


@dataclass(frozen=True, slots=True)
class BCConfig:
    """Runtime config for BC."""

    weights_root: str = "evaluation/weights/emage"
    sigma: float = 0.3
    order: int = 7
    upper_body: tuple[int, ...] = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
    gmr_upper_body_names: tuple[str, ...] = GMR_UPPER_BODY_NAMES
    gmr_velocity_threshold: float = 0.10
    gmr_use_runtime_mean_velocity: bool = True


def _import_emage_bc():
    from evaluation.emage_evaltools.mertic import BC as EmageBC

    return EmageBC


def _sanitize_onset_times(onset_times: np.ndarray) -> np.ndarray:
    onset = np.asarray(onset_times, dtype=np.float32).reshape(-1)
    if onset.size == 0:
        return onset
    onset = onset[np.isfinite(onset)]
    return np.sort(onset)


def _load_onset_times(audio_path: str | Path, bc_metric) -> np.ndarray:
    onset_times = bc_metric.load_audio(str(audio_path), without_file=False)
    return _sanitize_onset_times(onset_times)


def _compute_motion_beats_from_joint_positions(
        motion_flat: np.ndarray,
        fps: float,
        order: int,
        threshold: float,
        use_runtime_mean_velocity: bool,
) -> list[np.ndarray]:
    dt = 1.0 / float(fps)
    joints = motion_flat.transpose(1, 0)
    init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
    middle_vel = (joints[:, 2:] - joints[:, :-2]) / (2.0 * dt)
    final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
    vel = np.concatenate([init_vel, middle_vel, final_vel], axis=1).transpose(1, 0).reshape(motion_flat.shape[0], -1, 3)
    vel_norm = np.linalg.norm(vel, axis=2)

    if use_runtime_mean_velocity:
        mean_vel = np.mean(np.abs(vel_norm), axis=0)
        mean_vel = np.maximum(mean_vel, 1e-8)
        vel_norm = vel_norm / mean_vel

    beat_vel_all: list[np.ndarray] = []
    for joint_idx in range(vel_norm.shape[1]):
        valid_mask = np.flatnonzero(vel_norm[:, joint_idx] > threshold)
        local_minima = argrelextrema(vel_norm[:, joint_idx], np.less, order=order)[0]
        if valid_mask.size == 0 or local_minima.size == 0:
            beat_vel_all.append(np.empty((0,), dtype=np.int64))
            continue
        beat_vel_all.append(np.intersect1d(local_minima, valid_mask).astype(np.int64))
    return beat_vel_all


def _score_single_motion_with_audio(
        bc_metric,
        onset_times: np.ndarray,
        motion_flat: np.ndarray,
        fps: float,
        motion_format: BCMotionFormat,
        config: BCConfig,
) -> float:
    if onset_times.size == 0:
        return 0.0

    if motion_format == "beat":
        beat_vel = bc_metric.load_motion(
            pose=motion_flat,
            t_start=0,
            t_end=motion_flat.shape[0],
            pose_fps=fps,
            without_file=True,
        )
    elif motion_format == "gmr":
        beat_vel = _compute_motion_beats_from_joint_positions(
            motion_flat=motion_flat,
            fps=fps,
            order=config.order,
            threshold=config.gmr_velocity_threshold,
            use_runtime_mean_velocity=config.gmr_use_runtime_mean_velocity,
        )
    else:
        raise ValueError(f"Unsupported motion_format: {motion_format}")

    duration = motion_flat.shape[0] / float(fps)
    bc_metric.compute(onset_times, beat_vel, length=duration, pose_fps=fps)
    return float(bc_metric.avg())


def _prepare_motion_for_bc(
        motion_file: str | Path,
        motion_format: BCMotionFormat,
        config: BCConfig,
):
    if motion_format == "beat":
        sample = beat2_to_axis_angle(motion_file)
        motion_flat = flatten_axis_angle_sample(sample)
        upper_body = config.upper_body
        return sample, motion_flat, upper_body

    if motion_format == "gmr":
        sample, body_names = gmr_3d_to_joint_positions(motion_file)
        motion_flat = flatten_joint_position_sample(sample)
        body_index = {name: idx for idx, name in enumerate(body_names)}
        missing = [name for name in config.gmr_upper_body_names if name not in body_index]
        if missing:
            raise ValueError(
                f"GMR motion {motion_file} is missing required upper-body names: {missing}. "
                f"Available names: {body_names}"
            )
        upper_body = tuple(body_index[name] for name in config.gmr_upper_body_names)
        return sample, motion_flat, upper_body

    raise ValueError(f"Unsupported motion_format: {motion_format}")


def _make_bc_metric(config: BCConfig, motion_format: BCMotionFormat, upper_body: tuple[int, ...]):
    EmageBC = _import_emage_bc()
    download_path = str(config.weights_root) if motion_format == "beat" else None
    return EmageBC(
        download_path=download_path,
        sigma=config.sigma,
        order=config.order,
        upper_body=list(upper_body),
    )


def compute_bc_for_motion(
        motion_file: str | Path,
        audio_path: str | Path,
        config: BCConfig | None = None,
        motion_format: BCMotionFormat = "beat",
) -> float:
    cfg = config or BCConfig()
    if motion_format == "beat":
        errors = validate_assets_for_bc(cfg.weights_root)
        if errors:
            hint = format_fgd_download_commands(cfg.weights_root)
            joined = "\n".join(errors)
            raise FileNotFoundError(f"{joined}\n\n请先下载资源:\n{hint}")

    sample, motion_flat, upper_body = _prepare_motion_for_bc(motion_file, motion_format, cfg)
    bc_metric = _make_bc_metric(cfg, motion_format, upper_body)
    onset = _load_onset_times(audio_path, bc_metric)
    return _score_single_motion_with_audio(bc_metric, onset, motion_flat, sample.fps, motion_format, cfg)


def compute_bc_for_motions(
        motion_paths: list[str | Path],
        audio_path: str | Path,
        config: BCConfig | None = None,
        motion_format: BCMotionFormat = "beat",
) -> float:
    cfg = config or BCConfig()
    if motion_format == "beat":
        errors = validate_assets_for_bc(cfg.weights_root)
        if errors:
            hint = format_fgd_download_commands(cfg.weights_root)
            joined = "\n".join(errors)
            raise FileNotFoundError(f"{joined}\n\n请先下载资源:\n{hint}")

    first_sample, first_flat, first_upper_body = _prepare_motion_for_bc(motion_paths[0], motion_format, cfg)
    bc_metric = _make_bc_metric(cfg, motion_format, first_upper_body)
    onset = _load_onset_times(audio_path, bc_metric)
    if onset.size == 0:
        return 0.0

    for index, motion_path in enumerate(motion_paths):
        if index == 0:
            sample, motion_flat, upper_body = first_sample, first_flat, first_upper_body
        else:
            sample, motion_flat, upper_body = _prepare_motion_for_bc(motion_path, motion_format, cfg)
        bc_metric.upper_body = list(upper_body)
        _score_single_motion_with_audio(bc_metric, onset, motion_flat, sample.fps, motion_format, cfg)
    return float(bc_metric.avg())


def collect_motion_files(path: str | Path, motion_format: BCMotionFormat = "beat") -> list[Path]:
    if motion_format not in ("beat", "gmr"):
        raise ValueError(f"Unsupported motion_format: {motion_format}")
    suffix = ".npz"
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != suffix:
            raise ValueError(f"Expected {suffix} file, got {path}")
        return [path]
    files = sorted(p for p in path.rglob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No {suffix} files found under: {path}")
    return files
