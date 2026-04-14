from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.fgd import collect_motion_pairs
from evaluation.srgr import (
    SRGR_BODY_NAMES,
    SRGR_GLOBAL_SCALE_ALPHA,
    SRGRConfig,
    _prepare_gt_srgr_sample,
    _prepare_pred_srgr_sample,
)


@dataclass(frozen=True, slots=True)
class SmoothnessConfig:
    """Runtime config for smoothness metrics on BEAT2 / 3D_GMR motions."""

    global_scale_alpha: float = SRGR_GLOBAL_SCALE_ALPHA
    smplx_model_root: str = "assets/body_models"
    raw_beat2_up_axis: str = "auto"
    body_names: tuple[str, ...] = SRGR_BODY_NAMES
    torso_relative: bool = True
    use_dt_normalization: bool = False


def _make_srgr_compatible_config(config: SmoothnessConfig) -> SRGRConfig:
    return SRGRConfig(
        global_scale_alpha=config.global_scale_alpha,
        smplx_model_root=config.smplx_model_root,
        raw_beat2_up_axis=config.raw_beat2_up_axis,
        body_names=config.body_names,
    )


def _prepare_gt_smoothness_sample(gt_motion: str | Path, config: SmoothnessConfig):
    return _prepare_gt_srgr_sample(gt_motion, _make_srgr_compatible_config(config))


def _prepare_pred_smoothness_sample(pred_motion: str | Path, config: SmoothnessConfig):
    return _prepare_pred_srgr_sample(pred_motion, _make_srgr_compatible_config(config))


def _finite_difference(
    motion: np.ndarray,
    order: int,
    dt: float,
    use_dt_normalization: bool,
) -> np.ndarray:
    values = np.asarray(motion, dtype=np.float32)
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    for _ in range(order):
        values = np.diff(values, axis=0)
        if use_dt_normalization:
            values = values / dt
    return values.astype(np.float32)


def _mean_joint_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    norms = np.linalg.norm(values, axis=-1)
    return float(np.mean(norms))


def compute_pred_jerk_mean(
    pred_motion: str | Path,
    config: SmoothnessConfig | None = None,
) -> float:
    cfg = config or SmoothnessConfig()
    pred_sample = _prepare_pred_smoothness_sample(pred_motion, cfg)
    dt = 1.0 / float(pred_sample.fps)
    jerk = _finite_difference(
        pred_sample.motion,
        order=3,
        dt=dt,
        use_dt_normalization=cfg.use_dt_normalization,
    )
    return _mean_joint_norm(jerk)


def compute_acceleration_error(
    gt_motion: str | Path,
    pred_motion: str | Path,
    config: SmoothnessConfig | None = None,
) -> float:
    cfg = config or SmoothnessConfig()
    gt_sample = _prepare_gt_smoothness_sample(gt_motion, cfg)
    pred_sample = _prepare_pred_smoothness_sample(pred_motion, cfg)

    frame_count = min(gt_sample.motion.shape[0], pred_sample.motion.shape[0])
    if frame_count < 3:
        raise ValueError("Acceleration Error requires at least 3 shared frames.")

    dt = 1.0 / float(min(gt_sample.fps, pred_sample.fps))
    gt_motion_aligned = gt_sample.motion[:frame_count]
    pred_motion_aligned = pred_sample.motion[:frame_count]
    gt_acc = _finite_difference(
        gt_motion_aligned,
        order=2,
        dt=dt,
        use_dt_normalization=cfg.use_dt_normalization,
    )
    pred_acc = _finite_difference(
        pred_motion_aligned,
        order=2,
        dt=dt,
        use_dt_normalization=cfg.use_dt_normalization,
    )
    return _mean_joint_norm(pred_acc - gt_acc)


def compute_smoothness_for_pair(
    pred_motion: str | Path,
    gt_motion: str | Path | None = None,
    config: SmoothnessConfig | None = None,
) -> dict[str, float | None]:
    cfg = config or SmoothnessConfig()
    pred_jerk_mean = compute_pred_jerk_mean(pred_motion, cfg)
    acceleration_error = None if gt_motion is None else compute_acceleration_error(gt_motion, pred_motion, cfg)
    return {
        "acceleration_error": acceleration_error,
        "pred_jerk_mean": pred_jerk_mean,
    }


def compute_smoothness_for_pairs(
    pairs: list[tuple[Path, Path]],
    config: SmoothnessConfig | None = None,
) -> dict[str, float | None]:
    cfg = config or SmoothnessConfig()
    if not pairs:
        return {"acceleration_error": None, "pred_jerk_mean": 0.0}

    acceleration_errors: list[float] = []
    pred_jerk_means: list[float] = []
    for gt_motion, pred_motion in pairs:
        metrics = compute_smoothness_for_pair(pred_motion=pred_motion, gt_motion=gt_motion, config=cfg)
        if metrics["acceleration_error"] is not None:
            acceleration_errors.append(float(metrics["acceleration_error"]))
        pred_jerk_means.append(float(metrics["pred_jerk_mean"]))

    return {
        "acceleration_error": None if not acceleration_errors else float(np.mean(acceleration_errors)),
        "pred_jerk_mean": 0.0 if not pred_jerk_means else float(np.mean(pred_jerk_means)),
    }


__all__ = [
    "SmoothnessConfig",
    "collect_motion_pairs",
    "compute_acceleration_error",
    "compute_pred_jerk_mean",
    "compute_smoothness_for_pair",
    "compute_smoothness_for_pairs",
]
