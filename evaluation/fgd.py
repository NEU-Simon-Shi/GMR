from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from evaluation.adapters.beat2 import SMPLX_JOINT_COUNT, beat2_to_fgd_rot6d
from evaluation.adapters.pkl_motion import pkl_to_fgd_rot6d_proxy
from evaluation.emage_assets import (
    BC_MMAE_URL,
    FGD_MODEL_URL,
    SMPLX_MODEL_URL,
    ensure_trailing_sep,
    validate_assets_for_fgd,
)


@dataclass(frozen=True, slots=True)
class FGDConfig:
    """Runtime config for EMAGE FGD."""

    weights_root: str = "evaluation/weights/emage"
    device: str = "cuda"
    allow_partial_pose: bool = True
    pred_format: Literal["npz", "pkl"] = "npz"


def _import_emage_fgd():
    # The upstream file name is `mertic.py` in the EMAGE package.
    from evaluation.emage_evaltools.mertic import FGD as EmageFGD

    return EmageFGD


def format_fgd_download_commands(weights_root: str | Path) -> str:
    root = Path(weights_root)
    smplx_dir = root / "smplx_models" / "smplx"
    return "\n".join(
        [
            f"mkdir -p \"{root}\" \"{smplx_dir}\"",
            f"wget -O \"{root / 'AESKConv_240_100.bin'}\" \"{FGD_MODEL_URL}\"",
            f"wget -O \"{smplx_dir / 'SMPLX_NEUTRAL_2020.npz'}\" \"{SMPLX_MODEL_URL}\"",
            f"wget -O \"{root / 'mean_vel_smplxflame_30.npy'}\" \"{BC_MMAE_URL}\"",
        ]
    )


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _to_emage_tensor(rot6d_motion: np.ndarray) -> torch.Tensor:
    if rot6d_motion.ndim != 3 or rot6d_motion.shape[1:] != (SMPLX_JOINT_COUNT, 6):
        raise ValueError(
            f"FGD expects motion shape (T, {SMPLX_JOINT_COUNT}, 6), got {rot6d_motion.shape}."
        )
    flat = rot6d_motion.reshape(rot6d_motion.shape[0], -1)
    if flat.shape[0] < 32:
        pad = np.repeat(flat[-1:, :], 32 - flat.shape[0], axis=0)
        flat = np.concatenate([flat, pad], axis=0)
    usable_len = flat.shape[0] - (flat.shape[0] % 32)
    flat = flat[:usable_len, :]
    return torch.tensor(flat[None, ...], dtype=torch.float32)


def compute_fgd_for_pair(
    gt_motion: str | Path,
    pred_motion: str | Path,
    config: FGDConfig | None = None,
) -> float:
    cfg = config or FGDConfig()
    errors = validate_assets_for_fgd(cfg.weights_root)
    if errors:
        hint = format_fgd_download_commands(cfg.weights_root)
        joined = "\n".join(errors)
        raise FileNotFoundError(f"{joined}\n\n请先下载资源:\n{hint}")

    EmageFGD = _import_emage_fgd()
    device = _resolve_device(cfg.device)
    metric = EmageFGD(download_path=ensure_trailing_sep(cfg.weights_root), device=device)
    _update_metric_with_pair(metric, gt_motion, pred_motion, cfg.allow_partial_pose, cfg.pred_format)
    return float(metric.compute())


def compute_fgd_for_pairs(
    pairs: list[tuple[Path, Path]],
    config: FGDConfig | None = None,
) -> float:
    cfg = config or FGDConfig()
    errors = validate_assets_for_fgd(cfg.weights_root)
    if errors:
        hint = format_fgd_download_commands(cfg.weights_root)
        joined = "\n".join(errors)
        raise FileNotFoundError(f"{joined}\n\n请先下载资源:\n{hint}")
    EmageFGD = _import_emage_fgd()
    device = _resolve_device(cfg.device)
    metric = EmageFGD(download_path=ensure_trailing_sep(cfg.weights_root), device=device)
    for gt_motion, pred_motion in pairs:
        _update_metric_with_pair(metric, gt_motion, pred_motion, cfg.allow_partial_pose, cfg.pred_format)
    return float(metric.compute())


def collect_motion_pairs(
    gt_path: str | Path,
    pred_path: str | Path,
    pred_format: Literal["npz", "pkl"] = "npz",
) -> list[tuple[Path, Path]]:
    gt_path = Path(gt_path)
    pred_path = Path(pred_path)

    if gt_path.is_file() and pred_path.is_file():
        return [(gt_path, pred_path)]
    if gt_path.is_file() != pred_path.is_file():
        raise ValueError("gt 和 pred 必须同为文件或同为目录。")

    if pred_format not in ("npz", "pkl"):
        raise ValueError(f"Unsupported pred_format: {pred_format}")

    gt_files = sorted(p for p in gt_path.rglob("*.npz"))
    pred_suffix = ".npz" if pred_format == "npz" else ".pkl"
    pred_files = sorted(p for p in pred_path.rglob(f"*{pred_suffix}"))
    pred_map = {p.relative_to(pred_path): p for p in pred_files}

    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for gt_file in gt_files:
        rel = gt_file.relative_to(gt_path)
        pred_file = pred_map.get(rel.with_suffix(pred_suffix))
        if pred_file is None and pred_format == "pkl":
            pred_file = _find_pkl_by_stem(gt_file, gt_path, pred_path, pred_files)
        if pred_file is None:
            missing.append(rel)
            continue
        pairs.append((gt_file, pred_file))
    if missing:
        examples = ", ".join(str(x) for x in missing[:5])
        raise FileNotFoundError(f"pred 目录中缺少对应 npz: {examples}")
    if not pairs:
        raise FileNotFoundError("没有找到可匹配的 npz 文件对。")
    return pairs


def collect_npz_pairs(
    gt_path: str | Path,
    pred_path: str | Path,
) -> list[tuple[Path, Path]]:
    """Backward-compatible helper used by existing callers."""
    return collect_motion_pairs(gt_path, pred_path, pred_format="npz")


def _update_metric_with_pair(
    metric,
    gt_motion: str | Path,
    pred_motion: str | Path,
    allow_partial_pose: bool,
    pred_format: Literal["npz", "pkl"],
) -> None:
    gt_sample = beat2_to_fgd_rot6d(gt_motion, require_full_pose=not allow_partial_pose)
    if pred_format == "npz":
        pred_sample = beat2_to_fgd_rot6d(pred_motion, require_full_pose=not allow_partial_pose)
    elif pred_format == "pkl":
        pred_sample = pkl_to_fgd_rot6d_proxy(pred_motion)
    else:
        raise ValueError(f"Unsupported pred_format: {pred_format}")
    gt_tensor = _to_emage_tensor(gt_sample.motion)
    pred_tensor = _to_emage_tensor(pred_sample.motion)
    metric.update(pred_tensor, gt_tensor)


def _find_pkl_by_stem(
    gt_file: Path,
    gt_root: Path,
    pred_root: Path,
    pred_files: list[Path],
) -> Path | None:
    rel_parent = gt_file.relative_to(gt_root).parent
    gt_stem = gt_file.stem

    def _is_match(stem: str) -> bool:
        return stem == gt_stem or stem.startswith(gt_stem + "_") or gt_stem in stem

    same_parent_hits = [
        p
        for p in pred_files
        if p.relative_to(pred_root).parent == rel_parent and _is_match(p.stem)
    ]
    if len(same_parent_hits) == 1:
        return same_parent_hits[0]

    all_hits = [p for p in pred_files if _is_match(p.stem)]
    if len(all_hits) == 1:
        return all_hits[0]
    return None
