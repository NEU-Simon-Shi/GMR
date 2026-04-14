import argparse
import json
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.bc import BCConfig, collect_motion_files, compute_bc_for_motion, compute_bc_for_motions


def _resolve_audio_for_motion(audio_dir: Path, motion_file: Path, motion_root: Path) -> Path:
    rel = motion_file.relative_to(motion_root)
    candidate = (audio_dir / rel).with_suffix(".wav")
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Audio file not found for {motion_file}: expected {candidate}")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _collect_motion_pairs(
        gt_path: Path,
        pred_path: Path,
        pred_format: Literal["beat", "gmr"],
) -> list[tuple[Path, Path]]:
    if gt_path.is_file() and pred_path.is_file():
        return [(gt_path, pred_path)]
    if gt_path.is_file() != pred_path.is_file():
        raise ValueError("gt 和 pred 必须同为文件或同为目录。")

    gt_files = sorted(p for p in gt_path.rglob("*.npz"))
    pred_files = sorted(p for p in pred_path.rglob("*.npz"))
    pred_map = {p.relative_to(pred_path): p for p in pred_files}

    pairs: list[tuple[Path, Path]] = []
    missing: list[Path] = []
    for gt_file in gt_files:
        rel = gt_file.relative_to(gt_path)
        pred_file = pred_map.get(rel)
        if pred_file is None and pred_format == "gmr":
            pred_file = _find_npz_by_stem(gt_file, gt_path, pred_path, pred_files)
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


def _find_npz_by_stem(gt_file: Path, gt_root: Path, pred_root: Path, pred_files: list[Path]) -> Path | None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BC for BEAT2/GMR motions.")
    parser.add_argument("--pred", required=True, help="Prediction motion file or directory.")
    parser.add_argument("--gt", default=None, help="Optional ground-truth npz file or directory.")
    parser.add_argument("--audio", default=None, help="Single wav file used for all motions.")
    parser.add_argument(
        "--audio_dir",
        default=None,
        help="Directory of wav files matched to motion relative paths (.npz -> .wav).",
    )
    parser.add_argument(
        "--weights_root",
        default="evaluation/weights/emage",
        help="Directory containing EMAGE assets.",
    )
    parser.add_argument("--sigma", type=float, default=0.3, help="BC sigma parameter.")
    parser.add_argument("--order", type=int, default=7, help="BC local minima order.")
    parser.add_argument(
        "--pred_format",
        default="beat",
        choices=["beat", "gmr"],
        help="Prediction format. `beat` reads BEAT2 npz; `gmr` reads 3D_GMR npz. Default: beat.",
    )
    args = parser.parse_args()

    if (args.audio is None) == (args.audio_dir is None):
        raise ValueError("必须且只能提供一个: --audio 或 --audio_dir")

    cfg = BCConfig(weights_root=args.weights_root, sigma=args.sigma, order=args.order)
    pred = Path(args.pred)
    gt = Path(args.gt) if args.gt else None
    audio = Path(args.audio) if args.audio else None
    audio_dir = Path(args.audio_dir) if args.audio_dir else None

    if gt is None:
        pred_files = collect_motion_files(pred, motion_format=args.pred_format)
        if audio is not None:
            pred_bc = compute_bc_for_motions(pred_files, audio, cfg, motion_format=args.pred_format)
        else:
            if pred.is_file():
                pred_bc = compute_bc_for_motion(
                    pred,
                    _resolve_audio_for_motion(audio_dir, pred, pred.parent),
                    cfg,
                    motion_format=args.pred_format,
                )
            else:
                values = []
                for pred_file in pred_files:
                    wav = _resolve_audio_for_motion(audio_dir, pred_file, pred)
                    values.append(compute_bc_for_motion(pred_file, wav, cfg, motion_format=args.pred_format))
                pred_bc = _mean(values)
        output = {
            "mode": "pred_only",
            "pred_count": len(pred_files),
            "pred_format": args.pred_format,
            "pred_bc": pred_bc,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if pred.is_file() and gt.is_file():
        if audio is None:
            raise ValueError("单文件 gt/pred 模式下请使用 --audio")
        pred_bc = compute_bc_for_motion(pred, audio, cfg, motion_format=args.pred_format)
        gt_bc = compute_bc_for_motion(gt, audio, cfg, motion_format="beat")
        output = {
            "mode": "single_pair",
            "pred": str(pred),
            "gt": str(gt),
            "pred_format": args.pred_format,
            "pred_bc": pred_bc,
            "gt_bc": gt_bc,
            "bc_gap_pred_minus_gt": pred_bc - gt_bc,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    pairs = _collect_motion_pairs(gt, pred, pred_format=args.pred_format)
    pred_files = [pred_file for _, pred_file in pairs]
    gt_files = [gt_file for gt_file, _ in pairs]

    if audio is not None:
        pred_bc = compute_bc_for_motions(pred_files, audio, cfg, motion_format=args.pred_format)
        gt_bc = compute_bc_for_motions(gt_files, audio, cfg, motion_format="beat")
    else:
        pred_values = []
        gt_values = []
        for gt_file, pred_file in pairs:
            wav = _resolve_audio_for_motion(audio_dir, pred_file, pred)
            pred_values.append(compute_bc_for_motion(pred_file, wav, cfg, motion_format=args.pred_format))
            gt_values.append(compute_bc_for_motion(gt_file, wav, cfg, motion_format="beat"))
        pred_bc = _mean(pred_values)
        gt_bc = _mean(gt_values)

    output = {
        "mode": "batch_pairs",
        "pair_count": len(pairs),
        "pred_format": args.pred_format,
        "pred_bc": pred_bc,
        "gt_bc": gt_bc,
        "bc_gap_pred_minus_gt": pred_bc - gt_bc,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
