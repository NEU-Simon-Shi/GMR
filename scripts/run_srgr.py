import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.fgd import collect_motion_pairs
from evaluation.srgr import (
    SRGR_BODY_NAMES,
    SRGRConfig,
    SRGR_GLOBAL_SCALE_ALPHA,
    SRGR_THRESHOLD_METERS,
    compute_srgr_for_pair,
    compute_srgr_for_pairs,
    resolve_semantic_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SRGR for BEAT2 GT and 3D_GMR predictions.")
    parser.add_argument("--gt", required=True, help="Ground-truth BEAT2 npz file or directory.")
    parser.add_argument("--pred", required=True, help="Prediction 3D_GMR npz file or directory.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SRGR_THRESHOLD_METERS,
        help=f"SRGR success threshold in meters. Default: {SRGR_THRESHOLD_METERS}",
    )
    parser.add_argument(
        "--global_scale_alpha",
        type=float,
        default=SRGR_GLOBAL_SCALE_ALPHA,
        help=f"Fixed global scaling factor applied to GT human coordinates. Default: {SRGR_GLOBAL_SCALE_ALPHA}",
    )
    parser.add_argument(
        "--smplx_model_root",
        default="assets/body_models",
        help="Directory that contains the smplx/ model folder.",
    )
    parser.add_argument(
        "--raw_beat2_up_axis",
        default="auto",
        choices=["auto", "z", "y"],
        help=(
            "Up axis for raw BEAT2 `poses` files. "
            "`auto` treats raw BEAT2 as Y-up and leaves AMASS-compatible files unchanged."
        ),
    )
    parser.add_argument(
        "--semantic",
        default=None,
        help="Optional semantic weights file (.npy/.csv/.txt) for a single pair.",
    )
    parser.add_argument(
        "--semantic_dir",
        default=None,
        help=(
            "Optional directory for semantic weights in batch mode. The script matches "
            "files by the GT relative path and supports .npy/.csv/.txt."
        ),
    )
    args = parser.parse_args()

    cfg = SRGRConfig(
        threshold=args.threshold,
        global_scale_alpha=args.global_scale_alpha,
        smplx_model_root=args.smplx_model_root,
        raw_beat2_up_axis=args.raw_beat2_up_axis,
    )

    gt = Path(args.gt)
    pred = Path(args.pred)

    if gt.is_file() and pred.is_file():
        if args.semantic_dir is not None and args.semantic is None:
            semantic_path = resolve_semantic_path(None, args.semantic_dir, gt, gt.parent)
        else:
            semantic_path = args.semantic
        score = compute_srgr_for_pair(
            gt_motion=gt,
            pred_motion=pred,
            config=cfg,
            semantic_path=semantic_path,
        )
        output = {
            "mode": "single_pair",
            "gt": str(gt),
            "pred": str(pred),
            "threshold": args.threshold,
            "global_scale_alpha": args.global_scale_alpha,
            "body_names": list(SRGR_BODY_NAMES),
            "raw_beat2_up_axis": args.raw_beat2_up_axis,
            "srgr": score,
            "semantic": None if semantic_path is None else str(semantic_path),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if gt.is_file() != pred.is_file():
        raise ValueError("gt 和 pred 必须同为文件或同为目录。")
    if args.semantic is not None:
        raise ValueError("--semantic 只能用于单文件模式；批量模式请使用 --semantic_dir。")

    pairs = collect_motion_pairs(gt, pred, pred_format="npz")
    score = compute_srgr_for_pairs(
        pairs=pairs,
        config=cfg,
        semantic_dir=args.semantic_dir,
        gt_root=gt,
    )
    output = {
        "mode": "batch_pairs",
        "pair_count": len(pairs),
        "gt_root": str(gt),
        "pred_root": str(pred),
        "threshold": args.threshold,
        "global_scale_alpha": args.global_scale_alpha,
        "body_names": list(SRGR_BODY_NAMES),
        "raw_beat2_up_axis": args.raw_beat2_up_axis,
        "semantic_dir": args.semantic_dir,
        "srgr": score,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
