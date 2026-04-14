import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.fgd import collect_motion_pairs
from evaluation.smoothness import SmoothnessConfig, compute_smoothness_for_pair, compute_smoothness_for_pairs
from evaluation.srgr import SRGR_BODY_NAMES, SRGR_GLOBAL_SCALE_ALPHA


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoothness metrics for BEAT2 GT and 3D_GMR predictions.")
    parser.add_argument("--pred", required=True, help="Prediction 3D_GMR npz file or directory.")
    parser.add_argument("--gt", default=None, help="Optional ground-truth BEAT2 npz file or directory.")
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
        "--use_dt_normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to divide finite differences by dt, dt^2, dt^3 for velocity/acceleration/jerk. Default: false.",
    )
    args = parser.parse_args()

    cfg = SmoothnessConfig(
        global_scale_alpha=args.global_scale_alpha,
        smplx_model_root=args.smplx_model_root,
        raw_beat2_up_axis=args.raw_beat2_up_axis,
        use_dt_normalization=args.use_dt_normalization,
    )

    pred = Path(args.pred)
    gt = Path(args.gt) if args.gt else None

    if gt is None:
        if pred.is_file():
            metrics = compute_smoothness_for_pair(pred_motion=pred, gt_motion=None, config=cfg)
            output = {
                "mode": "pred_only",
                "pred": str(pred),
                "global_scale_alpha": args.global_scale_alpha,
                "body_names": list(SRGR_BODY_NAMES),
                "torso_relative": True,
                "use_dt_normalization": args.use_dt_normalization,
                **metrics,
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
            return

        pred_files = sorted(pred.rglob("*.npz"))
        if not pred_files:
            raise FileNotFoundError(f"No .npz files found under: {pred}")
        pred_jerk_values = [
            compute_smoothness_for_pair(pred_motion=pred_file, gt_motion=None, config=cfg)["pred_jerk_mean"]
            for pred_file in pred_files
        ]
        output = {
            "mode": "pred_only_batch",
            "pred_count": len(pred_files),
            "pred_root": str(pred),
            "global_scale_alpha": args.global_scale_alpha,
            "body_names": list(SRGR_BODY_NAMES),
            "torso_relative": True,
            "use_dt_normalization": args.use_dt_normalization,
            "acceleration_error": None,
            "pred_jerk_mean": float(sum(pred_jerk_values) / len(pred_jerk_values)),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if gt.is_file() and pred.is_file():
        metrics = compute_smoothness_for_pair(pred_motion=pred, gt_motion=gt, config=cfg)
        output = {
            "mode": "single_pair",
            "gt": str(gt),
            "pred": str(pred),
            "global_scale_alpha": args.global_scale_alpha,
            "body_names": list(SRGR_BODY_NAMES),
            "torso_relative": True,
            "use_dt_normalization": args.use_dt_normalization,
            **metrics,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if gt.is_file() != pred.is_file():
        raise ValueError("gt 和 pred 必须同为文件或同为目录。")

    pairs = collect_motion_pairs(gt, pred, pred_format="npz")
    metrics = compute_smoothness_for_pairs(pairs, cfg)
    output = {
        "mode": "batch_pairs",
        "pair_count": len(pairs),
        "gt_root": str(gt),
        "pred_root": str(pred),
        "global_scale_alpha": args.global_scale_alpha,
        "body_names": list(SRGR_BODY_NAMES),
        "torso_relative": True,
        "use_dt_normalization": args.use_dt_normalization,
        **metrics,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
