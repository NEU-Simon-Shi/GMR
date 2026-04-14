import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.fgd import FGDConfig, collect_motion_pairs, compute_fgd_for_pair, compute_fgd_for_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FGD for BEAT2/GMR motions.")
    parser.add_argument("--gt", required=True, help="Ground-truth npz file or directory.")
    parser.add_argument("--pred", required=True, help="Prediction motion file or directory.")
    parser.add_argument(
        "--weights_root",
        default="evaluation/weights/emage",
        help="Directory containing EMAGE assets.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device for FGD encoder.",
    )
    parser.add_argument(
        "--strict_fgd",
        action="store_true",
        help="Require full SMPL-X pose; disable GMR-chain padding fallback.",
    )
    parser.add_argument(
        "--pred_format",
        default="npz",
        choices=["npz", "pkl"],
        help="Prediction format. Default: npz.",
    )
    args = parser.parse_args()

    cfg = FGDConfig(
        weights_root=args.weights_root,
        device=args.device,
        allow_partial_pose=not args.strict_fgd,
        pred_format=args.pred_format,
    )
    gt = Path(args.gt)
    pred = Path(args.pred)

    if gt.is_file() and pred.is_file():
        score = compute_fgd_for_pair(gt, pred, cfg)
        output = {
            "mode": "single_pair",
            "gt": str(gt),
            "pred": str(pred),
            "pred_format": args.pred_format,
            "fgd": score,
        }
    else:
        pairs = collect_motion_pairs(gt, pred, pred_format=args.pred_format)
        score = compute_fgd_for_pairs(pairs, cfg)
        output = {
            "mode": "batch_pairs",
            "pair_count": len(pairs),
            "gt_root": str(gt),
            "pred_root": str(pred),
            "pred_format": args.pred_format,
            "fgd": score,
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
