import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.adapters.pkl_motion import pkl_to_gmr_3d


def _collect_pkl_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pkl":
            raise ValueError(f"Expected a .pkl file, got {path}")
        return [path]
    files = sorted(p for p in path.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No .pkl files found under: {path}")
    return files


def _default_output_root() -> Path:
    return PROJECT_ROOT / "motion_data" / "BEAT2" / "3D_GMR"


def _resolve_output_path(pkl_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / f"{pkl_path.stem}.npz"
    relative = pkl_path.relative_to(input_root)
    return (output_root / relative).with_suffix(".npz")


def _export_single_file(
        pkl_path: Path,
        input_root: Path,
        output_root: Path,
        use_only_position_weighted_joints: bool,
) -> Path:
    converted = pkl_to_gmr_3d(
        pkl_path=pkl_path,
        use_only_position_weighted_joints=use_only_position_weighted_joints,
    )
    output_path = _resolve_output_path(pkl_path, input_root, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **converted)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GMR NAO pkl motions into 3D_GMR npz files with body positions."
    )
    parser.add_argument(
        "pkl_path",
        help="A single GMR .pkl motion file or a directory containing .pkl files.",
    )
    parser.add_argument(
        "--use_only_position_weighted_joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to keep only ik_match_table1 entries with positive position weights. Default: true.",
    )
    parser.add_argument(
        "--output_root",
        default=str(_default_output_root()),
        help="Directory for exported 3D_GMR npz files. Default: motion_data/BEAT2/3D_GMR",
    )
    args = parser.parse_args()

    input_path = Path(args.pkl_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    pkl_files = _collect_pkl_files(input_path)

    for pkl_file in pkl_files:
        output_path = _export_single_file(
            pkl_path=pkl_file,
            input_root=input_path,
            output_root=output_root,
            use_only_position_weighted_joints=args.use_only_position_weighted_joints,
        )
        print(output_path)


if __name__ == "__main__":
    main()
