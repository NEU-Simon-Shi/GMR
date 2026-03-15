import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REQUIRED_SMPLX_KEYS = {
    "pose_body",
    "root_orient",
    "trans",
    "betas",
    "gender",
    "mocap_frame_rate",
}


def collect_npz_files(src: Path) -> list[Path]:
    if src.is_file():
        if src.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got: {src}")
        return [src]
    if src.is_dir():
        return sorted(src.rglob("*.npz"))
    raise FileNotFoundError(f"Path does not exist: {src}")


def build_amass_compatible_file(
    src_npz: Path, converted_path: Path, source_up_axis: str
) -> Path:
    with np.load(src_npz, allow_pickle=True) as data:
        keys = set(data.files)

        if REQUIRED_SMPLX_KEYS.issubset(keys):
            pose_body = data["pose_body"].astype(np.float32)
            root_orient = data["root_orient"].astype(np.float32)
            trans = data["trans"].astype(np.float32)
            betas_raw = np.asarray(data["betas"]).reshape(-1)
            if betas_raw.shape[0] >= 16:
                betas = betas_raw[:16].astype(np.float32)
            else:
                betas = np.zeros(16, dtype=np.float32)
                betas[: betas_raw.shape[0]] = betas_raw.astype(np.float32)
            gender = data["gender"]
            mocap_frame_rate = data["mocap_frame_rate"]
        else:
            if not {"poses", "trans"}.issubset(keys):
                missing = sorted({"poses", "trans"} - keys)
                raise ValueError(f"{src_npz} is not BEAT2-like. Missing keys: {missing}")

            poses = data["poses"]
            if poses.ndim != 2 or poses.shape[1] < 66:
                raise ValueError(
                    f"{src_npz} has invalid poses shape {poses.shape}, expected (N, >=66)."
                )

            pose_body = poses[:, 3:66].astype(np.float32)
            root_orient = poses[:, :3].astype(np.float32)
            trans = data["trans"].astype(np.float32)

            if "betas" in data:
                betas_raw = np.asarray(data["betas"]).reshape(-1)
                if betas_raw.shape[0] >= 16:
                    betas = betas_raw[:16].astype(np.float32)
                else:
                    betas = np.zeros(16, dtype=np.float32)
                    betas[: betas_raw.shape[0]] = betas_raw.astype(np.float32)
            else:
                betas = np.zeros(16, dtype=np.float32)

            gender = data["gender"] if "gender" in data else np.array("neutral")
            mocap_frame_rate = (
                data["mocap_frame_rate"] if "mocap_frame_rate" in data else np.array(30)
            )

    root_orient_out, trans_out = convert_up_axis_to_z_up(
        root_orient=root_orient,
        trans=trans,
        source_up_axis=source_up_axis,
    )

    converted_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        converted_path,
        pose_body=pose_body,
        root_orient=root_orient_out,
        trans=trans_out,
        betas=betas,
        gender=gender,
        mocap_frame_rate=mocap_frame_rate,
    )
    return converted_path


def convert_up_axis_to_z_up(
    root_orient: np.ndarray, trans: np.ndarray, source_up_axis: str
) -> tuple[np.ndarray, np.ndarray]:
    if source_up_axis == "z":
        return root_orient, trans

    if source_up_axis == "y":
        # Convert from Y-up coordinates into the pipeline's Z-up convention.
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported source_up_axis: {source_up_axis}")

    rotation_world = R.from_matrix(rotation_matrix)
    root_rot = R.from_rotvec(root_orient)
    root_orient_out = (rotation_world * root_rot).as_rotvec().astype(np.float32)
    trans_out = (trans @ rotation_matrix.T).astype(np.float32)
    return root_orient_out, trans_out


def run_retarget(
    repo_root: Path,
    smplx_file: Path,
    robot: str,
    save_path: Path,
    rate_limit: bool,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "smplx_to_robot.py"),
        "--smplx_file",
        str(smplx_file),
        "--robot",
        robot,
        "--save_path",
        str(save_path),
    ]
    if rate_limit:
        cmd.append("--rate_limit")

    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BEAT2 SMPL-X npz to AMASS-compatible format and retarget to robot motion."
    )
    parser.add_argument(
        "--src",
        required=True,
        help="BEAT2 source .npz file or a directory containing .npz files (e.g. smplxflame_30).",
    )
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        help="Target robot for retargeting.",
    )
    parser.add_argument(
        "--converted_root",
        default="motion_data/BEAT2/converted",
        help="Root folder to store converted AMASS-compatible .npz files.",
    )
    parser.add_argument(
        "--save_root",
        default="motion_data/BEAT2/retargeted",
        help="Root folder to store retargeted .pkl files.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Pass --rate_limit to scripts/smplx_to_robot.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated: outputs are overwritten by default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N files (for quick tests).",
    )
    parser.add_argument(
        "--source_up_axis",
        choices=["z", "y"],
        default="z",
        help="Up axis of source motions before conversion (default: z). Use y for Y-up datasets.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src = Path(args.src).expanduser().resolve()
    converted_root = (repo_root / args.converted_root).resolve()
    save_root = (repo_root / args.save_root).resolve()

    src_files = collect_npz_files(src)
    if args.limit is not None:
        src_files = src_files[: args.limit]

    print(f"[INFO] Found {len(src_files)} file(s) from {src}")
    if len(src_files) == 0:
        return

    for i, src_npz in enumerate(src_files, start=1):
        print(f"[{i}/{len(src_files)}] Processing {src_npz}")
        if src.is_dir():
            rel = src_npz.relative_to(src)
        else:
            rel = Path(src_npz.name)

        rel_no_suffix = rel.with_suffix("")
        converted_path = converted_root / f"{rel_no_suffix}_amass_compat.npz"
        save_path = save_root / f"{rel_no_suffix}_{args.robot}.pkl"

        smplx_file = build_amass_compatible_file(
            src_npz=src_npz,
            converted_path=converted_path,
            source_up_axis=args.source_up_axis,
        )

        run_retarget(
            repo_root=repo_root,
            smplx_file=smplx_file,
            robot=args.robot,
            save_path=save_path,
            rate_limit=args.rate_limit,
        )

    print(f"[DONE] Retargeted files are saved under: {save_root}")


if __name__ == "__main__":
    main()
