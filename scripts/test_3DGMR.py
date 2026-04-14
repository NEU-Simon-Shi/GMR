import argparse
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_3d_gmr(npz_relative_path: str) -> tuple[Path, dict[str, np.ndarray]]:
    npz_path = (PROJECT_ROOT / npz_relative_path).resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"3D_GMR file not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    try:
        loaded = {key: data[key] for key in data.files}
    finally:
        close = getattr(data, "close", None)
        if callable(close):
            close()
    return npz_path, loaded


def _as_scalar(value: np.ndarray):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def validate_3d_gmr(npz_relative_path: str) -> None:
    npz_path, data = _load_3d_gmr(npz_relative_path)

    required_keys = {
        "positions",
        "body_names",
        "fps",
        "source_pkl",
        "robot",
        "ik_table",
        "use_only_position_weighted_joints",
        "qpos",
    }
    missing = sorted(required_keys - set(data.keys()))
    if missing:
        raise ValueError(f"Missing required keys in {npz_path}: {missing}")

    positions = np.asarray(data["positions"])
    body_names = np.asarray(data["body_names"])
    fps = float(_as_scalar(np.asarray(data["fps"])))
    source_pkl = str(_as_scalar(np.asarray(data["source_pkl"])))
    robot = str(_as_scalar(np.asarray(data["robot"])))
    ik_table = str(_as_scalar(np.asarray(data["ik_table"])))
    use_only_position_weighted_joints = bool(
        _as_scalar(np.asarray(data["use_only_position_weighted_joints"]))
    )
    qpos = np.asarray(data["qpos"])

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"`positions` must have shape (T, J, 3), got {positions.shape}")
    if positions.shape[0] <= 0 or positions.shape[1] <= 0:
        raise ValueError(f"`positions` must contain at least one frame and one body, got {positions.shape}")
    if body_names.ndim != 1:
        raise ValueError(f"`body_names` must have shape (J,), got {body_names.shape}")
    if body_names.shape[0] != positions.shape[1]:
        raise ValueError(
            f"body_names count {body_names.shape[0]} does not match positions J {positions.shape[1]}"
        )
    if qpos.ndim != 2 or qpos.shape[0] != positions.shape[0]:
        raise ValueError(
            f"`qpos` must have shape (T, nq) and share T with positions, got {qpos.shape} vs {positions.shape}"
        )
    if not np.isfinite(positions).all():
        raise ValueError("`positions` contains NaN or Inf")
    if not np.isfinite(qpos).all():
        raise ValueError("`qpos` contains NaN or Inf")
    if fps <= 0:
        raise ValueError(f"`fps` must be positive, got {fps}")
    if robot != "nao":
        raise ValueError(f"`robot` must be 'nao', got {robot}")
    if ik_table != "ik_match_table1":
        raise ValueError(f"`ik_table` must be 'ik_match_table1', got {ik_table}")
    if not source_pkl:
        raise ValueError("`source_pkl` must not be empty")

    print(f"3D_GMR file: {npz_path}")
    print("Validation passed.")
    print(f"positions shape: {positions.shape}")
    print(f"qpos shape: {qpos.shape}")
    print(f"fps: {fps}")
    print(f"robot: {robot}")
    print(f"ik_table: {ik_table}")
    print(f"use_only_position_weighted_joints: {use_only_position_weighted_joints}")
    print(f"source_pkl: {source_pkl}")
    print(f"body_names ({body_names.shape[0]}): {list(body_names.tolist())}")
    print(f"first_frame_first_body: {positions[0, 0].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an exported 3D_GMR npz file.")
    parser.add_argument(
        "npz_relative_path",
        help="Path to the 3D_GMR .npz file, relative to the project root.",
    )
    args = parser.parse_args()
    validate_3d_gmr(args.npz_relative_path)


if __name__ == "__main__":
    main()
