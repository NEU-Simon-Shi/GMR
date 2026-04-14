from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.formats import MotionRepresentation, MotionSample


def load_gmr_3d_npz(npz_path: str | Path) -> dict[str, np.ndarray]:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    try:
        loaded = {key: data[key] for key in data.files}
    finally:
        close = getattr(data, "close", None)
        if callable(close):
            close()
    required = {"positions", "body_names", "fps"}
    missing = sorted(required - set(loaded.keys()))
    if missing:
        raise ValueError(f"{npz_path} missing required keys for 3D_GMR motion conversion: {missing}")
    return loaded


def gmr_3d_to_joint_positions(npz_path: str | Path) -> tuple[MotionSample, list[str]]:
    data = load_gmr_3d_npz(npz_path)
    positions = np.asarray(data["positions"], dtype=np.float32)
    body_names_raw = np.asarray(data["body_names"])
    fps = float(np.asarray(data["fps"]).item())

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"`positions` must have shape (T, J, 3), got {positions.shape}")
    if body_names_raw.ndim != 1 or body_names_raw.shape[0] != positions.shape[1]:
        raise ValueError(
            f"`body_names` must have shape ({positions.shape[1]},), got {body_names_raw.shape}"
        )

    body_names = [str(name.item() if isinstance(name, np.ndarray) and name.ndim == 0 else name) for name in
                  body_names_raw]
    sample = MotionSample(
        motion=positions.astype(np.float32),
        fps=fps,
        representation=MotionRepresentation.JOINT_POSITIONS,
        source_path=str(Path(npz_path)),
    )
    sample.validate()
    return sample, body_names


def flatten_joint_position_sample(sample: MotionSample) -> np.ndarray:
    if sample.representation != MotionRepresentation.JOINT_POSITIONS:
        raise ValueError(f"Expected joint position sample, got {sample.representation}.")
    sample.validate()
    if sample.motion.shape[-1] != 3:
        raise ValueError(f"Expected joint positions with final dim 3, got {sample.motion.shape}.")
    return sample.motion.reshape(sample.motion.shape[0], -1).astype(np.float32)
