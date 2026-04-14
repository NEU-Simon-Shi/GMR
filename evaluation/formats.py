from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MotionRepresentation(str, Enum):
    """Canonical motion representations used by evaluation adapters."""

    ROT6D = "rot6d"
    JOINT_POSITIONS = "joint_positions"
    AXIS_ANGLE = "axis_angle"


@dataclass(slots=True)
class MotionSample:
    """A minimal, model-agnostic motion container for evaluation."""

    motion: np.ndarray
    fps: float
    representation: MotionRepresentation
    source_path: str | None = None

    def validate(self, expected_ndim: int = 3) -> None:
        if self.motion.ndim != expected_ndim:
            raise ValueError(
                f"Expected motion ndim={expected_ndim}, got shape {self.motion.shape}."
            )
        if self.motion.shape[0] <= 0:
            raise ValueError("Motion must contain at least one frame.")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}.")
