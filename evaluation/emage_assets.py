from __future__ import annotations

from pathlib import Path

FGD_MODEL_FILENAME = "AESKConv_240_100.bin"
BC_MMAE_FILENAME = "mean_vel_smplxflame_30.npy"
SMPLX_MODEL_RELATIVE_PATH = Path("smplx_models") / "smplx" / "SMPLX_NEUTRAL_2020.npz"

FGD_MODEL_URL = (
    "https://huggingface.co/H-Liu1997/emage_evaltools/resolve/main/"
    "AESKConv_240_100.bin"
)
BC_MMAE_URL = (
    "https://huggingface.co/H-Liu1997/emage_evaltools/resolve/main/"
    "mean_vel_smplxflame_30.npy"
)
SMPLX_MODEL_URL = (
    "https://huggingface.co/H-Liu1997/emage_evaltools/resolve/main/"
    "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"
)


def ensure_trailing_sep(path: str | Path) -> str:
    text = str(path)
    if not text.endswith("/"):
        text = text + "/"
    return text


def resolve_required_paths(weights_root: str | Path) -> dict[str, Path]:
    root = Path(weights_root)
    return {
        "fgd_model": root / FGD_MODEL_FILENAME,
        "bc_mmae": root / BC_MMAE_FILENAME,
        "smplx_model": root / SMPLX_MODEL_RELATIVE_PATH,
    }


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            head = f.read(120)
    except UnicodeDecodeError:
        return False
    return "git-lfs.github.com/spec/v1" in head


def validate_assets_for_fgd(weights_root: str | Path) -> list[str]:
    errors: list[str] = []
    paths = resolve_required_paths(weights_root)
    for key in ("fgd_model", "smplx_model"):
        path = paths[key]
        if not path.exists():
            errors.append(f"Missing required file: {path}")
            continue
        if is_lfs_pointer(path):
            errors.append(f"File is a Git LFS pointer, not real content: {path}")
    return errors


def validate_assets_for_bc(weights_root: str | Path) -> list[str]:
    errors: list[str] = []
    path = resolve_required_paths(weights_root)["bc_mmae"]
    if not path.exists():
        errors.append(f"Missing required file: {path}")
    elif is_lfs_pointer(path):
        errors.append(f"File is a Git LFS pointer, not real content: {path}")
    return errors
