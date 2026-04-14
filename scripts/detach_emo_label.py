#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Known BEAT2 mismatch stems (kept as comments for traceability):
# BEAT2 缺少以下的.npz文件，这里就先行忽略
# 22_luqi_0_9_9
# 22_luqi_0_10_10
# 22_luqi_0_11_11
# 22_luqi_0_12_12
# 22_luqi_0_13_13
# 22_luqi_0_14_14
# 22_luqi_0_15_15
# 22_luqi_0_16_16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detach emotion labels from BEAT and copy matched CSV files into "
            "BEAT2/beat_english_v2.0.0/csv after cross-folder filename checks."
        )
    )
    parser.add_argument(
        "--beat-root",
        default="datasets/BEAT/beat_english_v0.2.1",
        help="Root directory of BEAT dataset (script will search CSV recursively).",
    )
    parser.add_argument(
        "--beat2-root",
        default="datasets/BEAT2/beat_english_v2.0.0",
        help="Root directory of BEAT2 English dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files under BEAT2 csv folder.",
    )
    parser.add_argument(
        "--ignore-folders",
        nargs="*",
        default=["weights", "csv"],
        help="Subfolders to ignore when checking BEAT2 filename prefixes.",
    )
    return parser.parse_args()


def collect_stems(folder: Path) -> set[str]:
    stems: set[str] = set()
    for file_path in folder.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        stems.add(file_path.stem)
    return stems


def resolve_beat2_stems(beat2_root: Path, ignore_folders: set[str]) -> set[str]:
    subfolders = sorted(
        p for p in beat2_root.iterdir() if p.is_dir() and p.name not in ignore_folders and not p.name.startswith(".")
    )
    if not subfolders:
        raise RuntimeError(f"No valid subfolders found under {beat2_root}")

    folder_to_stems: dict[str, set[str]] = {folder.name: collect_stems(folder) for folder in subfolders}
    all_sets = list(folder_to_stems.values())
    union_stems = set().union(*all_sets)
    common_stems = set.intersection(*all_sets)

    print("[INFO] BEAT2 stem counts by folder:")
    for folder_name in sorted(folder_to_stems):
        print(f"  - {folder_name}: {len(folder_to_stems[folder_name])}")
    print(f"[INFO] Shared stems across all folders: {len(common_stems)}")
    print(f"[INFO] Union stems across all folders: {len(union_stems)}")

    all_good = True
    for folder_name, stems in sorted(folder_to_stems.items()):
        missing = sorted(union_stems - stems)
        extra = sorted(stems - common_stems)
        if missing or extra:
            all_good = False
            print(f"[MISMATCH] Folder: {folder_name}")
            if missing:
                print(f"  Missing ({len(missing)}):")
                for name in missing:
                    print(f"    - {name}")
            if extra:
                print(f"  Extra ({len(extra)}):")
                for name in extra:
                    print(f"    + {name}")

    if all_good:
        return union_stems

    print(
        "[WARN] BEAT2 subfolders are inconsistent. "
        f"Continue with intersection strategy using {len(common_stems)} shared stems."
    )
    if len(common_stems) == 0:
        raise RuntimeError("No shared stems across BEAT2 subfolders, cannot continue.")
    return common_stems


def build_csv_index(beat_root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    csv_files = sorted(p for p in beat_root.rglob("*.csv") if p.is_file())
    for csv_file in csv_files:
        stem = csv_file.stem
        if stem in index:
            duplicates.setdefault(stem, [index[stem]]).append(csv_file)
            continue
        index[stem] = csv_file
    return index, duplicates


def main() -> int:
    args = parse_args()
    beat_root = Path(args.beat_root).expanduser().resolve()
    beat2_root = Path(args.beat2_root).expanduser().resolve()
    ignore_folders = set(args.ignore_folders)

    if not beat_root.exists():
        print(f"[ERROR] BEAT root does not exist: {beat_root}")
        return 1
    if not beat2_root.exists():
        print(f"[ERROR] BEAT2 root does not exist: {beat2_root}")
        return 1

    print(f"[INFO] Checking BEAT2 stem consistency under: {beat2_root}")
    stems = resolve_beat2_stems(beat2_root, ignore_folders)
    print(f"[INFO] Stems selected for CSV detach: {len(stems)}")

    print(f"[INFO] Indexing BEAT CSV files under: {beat_root}")
    csv_index, duplicates = build_csv_index(beat_root)
    if duplicates:
        print("[ERROR] Duplicate CSV stems found in BEAT. Ambiguous mapping:")
        for stem, paths in sorted(duplicates.items()):
            print(f"  - {stem}")
            for p in paths:
                print(f"      {p}")
        return 1

    missing_in_beat = sorted(stem for stem in stems if stem not in csv_index)
    if missing_in_beat:
        print("[ERROR] Missing CSV files in BEAT for these stems:")
        for stem in missing_in_beat:
            print(f"  - {stem}")
        print("[STOP] Terminated without copying.")
        return 1

    target_dir = beat2_root / "csv"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0

    for stem in sorted(stems):
        src = csv_index[stem]
        dst = target_dir / f"{stem}.csv"
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    print(f"[DONE] CSV detach completed. copied={copied}, skipped={skipped}, target={target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
