#!/usr/bin/env python3
"""Robust downloader for large Hugging Face datasets.

This script downloads dataset files one-by-one with retry and backoff so that
large public datasets can be resumed safely on unstable cloud machines.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError


RETRYABLE_STATUS_CODES = {408, 409, 423, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERROR_NAMES = {
    "RemoteProtocolError",
    "ReadTimeout",
    "ConnectTimeout",
    "ReadError",
    "ConnectError",
    "ProtocolError",
    "ChunkedEncodingError",
}


@dataclass
class DownloadStats:
    total_files: int = 0
    completed_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Hugging Face dataset robustly with retries, backoff, "
            "resume support, and progress manifest."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Dataset repo id, e.g. H-Liu1997/BEAT2",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Directory to store downloaded files.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help=(
            "Optional glob to include. Can be passed multiple times, "
            'e.g. --include "beat_*_v2.0.0/smplxflame_30/*"'
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Optional glob to exclude. Can be passed multiple times.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision or branch. Defaults to main.",
    )
    parser.add_argument(
        "--sleep-between-files",
        type=float,
        default=0.5,
        help="Seconds to sleep between files to reduce request bursts.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=50,
        help="Max retries per file before failing the run.",
    )
    parser.add_argument(
        "--base-backoff",
        type=float,
        default=5.0,
        help="Base seconds for exponential backoff.",
    )
    parser.add_argument(
        "--max-backoff",
        type=float,
        default=300.0,
        help="Maximum backoff sleep in seconds.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. If omitted, uses HF_TOKEN from env or cached login.",
    )
    parser.add_argument(
        "--manifest-name",
        default=".download_manifest.json",
        help="Progress manifest filename stored under local-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list files that would be downloaded.",
    )
    return parser.parse_args()


def should_retry(exc: Exception) -> tuple[bool, str]:
    if isinstance(exc, HfHubHTTPError):
        status_code = getattr(exc.response, "status_code", None)
        if status_code in RETRYABLE_STATUS_CODES:
            return True, f"http_status={status_code}"
        return False, f"http_status={status_code}"

    name = exc.__class__.__name__
    if name in RETRYABLE_ERROR_NAMES:
        return True, name

    cause = exc.__cause__
    while cause is not None:
        cause_name = cause.__class__.__name__
        if cause_name in RETRYABLE_ERROR_NAMES:
            return True, cause_name
        cause = cause.__cause__

    text = str(exc)
    retryable_phrases = [
        "Too Many Requests",
        "Server disconnected without sending a response",
        "peer closed connection without sending complete message body",
        "Connection reset by peer",
        "temporarily unavailable",
    ]
    if any(phrase in text for phrase in retryable_phrases):
        return True, "message_match"

    return False, name


def compute_backoff(attempt: int, base_backoff: float, max_backoff: float) -> float:
    backoff = min(max_backoff, base_backoff * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0, min(5.0, backoff * 0.2))
    return backoff + jitter


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"completed": {}, "failed": {}}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True, sort_keys=True)
    tmp_path.replace(path)


def list_files(
    api: HfApi,
    repo_id: str,
    revision: str,
    token: str | None,
    include: list[str],
    exclude: list[str],
) -> list[str]:
    files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        token=token,
    )

    if include:
        from fnmatch import fnmatch

        files = [path for path in files if any(fnmatch(path, pat) for pat in include)]
    if exclude:
        from fnmatch import fnmatch

        files = [path for path in files if not any(fnmatch(path, pat) for pat in exclude)]

    return sorted(files)


def iter_pending_files(files: Iterable[str], manifest: dict) -> Iterable[str]:
    completed = manifest.get("completed", {})
    for path in files:
        if path in completed:
            continue
        yield path


def download_one(
    repo_id: str,
    revision: str,
    token: str | None,
    local_dir: Path,
    file_path: str,
    max_retries: int,
    base_backoff: float,
    max_backoff: float,
) -> None:
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=file_path,
                revision=revision,
                token=token,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                force_download=False,
            )
            return
        except Exception as exc:  # noqa: BLE001
            retry, reason = should_retry(exc)
            if not retry or attempt >= max_retries:
                raise RuntimeError(
                    f"Failed to download {file_path} after {attempt} attempt(s): {exc}"
                ) from exc

            sleep_s = compute_backoff(attempt, base_backoff, max_backoff)
            print(
                f"[WARN] {file_path} attempt {attempt}/{max_retries} failed "
                f"({reason}). Sleeping {sleep_s:.1f}s before retry.",
                flush=True,
            )
            time.sleep(sleep_s)


def main() -> int:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = local_dir / args.manifest_name

    api = HfApi(token=token)
    manifest = load_manifest(manifest_path)

    print(f"[INFO] Listing files for dataset {args.repo_id}@{args.revision}", flush=True)
    files = list_files(
        api=api,
        repo_id=args.repo_id,
        revision=args.revision,
        token=token,
        include=args.include,
        exclude=args.exclude,
    )

    stats = DownloadStats(total_files=len(files))
    pending_files = list(iter_pending_files(files, manifest))
    print(
        f"[INFO] Total files: {len(files)}, already completed: {len(files) - len(pending_files)}, "
        f"pending: {len(pending_files)}",
        flush=True,
    )

    if args.dry_run:
        for file_path in pending_files:
            print(file_path)
        return 0

    for index, file_path in enumerate(pending_files, start=1):
        print(
            f"[{index}/{len(pending_files)}] Downloading {file_path}",
            flush=True,
        )
        try:
            download_one(
                repo_id=args.repo_id,
                revision=args.revision,
                token=token,
                local_dir=local_dir,
                file_path=file_path,
                max_retries=args.max_retries,
                base_backoff=args.base_backoff,
                max_backoff=args.max_backoff,
            )
            manifest.setdefault("completed", {})[file_path] = {
                "completed_at": int(time.time()),
            }
            if "failed" in manifest and file_path in manifest["failed"]:
                del manifest["failed"][file_path]
            save_manifest(manifest_path, manifest)
            stats.completed_files += 1
        except Exception as exc:  # noqa: BLE001
            manifest.setdefault("failed", {})[file_path] = {
                "failed_at": int(time.time()),
                "error": str(exc),
            }
            save_manifest(manifest_path, manifest)
            stats.failed_files += 1
            print(f"[ERROR] {exc}", flush=True)
            return 1

        if args.sleep_between_files > 0:
            time.sleep(args.sleep_between_files)

    print(
        f"[DONE] Completed {stats.completed_files} file(s). "
        f"Manifest: {manifest_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
