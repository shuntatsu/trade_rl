from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mars_lite.eval.residual_walk_forward import save_residual_walk_forward_report


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    root: str | Path,
    *,
    run_id: str,
    config: dict[str, object],
) -> dict[str, Any]:
    run_root = Path(root)
    if not run_root.is_dir():
        raise ValueError("run root must be an existing directory")
    entries: list[dict[str, object]] = []
    for path in sorted(item for item in run_root.rglob("*") if item.is_file()):
        relative = path.relative_to(run_root).as_posix()
        if relative == "run_manifest.json":
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ValueError("run manifest requires at least one artifact")
    return {
        "schema": "residual_wf_run_manifest_v1",
        "run_id": run_id,
        "config": config,
        "files": entries,
    }


def write_run_manifest(
    root: str | Path,
    *,
    run_id: str,
    config: dict[str, object],
) -> dict[str, Any]:
    run_root = Path(root)
    manifest = build_run_manifest(run_root, run_id=run_id, config=config)
    save_residual_walk_forward_report(run_root / "run_manifest.json", manifest)
    return manifest
