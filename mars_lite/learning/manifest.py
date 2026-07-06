"""
モデルマニフェスト自動生成モジュール
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def get_git_commit_hash() -> str:
    """Git コミットハッシュを取得。失敗時は 'unknown'"""
    try:
        repo_dir = Path(__file__).resolve().parents[2]
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def calculate_data_hash(fs: Any) -> str:
    """
    FeatureSet 内の主要配列 (close, features, global_features) から一貫性のあるハッシュ値を計算する。
    """
    hasher = hashlib.sha256()
    for attr_name in ["close", "features", "global_features"]:
        arr = getattr(fs, attr_name, None)
        if arr is not None and isinstance(arr, np.ndarray):
            # np.ascontiguousarray() でメモリレイアウトを一貫させる
            hasher.update(np.ascontiguousarray(arr).tobytes())
    return hasher.hexdigest()


def generate_and_save_manifest(
    output_filepath: str,
    fs: Any | None,
    hyperparams: dict[str, Any],
    seed: int,
    additional_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    モデルマニフェストを生成し、JSONファイルとして保存する。
    """
    git_hash = get_git_commit_hash()
    data_hash = calculate_data_hash(fs) if fs is not None else "none"

    manifest = {
        "git_commit": git_hash,
        "data_hash": data_hash,
        "hyperparameters": hyperparams,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    if additional_metadata:
        manifest.update(additional_metadata)

    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


def verify_reproducible_sharpe(
    baseline_manifest: dict[str, Any],
    replay_manifest: dict[str, Any],
    tolerance: float = 0.1,
) -> dict[str, Any]:
    """
    Compare two model manifests and their OOS Sharpe values for reproducibility.

    The manifests must describe the same code/data/hyperparameters/seed, and the
    OOS Sharpe drift must stay inside the configured tolerance.
    """
    required_fields = ["git_commit", "data_hash", "hyperparameters", "seed"]
    mismatched_fields = [
        field
        for field in required_fields
        if baseline_manifest.get(field) != replay_manifest.get(field)
    ]
    baseline_sharpe = float(baseline_manifest["oos_sharpe"])
    replay_sharpe = float(replay_manifest["oos_sharpe"])
    sharpe_diff = abs(replay_sharpe - baseline_sharpe)
    return {
        "reproducible": not mismatched_fields and sharpe_diff <= tolerance,
        "mismatched_fields": mismatched_fields,
        "baseline_sharpe": baseline_sharpe,
        "replay_sharpe": replay_sharpe,
        "sharpe_diff": sharpe_diff,
        "tolerance": tolerance,
    }
