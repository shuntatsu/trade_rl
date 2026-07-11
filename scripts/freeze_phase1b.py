"""
Phase 1b 凍結マニフェスト生成スクリプト
環境バージョン、関連ファイルの SHA-256 チェックサムを計算し保存する。
"""

import hashlib
import json
import platform
import sys
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "N/A"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    import stable_baselines3 as sb3
    import torch

    files_to_hash = [
        Path("mars_lite/configs/p0_phase1b_config.yaml"),
        Path("scripts/p0_phase1b_diagnostic.py"),
        Path("scripts/p0_phase1b_threshold_sweep.py"),
        Path("scripts/p0_phase1b_summarize_confirmation.py"),
        Path("output/p0_phase1b_grid5x5/phase1b_report.json"),
        Path("output/p0_phase1b_confirm_paired10/phase1b_report.json"),
    ]

    file_hashes = {}
    for fp in files_to_hash:
        file_hashes[str(fp)] = compute_sha256(fp)

    manifest = {
        "freeze_tag": "p0-phase1b-passed",
        "commit_sha": "f7d1799e299c6a4c84e04a143c71bad8e87d4624",
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "sb3_version": sb3.__version__,
        },
        "fixed_spec": {
            "min_sortino": 0.5,
            "min_total_return": 0.005,
            "max_drawdown": 0.12,
            "eval_deterministic": True,
            "data_split": {
                "train": "0.0 - 0.50",
                "val_select": "0.50 - 0.68",
                "val_confirm": "0.68 - 0.84",
                "test": "0.84 - 1.00",
                "purge_bars": 24,
            },
        },
        "file_sha256": file_hashes,
    }

    out_path = Path("output/p0_phase1b_freeze_manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Phase 1b 凍結マニフェストを作成しました: {out_path}")
    for k, v in file_hashes.items():
        print(f"  {k}: {v[:16]}...")


if __name__ == "__main__":
    main()
