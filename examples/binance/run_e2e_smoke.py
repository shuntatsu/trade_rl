#!/usr/bin/env python3
"""Run the maintained public Binance dataset, training, and walk-forward smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_DATA_BINANCE_COMMAND = ("data", "binance")
_TRAIN_RUN_COMMAND = ("train", "run")
_WALK_FORWARD_RUN_COMMAND = ("walk-forward", "run")


def _run_cli(arguments: list[str], *, root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "trade_rl.cli.app", *arguments]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed: "
            f"{command!r}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"command produced no JSON output: {command!r}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"command did not end with JSON: {command!r}\n{completed.stdout}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"command JSON result must be an object: {command!r}")
    return dict(payload)


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"dataset manifest does not exist: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"dataset manifest must be an object: {manifest}")
    return dict(payload)


def _require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"required artifact file is missing or empty: {path}")


def _data_arguments(
    *,
    start_time: str,
    end_time: str,
    output: Path,
) -> list[str]:
    return [
        *_DATA_BINANCE_COMMAND,
        "--market",
        "usds-m",
        "--symbol",
        "BTCUSDT",
        "--interval",
        "1h",
        "--start-time",
        start_time,
        "--end-time",
        end_time,
        "--transport",
        "vision",
        "--tick-size",
        "0.1",
        "--lot-size",
        "0.001",
        "--minimum-notional",
        "5",
        "--listed-at",
        "2019-09-08T00:00:00Z",
        "--output",
        str(output),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-time",
        default="2026-06-01T00:00:00Z",
    )
    parser.add_argument(
        "--end-time",
        default="2026-06-29T00:00:00Z",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("var/binance-live-smoke"),
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    work_root = args.work_root
    if not work_root.is_absolute():
        work_root = repository_root / work_root
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    dataset_a = work_root / "dataset-a"
    dataset_b = work_root / "dataset-b"
    data_a = _run_cli(
        _data_arguments(
            start_time=args.start_time,
            end_time=args.end_time,
            output=dataset_a,
        ),
        root=repository_root,
    )
    data_b = _run_cli(
        _data_arguments(
            start_time=args.start_time,
            end_time=args.end_time,
            output=dataset_b,
        ),
        root=repository_root,
    )
    manifest_a = _read_manifest(dataset_a)
    manifest_b = _read_manifest(dataset_b)
    _require_file(dataset_a / "arrays.npz")
    _require_file(dataset_b / "arrays.npz")
    if data_a.get("dataset_id") != data_b.get("dataset_id"):
        raise RuntimeError("fixed Binance range produced different dataset IDs")
    if manifest_a.get("dataset_id") != manifest_b.get("dataset_id"):
        raise RuntimeError("published Binance manifests have different dataset IDs")
    if data_a.get("artifact_digest") != data_b.get("artifact_digest"):
        raise RuntimeError("fixed Binance range produced different artifact digests")
    if data_a.get("n_bars") != 672:
        raise RuntimeError(f"expected 672 hourly bars, observed {data_a.get('n_bars')}")

    artifact_root = work_root / "artifacts"
    training = _run_cli(
        [
            *_TRAIN_RUN_COMMAND,
            "--config",
            str(repository_root / "examples/binance/training-smoke.json"),
            "--dataset",
            str(dataset_a),
            "--output",
            str(artifact_root),
            "--run-id",
            "binance-training-smoke",
        ],
        root=repository_root,
    )
    training_path = Path(str(training["artifact_path"]))
    if not training_path.is_absolute():
        training_path = repository_root / training_path
    _require_file(training_path / "run.json")
    _require_file(training_path / "ensemble.json")
    _require_file(training_path / "members/member-000/policy.zip")

    walk_forward = _run_cli(
        [
            *_WALK_FORWARD_RUN_COMMAND,
            "--config",
            str(repository_root / "examples/binance/walk-forward-smoke.json"),
            "--dataset",
            str(dataset_a),
            "--output",
            str(artifact_root),
            "--run-id",
            "binance-walk-forward-smoke",
        ],
        root=repository_root,
    )
    walk_forward_path = Path(str(walk_forward["artifact_path"]))
    if not walk_forward_path.is_absolute():
        walk_forward_path = repository_root / walk_forward_path
    _require_file(walk_forward_path / "run.json")

    summary = {
        "dataset": data_a,
        "dataset_manifest": manifest_a,
        "production_status": "NO-GO",
        "schema": "binance_e2e_smoke_result_v1",
        "training": training,
        "walk_forward": walk_forward,
    }
    summary_path = work_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
