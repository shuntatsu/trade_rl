#!/usr/bin/env python3
"""Run one tiny deterministic training job through the authoritative workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from trade_rl.data import write_market_dataset_files
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = ROOT / "examples" / "quickstart" / "training.json"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_template() -> dict[str, Any]:
    payload = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("quickstart training template must be a JSON object")
    return dict(payload)


def _smoke_config_payload(timesteps: int) -> dict[str, Any]:
    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:
        raise ValueError("timesteps must be a positive integer")
    payload = _load_template()
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("quickstart training template has no training object")
    training.update(
        {
            "asset_embedding_dim": 128,
            "batch_size": 32,
            "device": "cuda",
            "global_embedding_dim": 128,
            "n_envs": 4,
            "n_steps": 8,
            "policy_net_arch": [256, 256],
            "seeds": [0],
            "timesteps": timesteps,
        }
    )
    return payload


def build_smoke_config(timesteps: int) -> TrainingRunConfig:
    """Build the tiny run while retaining maintained CUDA/model dimensions."""

    return TrainingRunConfig.from_mapping(_smoke_config_payload(timesteps))


def _run_authoritative_training(
    *, config: Path, dataset: Path, artifacts: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "trade_rl.cli.app",
        "train",
        "run",
        "--config",
        str(config),
        "--dataset",
        str(dataset),
        "--output",
        str(artifacts),
        "--run-id",
        "gpu-training-smoke",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "authoritative training workflow failed: " + completed.stderr.strip()
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("authoritative training workflow returned no JSON")
    result = json.loads(lines[-1])
    if not isinstance(result, dict):
        raise RuntimeError("authoritative training result must be a JSON object")
    return dict(result)


def run_gpu_training_smoke(*, work_root: Path, timesteps: int) -> dict[str, object]:
    """Preflight CUDA, train one seed, and persist inspectable smoke evidence."""

    import torch

    preflight_module = _load_module(
        "training_cuda_preflight",
        Path(__file__).with_name("training_cuda_preflight.py"),
    )
    write_cuda_preflight_evidence: Callable[[Path, Any], dict[str, object]] = getattr(
        preflight_module, "write_cuda_preflight_evidence"
    )
    dataset_module = _load_module(
        "quickstart_dataset_builder",
        ROOT / "examples" / "quickstart" / "create_demo_dataset.py",
    )
    build_demo_dataset: Callable[[], Any] = getattr(
        dataset_module, "build_demo_dataset"
    )
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    preflight = write_cuda_preflight_evidence(
        work_root / "cuda-preflight.json", torch
    )

    dataset_path = work_root / "dataset"
    write_market_dataset_files(dataset_path, build_demo_dataset())
    config_payload = _smoke_config_payload(timesteps)
    config_path = work_root / "training.json"
    config_path.write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = TrainingRunConfig.from_json(config_path)
    result = _run_authoritative_training(
        config=config_path,
        dataset=dataset_path,
        artifacts=work_root / "artifacts",
    )

    artifact_path = Path(str(result["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    ensemble = json.loads((artifact_path / "ensemble.json").read_text(encoding="utf-8"))
    policy = json.loads(
        (artifact_path / "policy-loader.json").read_text(encoding="utf-8")
    )
    checkpoint = artifact_path / str(policy["members"][0])
    evidence: dict[str, object] = {
        "actual_timesteps": int(ensemble["actual_timesteps"]),
        "checkpoint": {
            "digest": ensemble["members"][0]["checkpoint_digest"],
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "cuda_preflight": preflight,
        "n_envs": config.training.n_envs,
        "policy": policy,
        "requested_timesteps": config.training.timesteps,
        "resolved_device": ensemble["resolved_device"],
        "schema": "gpu_training_smoke_evidence_v1",
    }
    evidence_path = work_root / "gpu-training-smoke.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("var/gpu-training-smoke"),
    )
    parser.add_argument("--timesteps", type=int, default=128)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    work_root = args.work_root if args.work_root.is_absolute() else ROOT / args.work_root
    evidence = run_gpu_training_smoke(
        work_root=work_root,
        timesteps=args.timesteps,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
