#!/usr/bin/env python3
"""Run one tiny deterministic training job through the authoritative workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from trade_rl.data import MarketDataset, write_market_dataset_files
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.rl.training import gamma_from_half_life
from trade_rl.workflows.training_run import TrainingRunConfig

ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = ROOT / "examples" / "quickstart" / "training.json"
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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


def _packaged_git_provenance() -> tuple[str, bool]:
    commit = os.environ.get("TRADE_RL_GIT_COMMIT", "")
    if not _GIT_COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(
            "TRADE_RL_GIT_COMMIT must be a 40-character lowercase Git commit"
        )
    dirty = os.environ.get("TRADE_RL_GIT_DIRTY")
    if dirty not in {"true", "false"}:
        raise ValueError("TRADE_RL_GIT_DIRTY must be exactly true or false")
    return commit, dirty == "true"


def _smoke_config_payload(timesteps: int) -> dict[str, Any]:
    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:
        raise ValueError("timesteps must be a positive integer")
    payload = _load_template()
    git_commit, git_dirty = _packaged_git_provenance()
    payload["git_commit"] = git_commit
    payload["git_dirty"] = git_dirty
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("quickstart training template has no training object")
    training.update(
        {
            "asset_set_encoder": False,
            "batch_size": 32,
            "device": "cuda",
            "n_envs": 4,
            "n_steps": 8,
            "policy": "MultiInputPolicy",
            "policy_net_arch": [384, 256, 128],
            "value_net_arch": [512, 384, 256],
            "sequence_encoder": True,
            "sequence_d_model": 336,
            "sequence_attention_heads": 8,
            "sequence_attention_layers": 2,
            "sequence_dropout": 0.05,
            "max_policy_parameters": 12_000_000,
            "max_rollout_buffer_bytes": 268_435_456,
            "seeds": [0],
            "timesteps": timesteps,
            "behavior_cloning_epochs": 1,
            "behavior_cloning_batch_size": 32,
            "behavior_cloning_validation_fraction": 0.1,
            "decision_hours": 0.25,
            "discount_half_life_hours": 168.0,
            "gamma": gamma_from_half_life(decision_hours=0.25, half_life_hours=168.0),
        }
    )
    payload["environment"] = {
        "episode_hours": 1.0,
        "decision_hours": 0.25,
        "initial_capital": 100_000.0,
        "finite_horizon_observation": True,
        "initial_state_modes": ["cash"],
        "structured_sequence_observation": True,
        "sequence_windows": [["15m", 96], ["1h", 168], ["4h", 120], ["1d", 60]],
    }
    payload["trend"] = {
        "fast_hours": 4.0,
        "base_hours": 12.0,
        "slow_hours": 24.0,
        "mode": "time_series",
        "signal_scale": 0.05,
    }
    payload["risk"]["max_turnover"] = None
    payload["action"] = {
        "mode": "target_weight",
        "alpha_enabled": False,
        "risk_tilt_enabled": False,
        "n_factors": 0,
        "target_weight_count": 1,
        "validation_mode": "clip",
    }
    return payload


def _build_sequence_smoke_dataset(n_bars: int = 5_680) -> MarketDataset:
    if n_bars < 5_680:
        raise ValueError("sequence smoke dataset needs at least 5680 bars")
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    phase = np.arange(n_bars, dtype=np.float64)
    returns = 0.00005 + 0.0004 * np.sin(phase / 47.0)
    close = 30_000.0 * np.exp(np.cumsum(returns))
    open_price = np.concatenate(([close[0]], close[:-1]))
    spread = 0.001 + 0.0002 * np.cos(phase / 19.0)
    features = np.stack(
        tuple(
            np.sin(phase / float(11 + index % 97))
            + 0.1 * np.cos(phase / float(7 + index % 43))
            for index in range(len(specs))
        ),
        axis=1,
    ).astype(np.float32)[:, None, :]
    timestamps = np.datetime64("2025-01-01T00:15:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(15, "m")
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=features,
        global_features=np.stack(
            (np.sin(phase / 97.0), np.cos(phase / 193.0)), axis=1
        ).astype(np.float32),
        open=open_price[:, None],
        high=(np.maximum(open_price, close) * (1.0 + spread))[:, None],
        low=(np.minimum(open_price, close) * (1.0 - spread))[:, None],
        close=close[:, None],
        volume=(1_000.0 + 100.0 * np.sin(phase / 13.0))[:, None],
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones(features.shape, dtype=np.bool_),
        feature_names=tuple(spec.name for spec in specs),
        global_feature_names=("market_cycle", "risk_cycle"),
        periods_per_year=35_040,
        fee_rate=np.full((n_bars, 1), 0.0005, dtype=np.float64),
        spread_rate=np.full((n_bars, 1), 0.0002, dtype=np.float64),
        max_participation_rate=np.full((n_bars, 1), 0.05, dtype=np.float64),
        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
    )
    return dataset.with_content_identity({"source": "sequence-gpu-smoke-v1"})


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
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
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
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    preflight = write_cuda_preflight_evidence(work_root / "cuda-preflight.json", torch)

    dataset_path = work_root / "dataset"
    write_market_dataset_files(dataset_path, _build_sequence_smoke_dataset())
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
    serving_support = json.loads(
        (artifact_path / "serving-support.json").read_text(encoding="utf-8")
    )
    if serving_support.get("status") != "unsupported":
        raise RuntimeError("structured smoke must fail closed for flat serving")
    checkpoint = artifact_path / "members" / "member-000" / "policy.zip"
    evidence: dict[str, object] = {
        "actual_timesteps": int(ensemble["actual_timesteps"]),
        "checkpoint": {
            "digest": ensemble["members"][0]["checkpoint_digest"],
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "cuda_preflight": preflight,
        "n_envs": config.training.n_envs,
        "behavior_cloning_epochs": config.training.behavior_cloning_epochs,
        "serving_support": serving_support,
        "requested_timesteps": config.training.timesteps,
        "resolved_device": ensemble["resolved_device"],
        "schema": "gpu_sequence_target_oracle_bc_training_smoke_v4",
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
    work_root = (
        args.work_root if args.work_root.is_absolute() else ROOT / args.work_root
    )
    evidence = run_gpu_training_smoke(
        work_root=work_root,
        timesteps=args.timesteps,
    )
    print(json.dumps(evidence, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
