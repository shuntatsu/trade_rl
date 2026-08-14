from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    write_training_run_manifest,
)
from trade_rl.data import write_market_dataset_files
from trade_rl.data.market import MarketDataset


def checkpoint_payload() -> dict[str, object]:
    return {
        "schema_version": "checkpoint_selection_v2_seed_aware",
        "checkpoint_range": [100, 120],
        "candidates": [
            {
                "evaluation_digest": "a" * 64,
                "policy_digest": "b" * 64,
                "score": math.log1p(0.05),
                "seed": 3,
            },
            {
                "evaluation_digest": "c" * 64,
                "policy_digest": "d" * 64,
                "score": math.log1p(0.02),
                "seed": 11,
            },
        ],
        "seed_finalists": [
            {
                "checkpoint_evaluation_digest": "a" * 64,
                "checkpoint_score": math.log1p(0.05),
                "policy_digest": "b" * 64,
                "seed": 3,
            }
        ],
    }


def telemetry_record(sequence: int, *, seed: int = 7):
    from trade_rl.telemetry.training import TrainingTelemetryRecord

    return TrainingTelemetryRecord(
        sequence=sequence,
        recorded_at="2026-07-21T08:00:00+00:00",
        global_step=sequence * 32,
        environment_step=sequence,
        seed=seed,
        environment_id=0,
        event_type="rollout",
        market_index=100 + sequence,
        market_time="2026-07-21T08:00:00.000000000",
        symbol="BTCUSDT",
        open=67_500.0,
        high=67_900.0,
        low=67_400.0,
        close=67_842.3,
        action=(0.4,),
        executed_target=(0.4,),
        weights_before=(0.2,),
        weights_after=(0.4,),
        portfolio_value=101_342.85,
        baseline_portfolio_value=100_400.0,
        reward=0.214,
        drawdown=0.0086,
        interval_cost=4.25,
        interval_return=0.0012,
        risk_reasons=(),
        emergency_deleverage=False,
        terminated=False,
        truncated=False,
    )


def telemetry_stream_path(
    tmp_path: Path,
    run_id: str,
    seed: int,
    *,
    namespace: str = ".staging",
) -> Path:
    return (
        tmp_path
        / "research"
        / namespace
        / run_id
        / f"seed-{seed}"
        / "telemetry"
        / "training-telemetry.jsonl"
    )


def write_dataset(root: Path, *, symbol: str = "BTCUSDT") -> MarketDataset:
    n_bars = 12
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")
    close = np.linspace(100.0, 111.0, n_bars, dtype=np.float64)[:, None]
    dataset = MarketDataset(
        dataset_id="a" * 64,
        symbols=(symbol,),
        timestamps=timestamps,
        features=np.linspace(0.0, 1.0, n_bars, dtype=np.float32)[:, None, None],
        global_features=np.linspace(1.0, 2.0, n_bars, dtype=np.float32)[:, None],
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 10_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("momentum",),
        global_feature_names=("market",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, 1), 0.0005),
        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
        cash_rate=np.linspace(0.0, 0.001, n_bars),
    ).with_content_identity()
    write_market_dataset_files(root, dataset)
    return dataset


def write_run(
    store_root: Path,
    *,
    run_id: str = "run-001",
    dataset_id: str = "a" * 64,
    algorithm: str = "ppo",
    with_walk_forward: bool = True,
) -> Path:
    root = store_root / "runs" / run_id
    member = root / "members" / "member-000" / "policy.zip"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"checkpoint")
    (root / "ensemble.json").write_text('{"schema":"ensemble"}', encoding="utf-8")
    (root / "training-config.json").write_text(
        json.dumps({"training": {"algorithm": algorithm}}), encoding="utf-8"
    )
    artifact_paths = [
        "ensemble.json",
        "members/member-000/policy.zip",
        "training-config.json",
    ]
    if with_walk_forward:
        payload = {
            "selected_metrics": {
                "sharpe": 1.25,
                "max_drawdown": 0.12,
                "total_return": 0.34,
            },
            "baseline_metrics": {
                "sharpe": 0.8,
                "max_drawdown": 0.18,
                "total_return": 0.22,
            },
            "folds": [
                {
                    "fold_index": 0,
                    "selected_returns": [0.01, -0.005, 0.02],
                    "baseline_returns": [0.005, 0.0, 0.01],
                }
            ],
            "production_status": "NO-GO",
            "schema_version": "market_walk_forward_run_v5_deployable_ensemble",
        }
        (root / "walk-forward.json").write_text(json.dumps(payload), encoding="utf-8")
        artifact_paths.append("walk-forward.json")
    manifest = TrainingRunManifest.build(
        root=root,
        run_id=run_id,
        dataset_id=dataset_id,
        environment_digest="b" * 64,
        ensemble_digest="c" * 64,
        training_config_digest="d" * 64,
        provenance_digest="e" * 64,
        artifact_paths=tuple(sorted(artifact_paths)),
        created_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 7, 19, 12, 5, tzinfo=UTC),
    )
    write_training_run_manifest(root, manifest)
    return root
