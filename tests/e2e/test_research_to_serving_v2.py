from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from trade_rl.data import write_market_dataset_files
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.domain.selection import PolicyMode
from trade_rl.integrations.sb3_serving import StableBaselines3PolicyLoader
from trade_rl.release.attestation import (
    ReleaseAttestation,
    default_attestation_path,
    write_release_attestation,
)
from trade_rl.serving.bundle import ServingBundleManifest, write_serving_bundle_manifest
from trade_rl.serving.runtime import RuntimeIdentityContract, ServingRuntime
from trade_rl.workflows.training_run import execute_training_run


def _dataset():
    n = 64
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n
    ) * np.timedelta64(1, "h")
    close = 100.0 * np.exp(np.arange(n, dtype=np.float64) * 0.001)
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=np.concatenate((close[:1], close[:-1])),
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=np.full(n, 1_000_000.0),
        funding_rate=np.zeros(n),
        tradable=np.ones(n, dtype=np.bool_),
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ),
    )


def _config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "training": {
                    "timesteps": 8,
                    "gamma": 0.99,
                    "seeds": [0],
                    "n_steps": 8,
                    "batch_size": 8,
                    "n_epochs": 1,
                    "asset_set_encoder": False,
                    "device": "cpu",
                },
                "environment": {
                    "episode_hours": 8.0,
                    "decision_hours": 1.0,
                    "episode_bars": 8,
                    "decision_every": 1,
                    "initial_capital": 1_000.0,
                    "initial_state_modes": ["cash"],
                },
                "risk": {
                    "max_gross": 1.0,
                    "max_abs_weight": 1.0,
                    "max_turnover": 2.0,
                },
                "reward": {
                    "scale": 1.0,
                    "baseline_window_hours": 4.0,
                    "baseline_minimum_history_hours": 1.0,
                },
                "trend": {
                    "fast_hours": 1.0,
                    "base_hours": 2.0,
                    "slow_hours": 3.0,
                    "fast_lookback": 1,
                    "base_lookback": 2,
                    "slow_lookback": 3,
                    "mode": "time_series",
                },
                "action": {"alpha_enabled": False, "n_factors": 0},
                "exports": {"onnx": False, "torchscript": False},
            }
        ),
        encoding="utf-8",
    )


def test_research_training_to_attested_runtime_prediction(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_market_dataset_files(dataset_root, _dataset())
    config_path = tmp_path / "training.json"
    _config(config_path)
    result = execute_training_run(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=tmp_path / "store",
        run_id="e2e",
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )

    run_root = result.path
    ensemble = json.loads((run_root / "ensemble.json").read_text(encoding="utf-8"))
    provenance = json.loads((run_root / "provenance.json").read_text(encoding="utf-8"))
    bundle_root = tmp_path / "bundle"
    (bundle_root / "members/member-000").mkdir(parents=True)
    shutil.copyfile(
        run_root / "members/member-000/policy.zip",
        bundle_root / "members/member-000/policy.zip",
    )
    shutil.copyfile(run_root / "policy-loader.json", bundle_root / "policy-loader.json")
    shutil.copyfile(run_root / "normalizer.json", bundle_root / "normalizer.json")

    manifest = ServingBundleManifest.build(
        root=bundle_root,
        dataset_id=ensemble["dataset_id"],
        action_schema=ensemble["action_schema"],
        observation_schema=ensemble["observation_schema"],
        observation_size=ensemble["observation_size"],
        environment_digest=ensemble["environment_digest"],
        initial_capital=ensemble["initial_capital"],
        policy_mode=PolicyMode.RESIDUAL_POLICY,
        policy_digest=ensemble["digest"],
        signal_digest="1" * 64,
        selection_digest="2" * 64,
        release_digest=None,
        artifact_paths=(
            "members/member-000/policy.zip",
            "normalizer.json",
            "policy-loader.json",
        ),
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
        action_size=ensemble["action_size"],
        action_names=tuple(ensemble["action_names"]),
        action_spec_digest=ensemble["action_spec_digest"],
        normalizer_digest=ensemble.get("normalizer_digest"),
    )
    write_serving_bundle_manifest(bundle_root, manifest)
    attestation = ReleaseAttestation.create(
        bundle_digest=manifest.bundle_digest,
        dataset_id=manifest.dataset_id,
        selection_evaluation_digest=manifest.selection_digest,
        gate_evaluation_digest="4" * 64,
        gate_evidence_digest="5" * 64,
        selected_policy_digest=manifest.policy_digest,
        git_commit=provenance["git_commit"],
        dependency_digest=provenance["digest"],
        approver="e2e-test",
        approved_at=datetime(2026, 7, 14, tzinfo=UTC),
        key_id="e2e-release-key",
        signing_key=b"e2e-release-signing-key",
    )
    write_release_attestation(default_attestation_path(bundle_root), attestation)

    runtime = ServingRuntime(
        StableBaselines3PolicyLoader(),
        trusted_attestation_keys={"e2e-release-key": b"e2e-release-signing-key"},
        identity_contract=RuntimeIdentityContract(
            environment_digest=manifest.environment_digest,
            action_names=manifest.action_names,
            action_spec_digest=manifest.action_spec_digest,
            normalizer_digest=manifest.normalizer_digest,
        ),
    )
    runtime.activate(bundle_root)
    action = runtime.predict(np.zeros(manifest.observation_size, dtype=np.float32))

    assert action.shape == (manifest.action_size,)
    assert np.isfinite(action).all()
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
