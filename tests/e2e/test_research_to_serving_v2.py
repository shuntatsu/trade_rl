from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

pytest.importorskip("stable_baselines3")

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.confirmation import write_confirmation_evidence
from trade_rl.evaluation.offline_confirmation import create_fresh_confirmation_evidence
from trade_rl.integrations.sb3_serving import StableBaselines3PolicyLoader
from trade_rl.release.asymmetric import PublicVerificationKey
from trade_rl.release.attestation import (
    default_attestation_path,
    write_release_attestation,
)
from trade_rl.release.offline_approval import create_release_attestation
from trade_rl.release.offline_signing import public_key_bytes
from trade_rl.serving.package import package_selected_training_run
from trade_rl.serving.runtime import RuntimeIdentityContract, ServingRuntime
from trade_rl.workflows.offline_selection_approval import create_selection_authorization
from trade_rl.workflows.selection_authorization import (
    SelectionProposal,
    write_selection_authorization,
    write_selection_proposal,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    execute_training_run,
    normalize_training_run_config,
)


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
                    "seeds": [0, 1],
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
                "git_commit": "a" * 40,
                "git_dirty": False,
            }
        ),
        encoding="utf-8",
    )


def _key_store(path: Path, key: PublicVerificationKey) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "public_verification_key_store_v1",
                "keys": [
                    {
                        "algorithm": key.algorithm,
                        "key_id": key.key_id,
                        "public_key": base64.b64encode(bytes(key.public_key)).decode(
                            "ascii"
                        ),
                        "purpose": key.purpose,
                        "valid_from": key.valid_from.isoformat(),
                        "valid_until": key.valid_until.isoformat(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_research_training_to_attested_runtime_prediction(tmp_path: Path) -> None:
    now = datetime(2026, 7, 14, tzinfo=UTC)
    dataset_root = tmp_path / "dataset"
    write_market_dataset_files(dataset_root, _dataset())
    dataset = load_market_dataset_artifact(dataset_root)
    config_path = tmp_path / "training.json"
    _config(config_path)
    config = normalize_training_run_config(TrainingRunConfig.from_json(config_path))

    selection_private = Ed25519PrivateKey.from_private_bytes(b"\x51" * 32)
    selection_public = PublicVerificationKey(
        key_id="e2e-selection-key",
        public_key=public_key_bytes(selection_private),
        purpose="selection-authorization",
        valid_from=now - timedelta(days=1),
        valid_until=now + timedelta(days=365),
    )
    root = Path(__file__).resolve().parents[2]
    proposal = SelectionProposal.create(
        walk_forward_run_digest="1" * 64,
        gate_evidence_digest="2" * 64,
        execution_sensitivity_digest="3" * 64,
        dataset_id=dataset.dataset_id,
        selected_configuration="e2e-selected",
        candidate_config_digest=content_digest(config.candidate_digest_payload()),
        seeds=config.training.seeds,
        git_commit="a" * 40,
        dependency_digest=hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        resume_checkpoint_digests=(),
    )
    authorization = create_selection_authorization(
        proposal,
        approver="selection-committee",
        approved_at=now,
        expires_at=now + timedelta(days=30),
        key_id=selection_public.key_id,
        private_key=selection_private,
    )
    proposal_path = write_selection_proposal(tmp_path / "proposal.json", proposal)
    authorization_path = write_selection_authorization(
        tmp_path / "authorization.json", authorization
    )
    selection_keys_path = tmp_path / "selection-keys.json"
    _key_store(selection_keys_path, selection_public)

    result = execute_training_run(
        config_path=config_path,
        dataset_path=dataset_root,
        store_root=tmp_path / "store",
        run_id="e2e",
        created_at=now,
        selection_proposal_path=proposal_path,
        selection_authorization_path=authorization_path,
        selection_public_keys_path=selection_keys_path,
        require_selection_authorization=True,
    )
    run_root = result.path
    ensemble = json.loads((run_root / "ensemble.json").read_text(encoding="utf-8"))
    training_manifest = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    confirmation_start = datetime.fromisoformat(
        training_manifest["completed_at"].replace("Z", "+00:00")
    )
    confirmation_end = confirmation_start + timedelta(days=30)

    confirmation_private = Ed25519PrivateKey.from_private_bytes(b"\x52" * 32)
    confirmation_public = PublicVerificationKey(
        key_id="e2e-confirmation-key",
        public_key=public_key_bytes(confirmation_private),
        purpose="fresh-confirmation",
        valid_from=now - timedelta(days=1),
        valid_until=now + timedelta(days=365),
    )
    confirmation = create_fresh_confirmation_evidence(
        dataset_id=result.dataset_id,
        environment_digest=ensemble["environment_digest"],
        policy_digest=result.policy_digest,
        training_run_digest=result.run_digest,
        git_commit="a" * 40,
        dependency_digest=proposal.dependency_digest,
        required_after=confirmation_start,
        start_time=confirmation_start,
        end_time=confirmation_end,
        returns=(0.001,) * 30,
        return_period_hours=24.0,
        order_log_digest="4" * 64,
        fill_log_digest="5" * 64,
        reconciliation_digest="6" * 64,
        created_at=confirmation_end,
        key_id=confirmation_public.key_id,
        private_key=confirmation_private,
    )
    confirmation_path = write_confirmation_evidence(
        tmp_path / "confirmation.json", confirmation
    )
    bundle_root = tmp_path / "bundle"
    manifest = package_selected_training_run(
        training_root=run_root,
        confirmation_path=confirmation_path,
        output_root=bundle_root,
        signal_digest="7" * 64,
        selection_digest="8" * 64,
        trusted_confirmation_keys={confirmation_public.key_id: confirmation_public},
        trusted_now=confirmation_end,
    )

    release_private = Ed25519PrivateKey.from_private_bytes(b"\x53" * 32)
    release_public = PublicVerificationKey(
        key_id="e2e-release-key",
        public_key=public_key_bytes(release_private),
        purpose="release-verification",
        valid_from=now - timedelta(days=1),
        valid_until=now + timedelta(days=365),
    )
    attestation = create_release_attestation(
        bundle_digest=manifest.bundle_digest,
        dataset_id=manifest.dataset_id,
        training_run_digest=manifest.training_run_digest,
        run_kind=manifest.run_kind,
        selection_proposal_digest=manifest.selection_proposal_digest,
        selection_authorization_digest=manifest.selection_authorization_digest,
        walk_forward_run_digest=manifest.walk_forward_run_digest,
        gate_evidence_digest=manifest.gate_evidence_digest,
        confirmation_evidence_digest=manifest.confirmation_evidence_digest,
        selected_policy_digest=manifest.policy_digest,
        git_commit="a" * 40,
        dependency_digest=proposal.dependency_digest,
        approver="release-committee",
        approved_at=confirmation_end,
        expires_at=confirmation_end + timedelta(days=30),
        key_id=release_public.key_id,
        private_key=release_private,
    )
    write_release_attestation(default_attestation_path(bundle_root), attestation)

    runtime = ServingRuntime(
        StableBaselines3PolicyLoader(),
        trusted_attestation_keys={release_public.key_id: release_public},
        clock=lambda: confirmation_end,
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
