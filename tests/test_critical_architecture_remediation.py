from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.serving.helpers import create_bundle
from trade_rl.data.market import MarketDataset
from trade_rl.release.attestation import (
    ReleaseAttestation,
    default_attestation_path,
    write_release_attestation,
)
from trade_rl.release.signing import VerificationKey
from trade_rl.risk.portfolio import PortfolioRiskConfig, PortfolioRiskModel
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.serving.bundle import load_serving_bundle
from trade_rl.serving.registry import ServingRegistry
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _dataset(*, multiplier: float = 0.1) -> MarketDataset:
    n_bars = 24
    close = np.linspace(100.0, 123.0, n_bars).reshape(-1, 1)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("PERP",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 4), dtype=np.float32),
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((n_bars, 1), 1_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("feature",),
        global_feature_names=("a", "b", "c", "d"),
        periods_per_year=8_760,
        contract_multipliers=np.array([multiplier]),
    )


def _environment(
    dataset: MarketDataset,
    *,
    portfolio_risk: PortfolioRiskModel | None = None,
) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        portfolio_risk=portfolio_risk,
        config=ResidualMarketEnvConfig(
            initial_capital=1_000.0,
            episode_bars=8,
            decision_every=1,
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def test_environment_uses_dataset_contract_multipliers() -> None:
    dataset = _dataset()
    environment = _environment(dataset)
    environment.reset(seed=0, options={"start_idx": 8})

    _, _, terminated, truncated, _ = environment.step(
        np.zeros(environment.action_spec.size, dtype=np.float32)
    )

    assert not terminated
    assert not truncated
    np.testing.assert_array_equal(
        environment.hybrid.contract_multipliers,
        dataset.contract_multipliers,
    )


def test_restore_rejects_contract_multiplier_mismatch() -> None:
    dataset = _dataset()
    environment = _environment(dataset)
    incompatible = BookState.zero(1, 1_000.0, dataset.close[8])

    with pytest.raises(ValueError, match="contract multipliers"):
        environment.reset(
            seed=0,
            options={
                "start_idx": 8,
                "initial_state_mode": "restore",
                "initial_book": incompatible,
            },
        )


def test_portfolio_risk_changes_final_target_and_environment_identity() -> None:
    dataset = _dataset(multiplier=1.0)
    unrestricted = _environment(dataset)
    restricted = _environment(
        dataset,
        portfolio_risk=PortfolioRiskModel(PortfolioRiskConfig(max_abs_weight=0.05)),
    )
    assert unrestricted.environment_digest != restricted.environment_digest
    restricted.reset(seed=0, options={"start_idx": 8})

    _, _, _, _, info = restricted.step(
        np.zeros(restricted.action_spec.size, dtype=np.float32)
    )

    risk = info["hybrid_risk"]
    assert np.max(np.abs(risk.weights)) <= 0.05 + 1e-12
    assert "portfolio:max_abs_weight" in risk.reasons


def _write_external_attestation(source: Path) -> ReleaseAttestation:
    manifest = load_serving_bundle(source).manifest
    attestation = ReleaseAttestation.create(
        bundle_digest=manifest.bundle_digest,
        dataset_id=manifest.dataset_id,
        selection_evaluation_digest=manifest.selection_digest,
        gate_evaluation_digest="2" * 64,
        gate_evidence_digest="3" * 64,
        selected_policy_digest=manifest.policy_digest,
        git_commit="e" * 40,
        dependency_digest="4" * 64,
        approver="architecture-audit",
        approved_at=datetime(2026, 7, 14, tzinfo=UTC),
        key_id="test-release-key",
        signing_key=b"test-release-signing-key",
    )
    write_release_attestation(default_attestation_path(source), attestation)
    return attestation


def test_registry_installs_external_release_attestation(tmp_path: Path) -> None:
    source = create_bundle(tmp_path / "source", release_digest=None)
    expected = _write_external_attestation(source)
    registry = ServingRegistry(
        tmp_path / "registry",
        trusted_attestation_keys={
            "test-release-key": VerificationKey(
                key_id="test-release-key",
                key=b"test-release-signing-key",
                purpose="release-verification",
            )
        },
    )

    active = registry.activate(source)
    reloaded = registry.active_bundle()

    assert active.release is not None
    assert active.release.digest == expected.digest
    assert reloaded.release is not None
    assert reloaded.release.digest == expected.digest
    assert default_attestation_path(reloaded.root).is_file()
