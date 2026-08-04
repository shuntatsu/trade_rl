from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.learning.episode_oracle_teacher import (
    OracleEpisodeSamplingConfig,
    build_episode_oracle_batch,
    episode_oracle_target_path,
    sample_oracle_episode_contracts,
)
from trade_rl.learning.oracle_teacher import OracleTeacherConfig
from trade_rl.simulation.execution import ExecutionCostConfig


def _market(close_values: np.ndarray) -> MarketDataset:
    close = np.asarray(close_values, dtype=np.float64)
    if close.ndim == 1:
        close = close[:, None]
    n_bars, n_symbols = close.shape
    open_price = np.vstack([close[0], close[:-1]])
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=tuple(f"S{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        feature_names=("return",),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _costly_oracle(*, signal_delay_decisions: int = 0) -> OracleTeacherConfig:
    return OracleTeacherConfig(
        execution_cost=ExecutionCostConfig(
            fee_rate=0.01,
            spread_rate=0.01,
            impact_rate=0.0,
            max_participation_rate=1.0,
        ),
        signal_delay_decisions=signal_delay_decisions,
    )


def test_episode_oracle_respects_non_cash_initial_weights() -> None:
    market = _market(np.full(8, 100.0))
    config = _costly_oracle()

    cash = episode_oracle_target_path(
        market,
        (0, 8),
        config,
        initial_weights=np.zeros(1, dtype=np.float64),
    )
    invested = episode_oracle_target_path(
        market,
        (0, 8),
        config,
        initial_weights=np.array([0.45], dtype=np.float64),
    )

    assert np.count_nonzero(cash) == 0
    assert np.all(invested[:, 0] > 0.0)


def test_episode_oracle_rejects_invalid_initial_weights() -> None:
    market = _market(np.full(8, 100.0))

    with pytest.raises(ValueError, match="initial weights"):
        episode_oracle_target_path(
            market,
            (0, 8),
            OracleTeacherConfig(),
            initial_weights=np.array([0.46], dtype=np.float64),
        )


def test_episode_contract_sampling_is_seeded_and_horizon_exact() -> None:
    market = _market(np.linspace(100.0, 110.0, 24))
    sampling = OracleEpisodeSamplingConfig(
        episode_bars=4,
        episode_count=12,
        initial_state_modes=("cash", "baseline"),
        seed=17,
    )

    def baseline(mode: str, start: int) -> np.ndarray:
        assert mode == "baseline"
        return np.array([0.10 + 0.01 * (start % 3)], dtype=np.float64)

    first = sample_oracle_episode_contracts(
        market,
        minimum_start_index=3,
        config=sampling,
        initial_weight_provider=baseline,
    )
    second = sample_oracle_episode_contracts(
        market,
        minimum_start_index=3,
        config=sampling,
        initial_weight_provider=baseline,
    )

    assert tuple(contract.digest for contract in first) == tuple(
        contract.digest for contract in second
    )
    assert {contract.initial_state_mode for contract in first} == {"cash", "baseline"}
    for contract in first:
        assert contract.stop - contract.start - 1 == sampling.episode_bars
        assert contract.start >= 3
        assert contract.stop <= market.n_bars
        assert contract.initial_weights.flags.writeable is False
        if contract.initial_state_mode == "cash":
            np.testing.assert_array_equal(contract.initial_weights, np.zeros(1))
        else:
            np.testing.assert_allclose(
                contract.initial_weights,
                np.array([0.10 + 0.01 * (contract.start % 3)]),
            )


def test_non_cash_sampling_requires_initial_weight_provider() -> None:
    market = _market(np.linspace(100.0, 110.0, 12))
    sampling = OracleEpisodeSamplingConfig(
        episode_bars=4,
        episode_count=2,
        initial_state_modes=("baseline",),
        seed=1,
    )

    with pytest.raises(ValueError, match="initial weight provider"):
        sample_oracle_episode_contracts(
            market,
            minimum_start_index=0,
            config=sampling,
        )


def test_episode_sampling_rejects_ranges_without_a_complete_horizon() -> None:
    market = _market(np.linspace(100.0, 105.0, 6))
    sampling = OracleEpisodeSamplingConfig(
        episode_bars=6,
        episode_count=1,
        initial_state_modes=("cash",),
        seed=0,
    )

    with pytest.raises(ValueError, match="complete episode"):
        sample_oracle_episode_contracts(
            market,
            minimum_start_index=0,
            config=sampling,
        )


def test_episode_oracle_batch_preserves_episode_boundaries_and_initial_state() -> None:
    market = _market(np.full(24, 100.0))
    sampling = OracleEpisodeSamplingConfig(
        episode_bars=5,
        episode_count=3,
        initial_state_modes=("baseline",),
        seed=9,
    )

    batch = build_episode_oracle_batch(
        market,
        minimum_start_index=2,
        sampling_config=sampling,
        teacher_config=_costly_oracle(),
        initial_weight_provider=lambda mode, start: np.array([0.45]),
    )

    assert batch.episode_count == sampling.episode_count
    assert batch.decision_count == sampling.episode_count * sampling.episode_bars
    assert len(batch.contracts) == len(batch.targets) == sampling.episode_count
    assert batch.digest
    assert batch.solver_provenance is not None
    assert batch.solver_provenance.backend == "numpy"
    for contract, targets in zip(batch.contracts, batch.targets, strict=True):
        assert contract.stop - contract.start - 1 == sampling.episode_bars
        assert targets.shape == (sampling.episode_bars, market.n_symbols)
        assert targets.flags.writeable is False
        assert np.all(targets[:, 0] > 0.0)


def test_parallel_episode_oracle_batch_matches_serial_digest() -> None:
    market = _market(np.linspace(100.0, 110.0, 48))
    sampling = OracleEpisodeSamplingConfig(
        episode_bars=6,
        episode_count=5,
        initial_state_modes=("cash", "baseline"),
        seed=19,
    )
    teacher = _costly_oracle()

    def provider(mode: str, start: int) -> np.ndarray:
        return np.array([0.35])

    serial = build_episode_oracle_batch(
        market,
        minimum_start_index=2,
        sampling_config=sampling,
        teacher_config=teacher,
        initial_weight_provider=provider,
    )
    parallel = build_episode_oracle_batch(
        market,
        minimum_start_index=2,
        sampling_config=sampling,
        teacher_config=teacher,
        initial_weight_provider=provider,
        max_workers=3,
    )

    assert parallel.digest == serial.digest
    assert tuple(contract.digest for contract in parallel.contracts) == tuple(
        contract.digest for contract in serial.contracts
    )
    for actual, expected in zip(parallel.targets, serial.targets, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_delayed_episode_oracle_holds_initial_position_before_pending_target_executes() -> (
    None
):
    market = _market(np.full(10, 100.0))
    targets = episode_oracle_target_path(
        market,
        (0, 10),
        _costly_oracle(signal_delay_decisions=1),
        initial_weights=np.array([0.45], dtype=np.float64),
    )

    assert np.all(targets[:-1, 0] > 0.0)
    assert targets[-1, 0] == pytest.approx(0.0)


def test_episode_sampling_rejects_duplicate_initial_state_modes() -> None:
    with pytest.raises(ValueError, match="initial_state_modes"):
        OracleEpisodeSamplingConfig(
            episode_bars=4,
            episode_count=1,
            initial_state_modes=("cash", "cash"),
            seed=0,
        )
