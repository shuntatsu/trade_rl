from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

ROOT = Path(__file__).resolve().parents[2]


def _market() -> MarketDataset:
    n_bars = 96
    close = np.column_stack(
        (
            np.linspace(100.0, 140.0, n_bars),
            np.linspace(80.0, 64.0, n_bars),
        )
    )
    open_price = np.vstack((close[0], close[:-1]))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 2, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((n_bars, 2), 10_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=np.ones((n_bars, 2, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        contract_multipliers=np.array([1.0, 2.0]),
    )


def _config(**overrides: object) -> ResidualMarketEnvConfig:
    values: dict[str, object] = {
        "initial_capital": 100_000.0,
        "episode_bars": 8,
        "decision_every": 2,
        "execution_cost": ExecutionCostConfig.zero(),
        "initial_state_modes": ("cash",),
    }
    values.update(overrides)
    return ResidualMarketEnvConfig(**values)  # type: ignore[arg-type]


def _env(**kwargs: object) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        _market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        config=kwargs.pop("config", _config()),  # type: ignore[arg-type]
        **kwargs,
    )


class _LegacyAlpha:
    artifact_digest = "1" * 64
    minimum_index = 8

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        del index
        return np.array([0.25, -0.25], dtype=np.float64)[: dataset.n_symbols]


class _FactorProvider:
    artifact_digest = "2" * 64
    n_factors = 1
    minimum_index = 8

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        del index
        return np.ones((1, dataset.n_symbols), dtype=np.float64)


def test_environment_provider_branches_are_explicitly_exercised() -> None:
    legacy = _env(
        alpha_provider=_LegacyAlpha(),
        alpha_enabled=True,
        action_spec=ActionSpec(alpha_enabled=True),
    )
    np.testing.assert_allclose(legacy._alpha_at(legacy.minimum_start_index), [0.25, -0.25])

    callable_alpha = _env(
        alpha_provider=lambda dataset, index: np.array([index, -index], dtype=np.float64),
        alpha_enabled=True,
        alpha_artifact_digest="3" * 64,
        action_spec=ActionSpec(alpha_enabled=True),
    )
    alpha = callable_alpha._alpha_at(callable_alpha.minimum_start_index)
    assert alpha.shape == (2,)
    assert np.isfinite(alpha).all()

    static = _env(factor_basis=np.array([[1.0, -1.0]], dtype=np.float64))
    first = static._factor_basis_at(static.minimum_start_index)
    first[0, 0] = 99.0
    assert static._factor_basis_at(static.minimum_start_index)[0, 0] != 99.0

    object_provider = _env(
        factor_basis_provider=_FactorProvider(),
        action_spec=ActionSpec(n_factors=1),
    )
    np.testing.assert_allclose(
        object_provider._factor_basis_at(object_provider.minimum_start_index),
        [[1.0, 1.0]],
    )

    callable_provider = _env(
        factor_basis_provider=lambda dataset, index: np.array([[index, -index]]),
        factor_artifact_digest="4" * 64,
        factor_count=1,
        action_spec=ActionSpec(n_factors=1),
    )
    assert callable_provider._factor_basis_at(callable_provider.minimum_start_index).shape == (
        1,
        2,
    )


def test_environment_factor_provider_rejections_remain_fail_closed() -> None:
    env = _env(
        factor_basis_provider=_FactorProvider(),
        action_spec=ActionSpec(n_factors=1),
    )
    env.factor_basis_provider = None
    with pytest.raises(RuntimeError, match="without a provider"):
        env._factor_basis_at(env.minimum_start_index)

    env.factor_basis_provider = lambda dataset, index: np.ones((2, dataset.n_symbols))
    with pytest.raises(ValueError, match="invalid basis"):
        env._factor_basis_at(env.minimum_start_index)

    env.factor_basis_provider = lambda dataset, index: np.array([[np.nan, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        env._factor_basis_at(env.minimum_start_index)


def test_environment_initial_state_modes_exercise_mutable_facade_branches() -> None:
    env = _env()
    start = env.minimum_start_index

    for mode in ("cash", "baseline", "random", "stress", "partial_fill"):
        observation, info = env.reset(
            seed=7,
            options={"start_idx": start, "initial_state_mode": mode},
        )
        assert observation is not None
        assert info["initial_state_mode"] == mode
        assert env.current_index == start

    assert np.any(env._execution_state.fill_ratio > 0.0)


def test_environment_restore_mode_validates_identity_and_value() -> None:
    env = _env()
    start = env.minimum_start_index
    valid = BookState.from_weights(
        weights=np.zeros(2),
        capital=env.initial_capital,
        prices=env.dataset.close[start],
        max_gross=env.pre_trade_risk.config.max_gross,
        contract_multipliers=env.dataset.resolved_array("contract_multipliers"),
    )
    _, info = env.reset(
        seed=11,
        options={
            "start_idx": start,
            "initial_state_mode": "restore",
            "initial_book": valid,
        },
    )
    assert info["initial_state_mode"] == "restore"

    with pytest.raises(ValueError, match="requires a BookState"):
        env.reset(
            options={
                "start_idx": start,
                "initial_state_mode": "restore",
                "initial_book": object(),
            }
        )

    wrong_symbols = BookState.from_weights(
        weights=np.zeros(1),
        capital=env.initial_capital,
        prices=np.array([100.0]),
        max_gross=1.0,
    )
    with pytest.raises(ValueError, match="dataset symbols"):
        env.reset(
            options={
                "start_idx": start,
                "initial_state_mode": "restore",
                "initial_book": wrong_symbols,
            }
        )

    wrong_multiplier = BookState.from_weights(
        weights=np.zeros(2),
        capital=env.initial_capital,
        prices=env.dataset.close[start],
        max_gross=1.0,
        contract_multipliers=np.ones(2),
    )
    with pytest.raises(ValueError, match="contract multipliers"):
        env.reset(
            options={
                "start_idx": start,
                "initial_state_mode": "restore",
                "initial_book": wrong_multiplier,
            }
        )

    wrong_value = BookState.from_weights(
        weights=np.zeros(2),
        capital=90_000.0,
        prices=env.dataset.close[start],
        max_gross=1.0,
        contract_multipliers=env.dataset.resolved_array("contract_multipliers"),
    )
    with pytest.raises(ValueError, match="value must match"):
        env.reset(
            options={
                "start_idx": start,
                "initial_state_mode": "restore",
                "initial_book": wrong_value,
            }
        )


def test_environment_pre_roll_and_terminal_guards_are_covered() -> None:
    env = _env()
    with pytest.raises(ValueError, match="non-empty"):
        env._bars_between(10, 10)
    assert env._bars_between(10, 13) == 2

    hourly = _env(config=_config(decision_every=None, decision_hours=3.0))
    assert hourly._bars_between(10, 20) == 3

    env.reset(seed=3, options={"start_idx": env.minimum_start_index})
    env.current_index = env.end_index
    with pytest.raises(RuntimeError, match="after the episode ended"):
        env.step(np.zeros(env.action_spec.size, dtype=np.float32))


def test_environment_file_has_a_stronger_branch_coverage_ratchet() -> None:
    configuration = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"trade_rl/rl/environment.py" = 75.0' in configuration
