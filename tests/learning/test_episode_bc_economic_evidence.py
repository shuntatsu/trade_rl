from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_behavior_cloning_holdout,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _market(n_bars: int = 64) -> MarketDataset:
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(phase * 0.001))[:, None]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:15:00", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=np.stack(
            tuple(np.sin(phase / divisor) for divisor in (3.0, 5.0, 7.0, 11.0)),
            axis=1,
        )[:, None, :].astype(np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack((close[:1], close[:-1])),
        high=np.maximum(np.vstack((close[:1], close[:-1])), close) + 0.1,
        low=np.minimum(np.vstack((close[:1], close[:-1])), close) - 0.1,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 4), dtype=np.bool_),
        feature_names=(
            "15m__return",
            "1h__return",
            "4h__return",
            "1d__return",
        ),
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _environment() -> ResidualMarketEnv:
    return ResidualMarketEnv(
        _market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=1, base_lookback=2, slow_lookback=3)
        ),
        action_spec=ActionSpec(
            mode="target_weight",
            alpha_enabled=False,
            risk_tilt_enabled=False,
            target_weight_count=1,
        ),
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(
                entry_threshold=0.0,
                exit_threshold=0.0,
                no_trade_band=0.0,
            )
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=4,
            decision_every=1,
            initial_state_modes=("cash",),
            execution_cost=ExecutionCostConfig.zero(),
        ),
    )


def _batch(environment: ResidualMarketEnv) -> EpisodeOracleBatch:
    start = environment.minimum_start_index
    starts = (start, start + 8)
    contracts = tuple(
        OracleEpisodeContract(
            dataset_id=environment.dataset.dataset_id,
            episode_index=episode_index,
            start=episode_start,
            stop=episode_start + 5,
            initial_state_mode="cash",
            initial_weights=np.zeros(1, dtype=np.float64),
        )
        for episode_index, episode_start in enumerate(starts)
    )
    targets = tuple(
        np.zeros((contract.stop - contract.start - 1, 1), dtype=np.float32)
        for contract in contracts
    )
    return EpisodeOracleBatch(
        dataset_id=environment.dataset.dataset_id,
        teacher_config_digest="b" * 64,
        sampling_config_digest="c" * 64,
        contracts=contracts,
        targets=targets,
    )


class _ZeroPolicy:
    def predict(
        self,
        observation: object,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        del observation, deterministic
        return np.zeros(1, dtype=np.float32), None


def test_episode_bc_holdout_persists_causal_net_return_lower_bound(
    tmp_path: Path,
) -> None:
    environment = _environment()
    batch = _batch(environment)
    environment.close()
    split = BehaviorCloningSplit(
        train_indices=np.asarray([0], dtype=np.int64),
        validation_indices=np.asarray([1, 2], dtype=np.int64),
        train_episode_ids=np.asarray([99], dtype=np.int64),
        validation_episode_ids=np.asarray([0, 1], dtype=np.int64),
    )

    audit, holdout = evaluate_episode_behavior_cloning_holdout(
        environment_factory=_environment,
        model=_ZeroPolicy(),
        batch=batch,
        split=split,
        output_root=tmp_path,
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=2_000,
    )

    assert holdout is not None
    observed = np.asarray(
        [
            record.causal_policy_performance.net_return
            for record in holdout.records
        ],
        dtype=np.float64,
    )
    assert (
        holdout.causal_net_return_lower_confidence_bound
        <= float(np.mean(observed))
    )
    assert audit["causal_net_return_lower_confidence_bound"] == (
        holdout.causal_net_return_lower_confidence_bound
    )
    payload = json.loads(
        (tmp_path / "behavior-cloning-holdout.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["causal_net_return_lower_confidence_bound"] == (
        holdout.causal_net_return_lower_confidence_bound
    )
    assert payload["schema_version"] == "episode_oracle_bc_evaluation_v2"
