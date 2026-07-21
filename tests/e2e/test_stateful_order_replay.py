from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.execution_replay import (
    StatefulReplayEvidence,
    build_stateful_replay_evidence,
)
from trade_rl.rl.actions import ActionMode, ActionSpec
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import OrderEvent
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _dataset() -> MarketDataset:
    n = 48
    close = 100.0 * np.exp(np.arange(n, dtype=np.float64)[:, None] * 0.001)
    open_price = np.vstack((close[:1], close[:-1]))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.002,
        low=np.minimum(open_price, close) * 0.998,
        close=close,
        volume=np.full_like(close, 2.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _environment(dataset: MarketDataset) -> ResidualMarketEnv:
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=6)
        ),
        action_spec=ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            risk_tilt_enabled=False,
            target_weight_count=1,
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=1_000.0,
            episode_bars=10,
            decision_every=1,
            liquidate_on_end=False,
            execution_cost=replace(
                ExecutionCostConfig.zero(),
                max_participation_rate=0.25,
                order_type="limit",
                limit_offset_rate=0.001,
                path_mode="conservative",
                processing_bar_volume_capacity=True,
                partial_fill_carry=True,
                trigger_volume_fractions=(1.0, 0.5, 0.25, 0.0),
            ),
        ),
    )


def _run_episode(*, seed: int) -> StatefulReplayEvidence:
    dataset = _dataset()
    env = _environment(dataset)
    actions = (
        np.array((1.0,), dtype=np.float32),
        np.array((1.0,), dtype=np.float32),
        np.array((-0.5,), dtype=np.float32),
        np.array((0.25,), dtype=np.float32),
        np.array((0.0,), dtype=np.float32),
        np.array((0.75,), dtype=np.float32),
    )
    env.reset(seed=seed, options={"start_idx": 10, "initial_state_mode": "cash"})
    equity_curve = [env.hybrid.portfolio_value]
    observation_digests = [env.observation_snapshot().snapshot_digest]
    events: list[OrderEvent] = []
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        events.extend(info["hybrid_execution"].order_events)
        equity_curve.append(env.hybrid.portfolio_value)
        observation_digests.append(env.observation_snapshot().snapshot_digest)
        assert not terminated
        assert not truncated
    return build_stateful_replay_evidence(
        dataset_id=dataset.dataset_id,
        seed=seed,
        execution_policy_digest=env.execution_policy_digest,
        actions=actions,
        order_events=events,
        equity_curve=equity_curve,
        observation_digests=observation_digests,
    )


def test_same_dataset_seed_and_actions_replay_identically() -> None:
    first = _run_episode(seed=7)
    second = _run_episode(seed=7)

    assert first == second
    assert first.digest == second.digest
    assert first.order_event_count > 0
    assert first.step_count == 6
    assert first.execution_policy_digest != "0" * 64


def test_replay_evidence_changes_when_action_trace_changes() -> None:
    baseline = _run_episode(seed=7)
    changed = build_stateful_replay_evidence(
        dataset_id=baseline.dataset_id,
        seed=baseline.seed,
        execution_policy_digest=baseline.execution_policy_digest,
        actions=((0.0,),),
        order_events=(),
        equity_curve=(1_000.0, 1_000.0),
        observation_digests=("1" * 64, "2" * 64),
    )

    assert changed.action_digest != baseline.action_digest
    assert changed.digest != baseline.digest
