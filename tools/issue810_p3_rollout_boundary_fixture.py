"""Closed synthetic fixture for Issue 810's PPO rollout-boundary diagnostic.

This module constructs all observations and execution data in memory. It never
loads market data or accepts a dataset path.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioGymObservationWrapper,
    PortfolioObservationRuntimeBinding,
    PortfolioObservationTransform,
)

PROTOCOL_SCHEMA = "issue810_p3_rollout_boundary_synthetic_v1"
BLOCK_TRANSITIONS = 128
BLOCKS_PER_WINDOW = 8
WINDOW_ACTOR_TRANSITIONS = BLOCK_TRANSITIONS * BLOCKS_PER_WINDOW
TERMINAL_RESERVE_TRANSITIONS = 1
WINDOW_STRIDE = WINDOW_ACTOR_TRANSITIONS + TERMINAL_RESERVE_TRANSITIONS
WINDOW_COUNT = 4
WINDOW_STARTS = tuple(index * WINDOW_STRIDE for index in range(WINDOW_COUNT))
WINDOW_STOPS = tuple(start + WINDOW_STRIDE for start in WINDOW_STARTS)
DATASET_ROW_COUNT = WINDOW_STOPS[-1] + 1
TRAIN_CUES = (1, -1, 0, 1, -1, 0, 1, -1)
OOS_CUES = (
    (-1, 0, 1, -1, 1, 0, -1, 1),
    (1, 0, -1, 1, -1, 0, 1, -1),
    (-1, 1, 0, -1, 1, 0, -1, 1),
)
ALL_WINDOW_CUES = (TRAIN_CUES, *OOS_CUES)
EVENT_RETURN = 0.015
INITIAL_CAPITAL = 1_000.0
TARGET_ABS_WEIGHT = 0.5
MAX_GROSS = 1.0
MAX_DRAWDOWN = 0.20
FEE_RATE = 0.0005
SPREAD_RATE = 0.0002
IMPACT_RATE = 0.0001
MAX_PARTICIPATION_RATE = 0.05
DECISION_INTERVAL_SECONDS = 3_600
ROLLOUT_STEPS = 64
GAE_LAMBDA = 0.95
_START = datetime(2026, 1, 1, tzinfo=UTC)
_EXECUTION_ECONOMICS_DIGEST = "a" * 64
_RISK_ECONOMICS_DIGEST = "b" * 64
_DEPLOYMENT_RECIPE_DIGEST = "c" * 64


def _validate_payoff_transition(payoff_transition: int) -> int:
    if type(payoff_transition) is not int or payoff_transition not in {48, 72}:
        raise ValueError("payoff_transition must be exactly 48 or 72")
    return payoff_transition


def _dataset_digest(
    payoff_transition: int,
    arrays: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(PROTOCOL_SCHEMA.encode("ascii"))
    digest.update(f"\0payoff_transition={payoff_transition}\0".encode("ascii"))
    digest.update(repr(ALL_WINDOW_CUES).encode("ascii"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_synthetic_dataset(payoff_transition: int) -> MarketDataset:
    """Build the exact train-plus-three-heldout synthetic Dataset in memory."""

    transition = _validate_payoff_transition(payoff_transition)
    n_rows = DATASET_ROW_COUNT
    close = np.full((n_rows, 1), 100.0, dtype=np.float64)
    cue = np.zeros((n_rows, 1, 1), dtype=np.float32)
    volume = np.zeros((n_rows, 1), dtype=np.float64)
    for window_index, (window_start, window_stop, cues) in enumerate(
        zip(WINDOW_STARTS, WINDOW_STOPS, ALL_WINDOW_CUES, strict=True)
    ):
        for block_index, block_cue in enumerate(cues):
            block_start = window_start + block_index * BLOCK_TRANSITIONS
            cue_stop = min(block_start + BLOCK_TRANSITIONS, n_rows)
            cue[block_start:cue_stop, 0, 0] = float(block_cue)
            opportunity_row = block_start + 1
            volume[opportunity_row, 0] = 1_000_000.0
        terminal_row = window_stop
        volume[terminal_row, 0] = 1_000_000.0

    # Apply each price transition once, in native row order. Every block's
    # payoff is relative to the immediately preceding close.
    close[0, 0] = 100.0
    event_rows = {
        window_start + block_index * BLOCK_TRANSITIONS + transition: cue_value
        for window_start, cue_schedule in zip(
            WINDOW_STARTS, ALL_WINDOW_CUES, strict=True
        )
        for block_index, cue_value in enumerate(cue_schedule)
        if cue_value
    }
    for row in range(1, n_rows):
        cue_value = event_rows.get(row)
        if cue_value is None:
            close[row, 0] = close[row - 1, 0]
        else:
            close[row, 0] = close[row - 1, 0] * (1.0 + EVENT_RETURN * cue_value)

    open_price = np.vstack((close[0], close[:-1]))
    high = np.maximum(open_price, close)
    low = np.minimum(open_price, close)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        n_rows
    ) * np.timedelta64(1, "h")
    global_features = np.zeros((n_rows, 1), dtype=np.float32)
    zeros = np.zeros((n_rows, 1), dtype=np.float64)
    ones_available = np.ones((n_rows, 1, 1), dtype=np.bool_)
    global_available = np.ones((n_rows, 1), dtype=np.bool_)
    tradable = np.ones((n_rows, 1), dtype=np.bool_)
    arrays = (
        timestamps,
        cue,
        global_features,
        open_price,
        high,
        low,
        close,
        volume,
        zeros,
        tradable,
        ones_available,
        global_available,
    )
    dataset_id = _dataset_digest(transition, arrays)
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=("SYNTH",),
        timestamps=timestamps,
        features=cue,
        global_features=global_features,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=zeros.copy(),
        tradable=tradable,
        feature_available=ones_available,
        feature_names=("persistent_signed_cue",),
        global_feature_names=("synthetic_constant_regime",),
        periods_per_year=8_760,
        global_feature_available=global_available,
        fee_rate=zeros.copy(),
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        borrow_rate=zeros.copy(),
        contract_multipliers=np.ones(1, dtype=np.float64),
    )


def _make_bound(
    *,
    start_index: int,
    stop_index: int,
) -> BoundObjectiveClock:
    objective = ObjectiveContract(
        capital=CapitalContract("shared_portfolio", "USD", (INITIAL_CAPITAL,)),
        evaluation_start=_START + timedelta(hours=start_index),
        evaluation_stop_exclusive=_START + timedelta(hours=stop_index),
        terminal_valuation="settled",
        economics_digest=_EXECUTION_ECONOMICS_DIGEST,
        risk_digest=_RISK_ECONOMICS_DIGEST,
        deployment_recipe_digest=_DEPLOYMENT_RECIPE_DIGEST,
        maximum_drawdown=MAX_DRAWDOWN,
    )
    clock = FinancialClockContract(
        decision_interval_seconds=DECISION_INTERVAL_SECONDS,
        execution_interval_seconds=DECISION_INTERVAL_SECONDS,
        reward_interval_seconds=DECISION_INTERVAL_SECONDS,
        economic_horizon_seconds=(stop_index - start_index) * DECISION_INTERVAL_SECONDS,
        rollout_steps=ROLLOUT_STEPS,
        gamma=1.0,
        gae_lambda=GAE_LAMBDA,
        reward_schema="equity_delta_v1",
    )
    return BoundObjectiveClock(objective, clock)


def _make_risk_config() -> PreTradeRiskConfig:
    return PreTradeRiskConfig(
        max_gross=MAX_GROSS,
        max_abs_weight=TARGET_ABS_WEIGHT,
        max_turnover=None,
        drawdown_start=MAX_DRAWDOWN,
        drawdown_stop=MAX_DRAWDOWN,
    )


def _make_execution_cost(execution_seed: int) -> ExecutionCostConfig:
    if type(execution_seed) is not int or execution_seed < 0:
        raise ValueError("execution_seed must be a non-negative integer")
    return ExecutionCostConfig(
        fee_rate=FEE_RATE,
        spread_rate=SPREAD_RATE,
        impact_rate=IMPACT_RATE,
        max_participation_rate=MAX_PARTICIPATION_RATE,
        random_seed=execution_seed,
    )


def make_training_env(
    dataset: MarketDataset,
    *,
    execution_seed: int = 1_000,
) -> SharedPortfolioGymEnv:
    """Create the only fit environment, bound to the first 1,024 actor rows."""

    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    start_index, stop_index = WINDOW_STARTS[0], WINDOW_STOPS[0]
    return SharedPortfolioGymEnv(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        bound=_make_bound(start_index=start_index, stop_index=stop_index),
        target_abs_weights=(TARGET_ABS_WEIGHT,),
        execution_cost=_make_execution_cost(execution_seed),
        risk_config=_make_risk_config(),
    )


def make_oos_env(
    dataset: MarketDataset,
    window_index: int,
    transform: PortfolioObservationTransform,
    training_env: SharedPortfolioGymEnv,
) -> PortfolioGymObservationWrapper:
    """Bind one post-fit window to training-only statistics and semantics."""

    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    if type(window_index) is not int or not 1 <= window_index < WINDOW_COUNT:
        raise ValueError("window_index must identify one of the three OOS windows")
    if not isinstance(transform, PortfolioObservationTransform):
        raise TypeError("transform must be a PortfolioObservationTransform")
    if not isinstance(training_env, SharedPortfolioGymEnv):
        raise TypeError("training_env must be a SharedPortfolioGymEnv")

    start_index = WINDOW_STARTS[window_index]
    stop_index = WINDOW_STOPS[window_index]
    runtime_env = SharedPortfolioGymEnv(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        bound=_make_bound(start_index=start_index, stop_index=stop_index),
        target_abs_weights=training_env.target_abs_weights,
        execution_cost=training_env.execution_cost,
        risk_config=training_env.risk_config,
        market_order_profile=training_env.market_order_profile,
    )
    runtime_binding = PortfolioObservationRuntimeBinding.create(
        transform,
        training_env.bound,
        runtime_env,
    )
    return PortfolioGymObservationWrapper(
        runtime_env,
        transform,
        runtime_binding=runtime_binding,
        training_bound=training_env.bound,
    )


__all__ = [
    "BLOCKS_PER_WINDOW",
    "BLOCK_TRANSITIONS",
    "DATASET_ROW_COUNT",
    "EVENT_RETURN",
    "FEE_RATE",
    "GAE_LAMBDA",
    "IMPACT_RATE",
    "INITIAL_CAPITAL",
    "MAX_DRAWDOWN",
    "MAX_GROSS",
    "MAX_PARTICIPATION_RATE",
    "PROTOCOL_SCHEMA",
    "ROLLOUT_STEPS",
    "SPREAD_RATE",
    "TARGET_ABS_WEIGHT",
    "WINDOW_ACTOR_TRANSITIONS",
    "WINDOW_COUNT",
    "WINDOW_STARTS",
    "WINDOW_STOPS",
    "build_synthetic_dataset",
    "make_oos_env",
    "make_training_env",
]
