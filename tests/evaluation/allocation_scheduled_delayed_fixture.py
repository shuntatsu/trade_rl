"""Result-blind fixed synthetic H16 early-capacity delayed-payoff declarations."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np

from tests.evaluation.test_allocation_preprocessing_runtime import bind_frozen
from tests.evaluation.test_allocation_rl_env import parameters
from tests.evaluation.test_allocation_rl_observation_v2 import opt_in
from tests.strategies.test_allocation_protocol_receipt import protocol
from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.evaluation.rl_allocation.preprocessing import (
    fit_allocation_feature_preprocessing,
    validate_training_preprocessing,
)
from trade_rl.evaluation.rl_allocation.training_schedule import (
    AllocationTrainingScheduleEnv,
    allocation_training_window,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.forecasts.simple_prequential import (
    fit_prequential_simple_ridge,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
)

TRAIN_ROSTER = ((1, 1), (2, -1), (3, 1), (4, -1), (5, 1), (6, -1))
HELDOUT_ROSTER = ((20, 1), (21, -1), (22, 1), (23, -1))
SEEDS = (0, 7, 17)
ROLLOUT_FACTORS = (4, 32)
BUDGET = 4096


def delayed_protocol(n_steps):
    return protocol(n_steps=n_steps, batch_size=4, n_epochs=10)


def _validate_payoff_index(payoff_index, *, payoff=True):
    if type(payoff_index) is not int or payoff_index not in (8, 22):
        raise ValueError("payoff_index must be the native integer 8 or 22")
    if payoff_index == 8 and not payoff:
        raise ValueError("early payoff_index requires payoff=True")


def delayed_raw_args(
    day, cue, *, n_steps=4, latency=0, payoff=True, processing=True, payoff_index=22
):
    _validate_payoff_index(payoff_index, payoff=payoff)
    args = parameters()
    rows = 24
    times = np.datetime64(f"2026-01-{day:02d}", "ns") + np.arange(
        rows
    ) * np.timedelta64(1, "h")
    shape = (rows, 1)
    marks = np.full(shape, 128.0)
    if payoff:
        marks[payoff_index:] = 128 * (1 + 0.08 * cue)
    opens = np.full(shape, 128.0)
    features = np.full((rows, 1, 1), cue, dtype=np.float32)
    features[:3, 0, 0] = [-1, 0, 1]
    volume = np.zeros(shape)
    volume[7, 0] = 1000
    data = MarketDataset(
        dataset_id=content_digest(
            {"scheduled_delayed_v1": day, "cue": cue, "payoff": payoff}
            if payoff_index == 22
            else {
                "scheduled_payoff_timing_v1": day,
                "cue": cue,
                "payoff": payoff,
                "payoff_index": payoff_index,
            }
        ),
        symbols=("S0",),
        timestamps=times,
        features=features,
        global_features=np.zeros((rows, 1), np.float32),
        open=opens,
        high=np.maximum(opens, marks),
        low=np.minimum(opens, marks),
        close=marks,
        volume=volume,
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, bool),
        feature_available=np.ones(features.shape, bool),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8760,
        volume_units=(VolumeUnit.BASE_ASSET,),
        borrow_available=np.ones(shape, bool),
        borrow_rate=np.zeros(shape),
        mark_price=marks,
        index_price=marks,
        cash_rate=np.zeros(rows),
        available_at=np.repeat(times[:, None], 1, axis=1),
        information_available=np.ones(shape, bool),
        contract_multipliers=np.ones(1),
    )
    stream = fit_prequential_simple_ridge(
        data,
        blocks=(
            ForecastBlock(
                times[5], times[5] + np.timedelta64(15, "m"), times[6], times[23]
            ),
        ),
        feature_indices=(0,),
        horizon_hours=1,
        alpha=1e6,
    )
    args["dataset"], args["stream"] = data, stream
    args["estimates"] = tuple(
        replace(
            args["estimates"][0],
            decision_time=times[index],
            available_at=times[index],
            horizon_end=times[index + 1],
            buy_cost=1 / 512,
            sell_cost=1 / 512,
        )
        for index in range(6, 22)
    )
    args["allocator"] = replace(args["allocator"], lower_weight=-0.5, upper_weight=0.5)
    args["execution_cost"] = replace(
        ExecutionCostConfig.zero(),
        fee_rate=1 / 512,
        max_participation_rate=1.0,
        allow_short=True,
        order_latency_bars=latency,
        processing_bar_volume_capacity=processing,
    )
    args["bound"] = replace(
        args["bound"],
        objective=replace(
            args["bound"].objective,
            capital=replace(
                args["bound"].objective.capital, initial_equities=(1024.0,)
            ),
            evaluation_start=datetime(2026, 1, day, 6, tzinfo=UTC),
            evaluation_stop_exclusive=datetime(2026, 1, day, 22, tzinfo=UTC),
        ),
        clock=replace(
            args["bound"].clock,
            economic_horizon_seconds=16 * 3600,
            rollout_steps=n_steps,
        ),
    )
    args["stop_index"] = 22
    return opt_in(args)


def delayed_preprocessing(*, payoff_index=22):
    _validate_payoff_index(payoff_index)
    data = delayed_raw_args(1, 1, payoff_index=payoff_index)["dataset"]
    return fit_allocation_feature_preprocessing(
        data,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        fit_start=0,
        fit_stop=3,
        fit_as_of=data.timestamps[3],
        first_decision_index=6,
    )


def delayed_env(day, cue, *, n_steps=4, frozen=None, payoff_index=22, **controls):
    _validate_payoff_index(payoff_index, payoff=controls.get("payoff", True))
    if frozen is None:
        frozen = delayed_preprocessing(payoff_index=payoff_index)
    elif payoff_index == 8:
        # All children consume this arm's canonical day1/cue+1 declaration.
        # Reconstruct to check supplied content, never replace its coefficients.
        validate_training_preprocessing(
            delayed_raw_args(1, 1, payoff_index=payoff_index)["dataset"],
            frozen,
            feature_indices=(0,),
            symbol_index=0,
            first_decision_index=6,
        )
    return AllocationTradingEnv(
        **bind_frozen(
            delayed_raw_args(
                day, cue, n_steps=n_steps, payoff_index=payoff_index, **controls
            ),
            frozen,
        )
    )


def delayed_schedule(n_steps, *, frozen=None, payoff_index=22):
    _validate_payoff_index(payoff_index)
    frozen = (
        delayed_preprocessing(payoff_index=payoff_index) if frozen is None else frozen
    )
    children = tuple(
        delayed_env(day, cue, n_steps=n_steps, frozen=frozen, payoff_index=payoff_index)
        for day, cue in TRAIN_ROSTER
    )
    heldout = tuple(
        delayed_env(day, cue, n_steps=n_steps, frozen=frozen, payoff_index=payoff_index)
        for day, cue in HELDOUT_ROSTER
    )
    windows = tuple(allocation_training_window(child) for child in children)
    excluded = tuple(
        allocation_training_window(child, role="held_out") for child in heldout
    )
    return AllocationTrainingScheduleEnv(
        AllocationTrainingSchedule(
            windows + excluded, tuple(w.window_id for w in windows)
        ),
        children,
    )
