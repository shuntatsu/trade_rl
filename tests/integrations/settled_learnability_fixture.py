"""Frozen manufactured cues/capacity; never discovers or loads market data."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import product

import numpy as np

from trade_rl.artifacts import content_digest
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import CapitalContract, ObjectiveContract
from trade_rl.evaluation.rl_allocation.settled_binding import (
    BoundSettledAllocationClock,
)
from trade_rl.evaluation.rl_allocation.settled_env import SettledAllocationTradingEnv
from trade_rl.risk import PreTradeRiskConfig
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import AllocationActionContract
from trade_rl.strategies.forecasts.simple_prequential import (
    fit_prequential_simple_ridge,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import AllocationRuntimeProfile
from trade_rl.strategies.rl.allocation_recipe_v4 import allocation_recipe_payload_v4
from trade_rl.strategies.rl.allocation_settled_clock import (
    SettledAllocationClockContract,
)

PATTERNS = tuple(product((-1, 1), repeat=4))
TRAIN_SIGNS = (1, -1, -1, 1)


def error_summary(error):
    try:
        message = str(error)
    except BaseException:
        message = "<exception message unavailable>"
    return {"type": type(error).__name__, "message": message}


def note_failure(primary, context, secondary):
    try:
        BaseException.add_note(primary, f"{context}: {error_summary(secondary)}")
    except BaseException:
        pass


def environment(
    signs=TRAIN_SIGNS,
    *,
    price_response=0.02,
    fee=0.0005,
    cue_transform=1,
    capacity=100000,
    future_perturbation=None,
):
    """One P16/R2/H18 episode; zero capacity forbids a just-in-time entry."""
    if len(signs) != 4 or any(sign not in (-1, 0, 1) for sign in signs):
        raise ValueError("four literal signs are required")
    if cue_transform not in (-1, 0, 1):
        raise ValueError("cue transform must erase, preserve or invert")
    times = np.datetime64("2026-01-01", "ns") + np.arange(25) * np.timedelta64(1, "h")
    cues = np.zeros(25)
    close = np.full((25, 1), 100.0)
    volume = np.full((25, 1), capacity, dtype=float)
    for block, side in enumerate(signs):
        start = 6 + 4 * block
        cues[start : start + 3] = side * cue_transform
        close[start + 3 :, 0] *= 1.0 + price_response * side
        volume[start + 2 : start + 4] = 0
    if future_perturbation is not None:
        if type(future_perturbation) is not int or not 6 <= future_perturbation < 22:
            raise ValueError("future perturbation requires a declared actor index")
        close[future_perturbation + 1 :] *= 1.4
        volume[future_perturbation + 1 :] = volume[future_perturbation + 1 :] * 1.7 + 17
    opens = close.copy()
    opens[1:, 0] = close[:-1, 0]
    available = np.ones((25, 1, 2), dtype=bool)
    available[22:] = False
    dataset = MarketDataset(
        dataset_id=content_digest(
            {
                "settled_delayed_cue_v1": list(signs),
                "price_response": price_response,
                "cue_transform": cue_transform,
                "capacity": capacity,
                "future_perturbation": future_perturbation,
            }
        ),
        symbols=("S0",),
        timestamps=times,
        features=np.stack((np.ones(25), cues), axis=1)[:, None, :].astype(np.float32),
        feature_names=("constant", "current_cue"),
        feature_available=available,
        global_features=np.zeros((25, 1), dtype=np.float32),
        global_feature_names=("regime",),
        periods_per_year=8760,
        open=opens,
        high=np.maximum(opens, close),
        low=np.minimum(opens, close),
        close=close,
        mark_price=close,
        index_price=close,
        volume=volume,
        volume_units=(VolumeUnit.BASE_ASSET,),
        funding_rate=np.zeros((25, 1)),
        tradable=np.ones((25, 1), dtype=bool),
        split_factor=np.ones((25, 1)),
    )
    stream = fit_prequential_simple_ridge(
        dataset,
        blocks=(
            ForecastBlock(
                times[5], times[5] + np.timedelta64(15, "m"), times[6], times[22]
            ),
        ),
        feature_indices=(0,),
        horizon_hours=1,
    )
    assert all(
        packet.expected_simple_return == packet.fit_prefix_marginal_variance == 0
        for packet in stream.packets
    )
    cost = replace(ExecutionCostConfig.zero(), fee_rate=fee, max_participation_rate=1.0)
    risk = PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=1.0,
        max_turnover=None,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    action = AllocationActionContract("direct", 0.5)
    allocator = AfterCostTargetAllocator(
        lower_weight=-0.5, upper_weight=0.5, risk_aversion=0.0, max_turnover=None
    )
    clock = SettledAllocationClockContract(
        actor_steps=16,
        reserve_steps=2,
        execution_interval_seconds=3600,
        rollout_steps=32,
        gamma=1.0,
        gae_lambda=0.95,
    )
    schema = AllocationObservationSchema(("current_cue",), 1, 1000.0, 18)
    executor = MarketExecutor(dataset, cost, insolvency_valuation="retain_debt")
    economics = executor.execution_policy_digest
    profile = AllocationRuntimeProfile(
        economics, content_digest(risk), 1000.0, "USD", 3600, 18 * 3600
    )
    recipe = allocation_recipe_payload_v4(
        action,
        ("current_cue",),
        allocator=allocator,
        expected_horizon_seconds=3600,
        runtime_profile=profile,
        settled_clock=clock,
        observation_schema=schema,
    )
    objective = ObjectiveContract(
        CapitalContract("independent_symbol", "USD", (1000.0,)),
        datetime(2026, 1, 1, 6, tzinfo=UTC),
        datetime(2026, 1, 1, 6, tzinfo=UTC) + timedelta(hours=18),
        "settled",
        economics,
        content_digest(risk),
        content_digest(recipe),
        maximum_drawdown=0.2,
    )
    estimates = tuple(
        HorizonCostEstimates(
            "S0",
            times[i],
            times[i],
            times[i + 1],
            "declared-synthetic-fee",
            buy_cost=fee,
            sell_cost=fee,
        )
        for i in range(6, 22)
    )
    return SettledAllocationTradingEnv(
        dataset,
        stream=stream,
        estimates=estimates,
        bound=BoundSettledAllocationClock(objective, clock),
        action_contract=action,
        allocator=allocator,
        execution_cost=cost,
        risk_config=risk,
        feature_indices=(1,),
        symbol_index=0,
        start_index=6,
        account_id="synthetic-settled-S0",
        observation_schema=schema,
    )


def cue_rule(observation):
    """Known current cue plus actual position, with no future read or forecast."""
    cue = observation[0]
    if cue == 0:
        return 2
    if cue * observation[11] > 0:
        return 0
    return 3 if cue > 0 else 1


def run(env, actor, *, diagnostic=None):
    """Execute each native actor/reserve once and preserve full diagnostics."""
    result = {} if diagnostic is None else diagnostic
    transitions, rewards, entries, alignment = [], [], [], []
    result.update(
        {
            "status": "INCOMPLETE",
            "actor_rewards": rewards,
            "entry_observations": entries,
            "transitions": transitions,
            "effective_directions": alignment,
        }
    )
    try:
        observation, _ = env.reset(seed=41)
        while True:
            index = env.index
            if (index - 6) % 4 == 0:
                entries.append(observation.tolist())
            observation, reward, done, truncated, info = env.step(actor(observation))
            # Retain the returned transition before any diagnostic assertion.
            rewards.append(float(reward))
            transitions.append(json.loads(info["settled_transition"]))
            result["witness"] = env._witness()
            assert not truncated
            if (index - 6) % 4 == 2:
                response = (
                    env.dataset.close[index + 1, 0] / env.dataset.close[index, 0] - 1
                )
                alignment.append(int(response * env.book.weights[0] > 0))
            if done:
                break
        witness = env._witness()
        gain = (env.book.portfolio_value - 1000.0) / 1000.0
        result.update(
            gain=gain,
            alignment=sum(alignment) / 4,
            fill_count=env.book.fill_count,
            witness=witness,
        )
        assert witness["settled"] and witness["eligible"]
        assert witness["physical_steps_processed"] == 18
        assert witness["maximum_drawdown"] <= 0.2
        assert all(q == 0 for q in env.book.exact_quantities)
        assert not env.order_book.active_orders
        assert abs(sum(rewards) - gain) < 1e-12
        result["status"] = "VALID"
        return result
    except BaseException as error:
        result["status"] = "INVALID"
        result["error"] = error_summary(error)
        try:
            result["witness"] = env._witness()
            receipt = getattr(error, "settled_execution_receipt", None)
            if receipt is not None:
                result["failed_execution_receipt"] = json.loads(receipt)
        except BaseException as diagnostic_error:
            note_failure(error, "Partial frontier capture failed", diagnostic_error)
        raise
