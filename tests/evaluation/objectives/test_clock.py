from __future__ import annotations

import math
from typing import Any

import pytest

from trade_rl.evaluation.objectives import FinancialClockContract


def _clock(**changes: Any) -> FinancialClockContract:
    values = {
        "decision_interval_seconds": 3600,
        "execution_interval_seconds": 900,
        "reward_interval_seconds": 3600,
        "economic_horizon_seconds": 7 * 24 * 3600,
        "rollout_steps": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "reward_schema": "net_log_return_v1",
    }
    values.update(changes)
    return FinancialClockContract(**values)


def test_discount_depends_on_economic_time_not_bare_step_count() -> None:
    hourly = _clock()
    four_hourly = _clock(
        decision_interval_seconds=14400, reward_interval_seconds=14400, gamma=0.99**4
    )
    three_days = 72 * 3600
    assert hourly.discount_after(three_days) == pytest.approx(0.48499137027416284)
    assert four_hourly.discount_after(three_days) == pytest.approx(
        hourly.discount_after(three_days)
    )
    assert hourly.discount_after(0) == 1.0


def test_reusing_bare_gamma_at_a_new_decision_frequency_changes_the_objective() -> None:
    hourly = _clock()
    four_hourly = _clock(decision_interval_seconds=14400, reward_interval_seconds=14400)
    week = 7 * 24 * 3600
    assert hourly.discount_after(week) != pytest.approx(
        four_hourly.discount_after(week)
    )
    assert four_hourly.discount_after(week) > hourly.discount_after(week)


@pytest.mark.parametrize(
    ("reward_schema", "gamma", "aligned"),
    [
        ("equity_delta_v1", 1.0, True),
        ("equity_delta_v1", 0.99, False),
        ("net_log_return_v1", 1.0, False),
        ("net_log_return_v1", 0.99, False),
    ],
)
def test_only_undiscounted_equity_delta_is_terminal_profit_aligned(
    reward_schema: str, gamma: float, aligned: bool
) -> None:
    assert (
        _clock(reward_schema=reward_schema, gamma=gamma).terminal_profit_aligned
        is aligned
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision_interval_seconds", 0),
        ("decision_interval_seconds", True),
        ("execution_interval_seconds", -1),
        ("execution_interval_seconds", 900.0),
        ("reward_interval_seconds", 0),
        ("rollout_steps", 0),
        ("rollout_steps", True),
        ("economic_horizon_seconds", 3599),
        ("economic_horizon_seconds", 7 * 24 * 3600 + 1),
        ("economic_horizon_seconds", math.inf),
        ("execution_interval_seconds", 1000),
        ("execution_interval_seconds", 7200),
        ("reward_interval_seconds", 7200),
        ("gamma", 0.0),
        ("gamma", 1.01),
        ("gamma", math.nan),
        ("gamma", math.inf),
        ("gamma", True),
        ("gae_lambda", -0.01),
        ("gae_lambda", 1.01),
        ("gae_lambda", math.nan),
        ("gae_lambda", True),
        ("reward_schema", "profit_equals_log"),
    ],
)
def test_inconsistent_or_invalid_financial_clock_is_rejected(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _clock(**{field: value})


@pytest.mark.parametrize("seconds", [-1, True, 3600.0, math.nan, math.inf, None])
def test_invalid_economic_discount_duration_is_rejected(seconds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _clock().discount_after(seconds)
