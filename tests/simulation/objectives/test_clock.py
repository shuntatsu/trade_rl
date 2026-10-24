from __future__ import annotations

import math

import pytest

from trade_rl.simulation.objectives import FinancialClockContract


def _clock(**changes: object) -> FinancialClockContract:
    values: dict[str, object] = {
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


def test_discount_is_bound_to_elapsed_time_and_terminal_alignment_is_declared() -> None:
    hourly = _clock()
    four_hourly = _clock(
        decision_interval_seconds=14_400,
        reward_interval_seconds=14_400,
        gamma=0.99**4,
    )
    three_days = 72 * 3600

    assert hourly.discount_after(three_days) == pytest.approx(0.99**72)
    assert four_hourly.discount_after(three_days) == pytest.approx(
        hourly.discount_after(three_days)
    )
    assert hourly.terminal_profit_aligned is False
    assert _clock(reward_schema="equity_delta_v1", gamma=1.0).terminal_profit_aligned
    assert not _clock(
        reward_schema="equity_delta_v1", gamma=0.99
    ).terminal_profit_aligned
    assert not _clock(
        reward_schema="net_log_return_v1", gamma=1.0
    ).terminal_profit_aligned


@pytest.mark.parametrize("gamma", [1.0, 0.99])
def test_discount_handles_huge_elapsed_seconds_without_rejecting_valid_input(
    gamma: float,
) -> None:
    huge_seconds = 10**400
    clock = _clock(gamma=gamma)

    expected = 1.0 if gamma == 1.0 else 0.0
    assert clock.discount_after(huge_seconds) == expected


def test_discount_preserves_finite_ratio_for_huge_elapsed_and_decision_intervals() -> (
    None
):
    huge_seconds = 10**400
    clock = _clock(
        decision_interval_seconds=huge_seconds,
        execution_interval_seconds=huge_seconds,
        reward_interval_seconds=huge_seconds,
        economic_horizon_seconds=huge_seconds,
        gamma=0.99,
    )

    assert clock.discount_after(huge_seconds) == pytest.approx(0.99)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decision_interval_seconds", True),
        ("decision_interval_seconds", 0),
        ("execution_interval_seconds", 1000),
        ("reward_interval_seconds", 7200),
        ("economic_horizon_seconds", 7 * 24 * 3600 + 1),
        ("rollout_steps", False),
        ("gamma", math.nan),
        ("gamma", 0.0),
        ("gae_lambda", math.inf),
        ("gae_lambda", 1.01),
        ("reward_schema", "log_profit_v1"),
    ],
)
def test_incompatible_clock_contract_is_rejected(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _clock(**{field: value})


@pytest.mark.parametrize("seconds", [-1, True, 3600.0, math.nan, math.inf, None])
def test_invalid_elapsed_discount_duration_is_rejected(seconds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _clock().discount_after(seconds)
