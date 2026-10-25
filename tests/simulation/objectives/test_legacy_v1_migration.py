"""Synthetic legacy-v1 objective compatibility oracles for Issue #810.

The source fixture mirrors the declaration shape from historical
evaluation/objectives (PR #954), not a persisted R2/R3 market artifact.
It must never be presented as authenticated fit or economic evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo

import pytest

from trade_rl.simulation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)

START = datetime(2025, 4, 1, tzinfo=UTC)
ONE_DAY = 24 * 3600


def _objective(
    start: datetime = START,
    stop: datetime = START + timedelta(days=1),
) -> ObjectiveContract:
    return ObjectiveContract(
        capital=CapitalContract("independent_symbol", "USDT", (100_000.0,)),
        evaluation_start=start,
        evaluation_stop_exclusive=stop,
        terminal_valuation="settled",
        economics_digest="a" * 64,
        risk_digest="b" * 64,
        deployment_recipe_digest="c" * 64,
    )


def _clock(horizon: int = ONE_DAY, *, decision: int = 3600) -> FinancialClockContract:
    return FinancialClockContract(
        decision_interval_seconds=decision,
        execution_interval_seconds=decision,
        reward_interval_seconds=decision,
        economic_horizon_seconds=horizon,
        rollout_steps=32,
        gamma=1.0,
        gae_lambda=0.95,
        reward_schema="equity_delta_v1",
    )


def _old_v1_digest(payload: dict[str, object]) -> str:
    """Independent canonical-JSON hashing, not the production digest helper."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_old_v1_declaration_payloads_reconstruct_canonical_bytes_and_digests() -> None:
    """The declarations stay portable; the old module class is NOT imported."""
    objective, clock = _objective(), _clock()
    bound = BoundObjectiveClock(objective, clock)

    expected_objective: dict[str, object] = {
        "schema": "net_profit_objective_v1",
        "objective_id": "expected_terminal_net_profit_v1",
        "tax_treatment": "pretax",
        "infrastructure_cost_treatment": "reported_separately",
        "external_cash_flow_convention": "net_deposits_positive_withdrawals_negative",
        "capital": {
            "mode": "independent_symbol",
            "currency": "USDT",
            "initial_equities": [100000.0],
        },
        "evaluation_start": "2025-04-01T00:00:00+00:00",
        "evaluation_stop_exclusive": "2025-04-02T00:00:00+00:00",
        "terminal_valuation": "settled",
        "economics_digest": "a" * 64,
        "risk_digest": "b" * 64,
        "deployment_recipe_digest": "c" * 64,
        "maximum_drawdown": 0.20,
    }
    expected_clock: dict[str, object] = {
        "schema": "financial_clock_v1",
        "decision_interval_seconds": 3600,
        "execution_interval_seconds": 3600,
        "reward_interval_seconds": 3600,
        "economic_horizon_seconds": ONE_DAY,
        "rollout_steps": 32,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "reward_schema": "equity_delta_v1",
    }
    expected_bound: dict[str, object] = {
        "schema": "bound_objective_clock_v1",
        "objective_contract_digest": _old_v1_digest(expected_objective),
        "financial_clock_digest": _old_v1_digest(expected_clock),
    }
    for actual, expected in (
        (objective, expected_objective),
        (clock, expected_clock),
        (bound, expected_bound),
    ):
        assert actual.payload() == expected
        assert actual.digest == _old_v1_digest(expected)


@pytest.mark.parametrize(
    "changes",
    [
        {"capital": CapitalContract("shared_portfolio", "USDT", (100_000.0,))},
        {"capital": CapitalContract("independent_symbol", "USD", (100_000.0,))},
        {"evaluation_start": START + timedelta(seconds=1)},
        {"terminal_valuation": "marked_continuation"},
        {"economics_digest": "d" * 64},
        {"risk_digest": "d" * 64},
        {"deployment_recipe_digest": "d" * 64},
        {"maximum_drawdown": 0.10},
    ],
)
def test_material_legacy_v1_changes_cannot_reuse_objective_digest(
    changes: dict[str, object],
) -> None:
    original = _objective()
    updated = replace(original, **changes)
    assert updated.digest != original.digest


class _RepeatedHourTimezone(tzinfo):
    def utcoffset(self, value: datetime | None) -> timedelta:
        return timedelta(hours=-5 if value is not None and value.fold else -4)

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)


def test_legacy_repeated_hour_window_uses_utc_order_and_clock_binding() -> None:
    local = _RepeatedHourTimezone()
    start = datetime(2026, 11, 1, 1, 30, tzinfo=local, fold=0)
    stop = datetime(2026, 11, 1, 1, 15, tzinfo=local, fold=1)
    assert stop.astimezone(UTC) - start.astimezone(UTC) == timedelta(minutes=45)

    clock = _clock(horizon=45 * 60, decision=900)
    actual = BoundObjectiveClock(_objective(start, stop), clock)
    normalized = BoundObjectiveClock(
        _objective(start.astimezone(UTC), stop.astimezone(UTC)), clock
    )
    assert actual.payload() == normalized.payload()
    assert actual.digest == normalized.digest

    with pytest.raises(ValueError, match="follow"):
        _objective(stop, start)


def test_legacy_microsecond_fraction_cannot_be_rounded_out_of_horizon() -> None:
    start = datetime(1000, 1, 1, tzinfo=UTC)
    stop = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(microseconds=1)
    delta = stop - start
    whole_seconds = delta.days * 86400 + delta.seconds
    assert delta.microseconds == 1
    assert delta.total_seconds() == whole_seconds

    with pytest.raises(ValueError, match="economic horizon"):
        BoundObjectiveClock(_objective(start, stop), _clock(whole_seconds, decision=1))


def test_clock_overflow_is_an_intentional_forward_only_correction() -> None:
    """Old direct exponentiation could overflow; never restore that behavior."""
    discounted = replace(_clock(), gamma=0.99)
    assert discounted.discount_after(10**400) == 0.0
    assert _clock().discount_after(10**400) == 1.0
