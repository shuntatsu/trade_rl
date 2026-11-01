from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from trade_rl.simulation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
HORIZON = 7 * 24 * 3600


def _objective(**changes: object) -> ObjectiveContract:
    values: dict[str, object] = {
        "capital": CapitalContract("shared_portfolio", "USDT", (100.0,)),
        "evaluation_start": START,
        "evaluation_stop_exclusive": START + timedelta(seconds=HORIZON),
        "terminal_valuation": "settled",
        "economics_digest": "a" * 64,
        "risk_digest": "b" * 64,
        "deployment_recipe_digest": "c" * 64,
    }
    values.update(changes)
    return ObjectiveContract(**values)


def _clock(**changes: object) -> FinancialClockContract:
    values: dict[str, object] = {
        "decision_interval_seconds": 3600,
        "execution_interval_seconds": 900,
        "reward_interval_seconds": 3600,
        "economic_horizon_seconds": HORIZON,
        "rollout_steps": 2048,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "reward_schema": "equity_delta_v1",
    }
    values.update(changes)
    return FinancialClockContract(**values)


def test_bound_clock_requires_exact_evaluation_duration_and_binds_both_contracts() -> (
    None
):
    objective = _objective()
    clock = _clock()
    bound = BoundObjectiveClock(objective, clock)

    assert bound.payload() == {
        "schema": "bound_objective_clock_v1",
        "objective_contract_digest": objective.digest,
        "financial_clock_digest": clock.digest,
    }
    assert len(bound.digest) == 64

    for drift in (timedelta(seconds=1), timedelta(microseconds=1)):
        with pytest.raises(ValueError, match="economic horizon"):
            BoundObjectiveClock(
                _objective(
                    evaluation_stop_exclusive=objective.evaluation_stop_exclusive
                    + drift
                ),
                clock,
            )


def test_bound_clock_uses_normalized_utc_window_identity() -> None:
    local = timezone(timedelta(hours=9))
    objective = _objective(
        evaluation_start=START.astimezone(local),
        evaluation_stop_exclusive=(START + timedelta(seconds=HORIZON)).astimezone(
            local
        ),
    )
    utc = _objective()

    assert (
        BoundObjectiveClock(objective, _clock()).digest
        == BoundObjectiveClock(utc, _clock()).digest
    )


def test_bound_clock_rejects_wrong_declaration_types() -> None:
    with pytest.raises(ValueError, match="ObjectiveContract"):
        BoundObjectiveClock(object(), _clock())
    with pytest.raises(ValueError, match="FinancialClockContract"):
        BoundObjectiveClock(_objective(), object())
