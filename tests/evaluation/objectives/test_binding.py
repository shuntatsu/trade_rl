from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta, tzinfo

import pytest

from trade_rl.evaluation.objectives import (
    BoundObjectiveClock,
    CapitalContract,
    FinancialClockContract,
    ObjectiveContract,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
STOP = START + timedelta(days=7)
HORIZON = 7 * 24 * 3600


def _objective(
    start: datetime = START,
    stop: datetime = STOP,
    *,
    economics_digest: str = "a" * 64,
) -> ObjectiveContract:
    return ObjectiveContract(
        capital=CapitalContract("independent_symbol", "USDT", (100.0,)),
        evaluation_start=start,
        evaluation_stop_exclusive=stop,
        terminal_valuation="settled",
        economics_digest=economics_digest,
        risk_digest="b" * 64,
        deployment_recipe_digest="c" * 64,
    )


def _clock(
    horizon: int = HORIZON, *, gamma: float = 0.99, decision_seconds: int = 3600
) -> FinancialClockContract:
    return FinancialClockContract(
        decision_interval_seconds=decision_seconds,
        execution_interval_seconds=900,
        reward_interval_seconds=decision_seconds,
        economic_horizon_seconds=horizon,
        rollout_steps=2048,
        gamma=gamma,
        gae_lambda=0.95,
        reward_schema="net_log_return_v1",
    )


def test_matched_horizon_binds_both_existing_declaration_identities() -> None:
    objective, clock = _objective(), _clock()
    binding = BoundObjectiveClock(objective, clock)
    expected = {
        "schema": "bound_objective_clock_v1",
        "objective_contract_digest": objective.digest,
        "financial_clock_digest": clock.digest,
    }
    assert binding.payload() == expected
    raw = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    assert binding.digest == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("start_shift", [1, 3600])
def test_shortened_utc_period_cannot_reuse_the_original_clock(start_shift: int) -> None:
    objective = _objective(START + timedelta(seconds=start_shift), STOP)
    with pytest.raises(ValueError):
        BoundObjectiveClock(objective, _clock())


def test_fractional_second_cannot_disappear_when_elapsed_float_rounds() -> None:
    start = datetime(1000, 1, 1, tzinfo=UTC)
    stop = START + timedelta(microseconds=1)
    elapsed = stop - start
    whole_seconds = elapsed.days * 86400 + elapsed.seconds
    assert elapsed.microseconds == 1
    assert elapsed.total_seconds() == whole_seconds
    with pytest.raises(ValueError):
        BoundObjectiveClock(_objective(start, stop), _clock(whole_seconds))


class _RepeatedHourTimezone(tzinfo):
    def utcoffset(self, value: datetime | None) -> timedelta:
        return timedelta(hours=-5 if value is not None and value.fold else -4)

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)


def test_dst_fold_uses_the_objectives_normalized_utc_duration() -> None:
    local = _RepeatedHourTimezone()
    start = datetime(2026, 11, 1, 1, 30, tzinfo=local, fold=0)
    stop = datetime(2026, 11, 1, 1, 15, tzinfo=local, fold=1)
    clock = _clock(45 * 60, decision_seconds=900)
    actual = BoundObjectiveClock(_objective(start, stop), clock)
    expected = BoundObjectiveClock(
        _objective(start.astimezone(UTC), stop.astimezone(UTC)), clock
    )
    assert actual.payload() == expected.payload()
    assert actual.digest == expected.digest


def test_change_to_either_existing_contract_changes_combined_identity() -> None:
    original = BoundObjectiveClock(_objective(), _clock())
    economic_change = BoundObjectiveClock(
        _objective(economics_digest="d" * 64), _clock()
    )
    discount_change = BoundObjectiveClock(_objective(), _clock(gamma=1.0))
    assert len({original.digest, economic_change.digest, discount_change.digest}) == 3


def test_large_integer_clock_horizon_is_rejected_without_float_overflow() -> None:
    with pytest.raises(ValueError):
        BoundObjectiveClock(_objective(), _clock(3600 * 10**400))


@pytest.mark.parametrize("wrong", [None, {}, object(), "declared-digest"])
def test_wrong_objective_contract_type_is_rejected(wrong: object) -> None:
    with pytest.raises(ValueError):
        BoundObjectiveClock(wrong, _clock())


@pytest.mark.parametrize("wrong", [None, {}, object(), "declared-digest"])
def test_wrong_clock_contract_type_is_rejected(wrong: object) -> None:
    with pytest.raises(ValueError):
        BoundObjectiveClock(_objective(), wrong)


def test_binding_cannot_mutate_its_nested_contract_references() -> None:
    binding = BoundObjectiveClock(_objective(), _clock())
    before = binding.digest
    with pytest.raises(AttributeError):
        setattr(binding, "objective", _objective(economics_digest="d" * 64))
    assert binding.digest == before
