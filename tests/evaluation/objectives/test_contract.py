from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from fractions import Fraction
from typing import Any

import pytest

from trade_rl.evaluation.objectives import (
    CapitalContract,
    ObjectiveContract,
    net_equity_increment,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
STOP = START + timedelta(days=7)


class _FoldAwareTimezone(tzinfo):
    """Portable repeated-hour fixture independent of the platform timezone DB."""

    def utcoffset(self, value: datetime | None) -> timedelta:
        return timedelta(hours=-5 if value is not None and value.fold else -4)

    def dst(self, value: datetime | None) -> timedelta:
        return timedelta(0)


def _objective(**changes: Any) -> ObjectiveContract:
    values = {
        "capital": CapitalContract("independent_symbol", "USDT", (100.0,)),
        "evaluation_start": START,
        "evaluation_stop_exclusive": STOP,
        "terminal_valuation": "settled",
        "economics_digest": "a" * 64,
        "risk_digest": "b" * 64,
        "deployment_recipe_digest": "c" * 64,
    }
    values.update(changes)
    return ObjectiveContract(**values)


def test_expected_terminal_profit_and_expected_log_rank_policies_differently() -> None:
    objective = _objective()
    uncertain = (
        objective.net_profit_rate((120.0,)),
        objective.net_profit_rate((90.0,)),
    )
    certain = objective.net_profit_rate((104.5,))
    assert math.fsum(uncertain) / 2 == pytest.approx(0.05)
    assert certain == pytest.approx(0.045)
    assert math.fsum(uncertain) / 2 > certain
    assert (math.log(1.2) + math.log(0.9)) / 2 < math.log(1.045)


def test_capital_denominator_distinguishes_five_independent_books_from_one() -> None:
    independent = CapitalContract("independent_symbol", "USDT", (100.0,) * 5)
    shared = CapitalContract("shared_portfolio", "USDT", (100.0,))
    assert independent.total_initial_capital == 500.0
    assert shared.total_initial_capital == 100.0
    assert _objective(capital=independent).net_profit_rate(
        (110.0,) * 5
    ) == pytest.approx(0.10)
    assert _objective(capital=shared).net_profit_rate((150.0,)) == pytest.approx(0.50)


def test_unequal_capitals_use_total_capital_not_mean_of_book_returns() -> None:
    capital = CapitalContract("independent_symbol", "USDT", (100.0, 900.0))
    assert _objective(capital=capital).net_profit_rate((110.0, 900.0)) == pytest.approx(
        0.01
    )


def test_large_book_cancellation_preserves_another_books_one_unit_profit() -> None:
    initial = (float(2**53), 1.0)
    terminal = (float(2**53), 2.0)
    capital = CapitalContract("independent_symbol", "USDT", initial)
    exact_initial = sum((Fraction(value) for value in initial), Fraction(0))
    exact_terminal = sum((Fraction(value) for value in terminal), Fraction(0))
    assert exact_terminal - exact_initial == 1
    expected = float((exact_terminal - exact_initial) / exact_initial)
    assert _objective(capital=capital).net_profit_rate(terminal) == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )


def test_external_cash_flows_telescope_without_treating_deposits_as_profit() -> None:
    first = net_equity_increment(
        100.0, 120.0, initial_capital=100.0, net_external_cash_flow=10.0
    )
    second = net_equity_increment(
        120.0, 90.0, initial_capital=100.0, net_external_cash_flow=-40.0
    )
    endpoint = _objective().net_profit_rate((90.0,), net_external_cash_flows=(-30.0,))
    assert first == pytest.approx(0.10)
    assert second == pytest.approx(0.10)
    assert first + second == pytest.approx(endpoint)
    assert endpoint == pytest.approx(0.20)
    assert (
        _objective().net_profit_rate((150.0,), net_external_cash_flows=(50.0,)) == 0.0
    )


def test_negative_terminal_equity_preserves_debt_and_loss() -> None:
    assert _objective().net_profit_rate((-25.0,)) == pytest.approx(-1.25)
    assert net_equity_increment(0.0, -25.0, initial_capital=100.0) == pytest.approx(
        -0.25
    )


@pytest.mark.parametrize(
    "capital",
    [(), (0.0,), (-1.0,), (math.nan,), (math.inf,), (True,), [100.0]],
)
def test_invalid_initial_capital_is_rejected(capital: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CapitalContract("independent_symbol", "USDT", capital)


@pytest.mark.parametrize("mode", ["portfolio", "", None, True])
def test_unknown_account_mode_is_rejected(mode: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CapitalContract(mode, "USDT", (100.0,))


def test_shared_portfolio_cannot_hide_multiple_capital_books() -> None:
    with pytest.raises((TypeError, ValueError)):
        CapitalContract("shared_portfolio", "USDT", (100.0, 100.0))


@pytest.mark.parametrize("currency", ["", "   ", None, True])
def test_missing_account_currency_is_rejected(currency: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        CapitalContract("independent_symbol", currency, (100.0,))


@pytest.mark.parametrize(
    "terminal",
    [(), (100.0, 100.0), (math.nan,), (math.inf,), (True,), [100.0], None],
)
def test_invalid_terminal_endpoint_is_rejected(terminal: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _objective().net_profit_rate(terminal)


@pytest.mark.parametrize(
    "flows", [(), (0.0, 0.0), (math.nan,), (math.inf,), (True,), [0.0]]
)
def test_invalid_external_cash_flow_endpoint_is_rejected(flows: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _objective().net_profit_rate((100.0,), net_external_cash_flows=flows)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_start", START.replace(tzinfo=None)),
        ("evaluation_stop_exclusive", STOP.replace(tzinfo=None)),
        ("evaluation_stop_exclusive", START),
        ("evaluation_stop_exclusive", START - timedelta(seconds=1)),
        ("evaluation_start", "2026-01-01"),
        ("terminal_valuation", "free_reset"),
        ("maximum_drawdown", math.nan),
        ("maximum_drawdown", math.inf),
        ("maximum_drawdown", True),
        ("economics_digest", "a" * 63),
        ("risk_digest", "x" * 64),
        ("deployment_recipe_digest", ""),
    ],
)
def test_invalid_objective_declaration_is_rejected(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _objective(**{field: value})


def test_objective_identity_includes_fixed_business_and_cash_flow_conventions() -> None:
    objective = _objective()
    payload = objective.payload()
    assert payload["schema"] == "net_profit_objective_v1"
    assert payload["objective_id"] == "expected_terminal_net_profit_v1"
    assert payload["tax_treatment"] == "pretax"
    assert payload["infrastructure_cost_treatment"] == "reported_separately"
    assert (
        payload["external_cash_flow_convention"]
        == "net_deposits_positive_withdrawals_negative"
    )
    assert payload["capital"] == {
        "mode": "independent_symbol",
        "currency": "USDT",
        "initial_equities": [100.0],
    }
    assert payload["maximum_drawdown"] == 0.20
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert objective.digest == hashlib.sha256(encoded).hexdigest()


def test_equivalent_aware_timezone_bounds_have_one_utc_identity() -> None:
    local = timezone(timedelta(hours=9))
    utc = _objective()
    shifted = _objective(
        evaluation_start=START.astimezone(local),
        evaluation_stop_exclusive=STOP.astimezone(local),
    )
    assert shifted.payload() == utc.payload()
    assert shifted.digest == utc.digest
    for key in ("evaluation_start", "evaluation_stop_exclusive"):
        assert datetime.fromisoformat(utc.payload()[key]).utcoffset() == timedelta(0)


def test_wall_clock_increasing_dst_fold_bounds_cannot_hide_reversed_utc_time() -> None:
    local = _FoldAwareTimezone()
    start = datetime(2026, 11, 1, 1, 15, tzinfo=local, fold=1)
    stop = datetime(2026, 11, 1, 1, 30, tzinfo=local, fold=0)
    assert stop.astimezone(UTC) < start.astimezone(UTC)
    with pytest.raises(ValueError):
        _objective(evaluation_start=start, evaluation_stop_exclusive=stop)


def test_repeated_hour_window_is_ordered_by_utc_not_local_wall_clock() -> None:
    local = _FoldAwareTimezone()
    start = datetime(2026, 11, 1, 1, 30, tzinfo=local, fold=0)
    stop = datetime(2026, 11, 1, 1, 15, tzinfo=local, fold=1)
    assert stop.astimezone(UTC) - start.astimezone(UTC) == timedelta(minutes=45)
    actual = _objective(evaluation_start=start, evaluation_stop_exclusive=stop)
    expected = _objective(
        evaluation_start=start.astimezone(UTC),
        evaluation_stop_exclusive=stop.astimezone(UTC),
    )
    assert actual.payload() == expected.payload()
    assert actual.digest == expected.digest


@pytest.mark.parametrize(
    "changes",
    [
        {"capital": CapitalContract("shared_portfolio", "USDT", (100.0,))},
        {"capital": CapitalContract("independent_symbol", "USD", (100.0,))},
        {"capital": CapitalContract("independent_symbol", "USDT", (200.0,))},
        {"evaluation_start": START + timedelta(seconds=1)},
        {"evaluation_stop_exclusive": STOP + timedelta(seconds=1)},
        {"terminal_valuation": "marked_continuation"},
        {"economics_digest": "d" * 64},
        {"risk_digest": "d" * 64},
        {"deployment_recipe_digest": "d" * 64},
        {"maximum_drawdown": 0.10},
    ],
)
def test_material_objective_change_cannot_reuse_old_identity(
    changes: dict[str, object],
) -> None:
    assert _objective(**changes).digest != _objective().digest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("before", math.nan),
        ("after", math.inf),
        ("after", True),
        ("initial_capital", 0.0),
        ("initial_capital", True),
        ("net_external_cash_flow", math.nan),
        ("net_external_cash_flow", True),
    ],
)
def test_invalid_equity_increment_is_rejected(field: str, value: object) -> None:
    values = {
        "before": 100.0,
        "after": 105.0,
        "initial_capital": 100.0,
        "net_external_cash_flow": 0.0,
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        net_equity_increment(**values)
