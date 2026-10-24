from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction

import pytest

from trade_rl.simulation.objectives import (
    CapitalContract,
    ObjectiveContract,
    net_equity_increment,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
STOP = START + timedelta(days=7)


def _objective(**changes: object) -> ObjectiveContract:
    values: dict[str, object] = {
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


def test_terminal_net_profit_can_rank_policies_differently_from_log_utility() -> None:
    objective = _objective()
    uncertain_expected_profit = (
        math.fsum(
            (objective.net_profit_rate((120.0,)), objective.net_profit_rate((90.0,)))
        )
        / 2
    )
    certain_profit = objective.net_profit_rate((104.5,))

    assert uncertain_expected_profit == pytest.approx(0.05)
    assert certain_profit == pytest.approx(0.045)
    assert uncertain_expected_profit > certain_profit
    assert (math.log(1.2) + math.log(0.9)) / 2 < math.log(1.045)


def test_fixed_initial_capital_increments_telescope_external_cash_flows() -> None:
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


def test_capital_denominator_matches_independent_books_or_one_shared_book() -> None:
    independent = CapitalContract("independent_symbol", "USDT", (100.0,) * 5)
    shared = CapitalContract("shared_portfolio", "USDT", (100.0,))

    assert independent.total_initial_capital == 500.0
    assert shared.total_initial_capital == 100.0
    assert _objective(capital=independent).net_profit_rate(
        (110.0,) * 5
    ) == pytest.approx(0.10)
    assert _objective(capital=shared).net_profit_rate((150.0,)) == pytest.approx(0.50)
    unequal = CapitalContract("independent_symbol", "USDT", (100.0, 900.0))
    assert _objective(capital=unequal).net_profit_rate((110.0, 900.0)) == pytest.approx(
        0.01
    )


def test_terminal_loss_preserves_debt_and_small_book_profit_under_large_capital() -> (
    None
):
    assert _objective().net_profit_rate((-25.0,)) == pytest.approx(-1.25)
    assert net_equity_increment(0.0, -25.0, initial_capital=100.0) == pytest.approx(
        -0.25
    )

    initial = (float(2**53), 1.0)
    terminal = (float(2**53), 2.0)
    capital = CapitalContract("independent_symbol", "USDT", initial)
    initial_exact = sum((Fraction(value) for value in initial), Fraction(0))
    terminal_exact = sum((Fraction(value) for value in terminal), Fraction(0))
    expected = float((terminal_exact - initial_exact) / initial_exact)
    assert _objective(capital=capital).net_profit_rate(terminal) == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )


def test_objective_canonical_identity_normalizes_equivalent_aware_bounds() -> None:
    objective = _objective()
    shifted = _objective(
        evaluation_start=START.astimezone(timezone(timedelta(hours=9))),
        evaluation_stop_exclusive=STOP.astimezone(timezone(timedelta(hours=9))),
    )

    assert objective.payload()["schema"] == "net_profit_objective_v1"
    assert objective.payload()["objective_id"] == "expected_terminal_net_profit_v1"
    assert objective.payload()["capital"] == {
        "mode": "independent_symbol",
        "currency": "USDT",
        "initial_equities": [100.0],
    }
    assert shifted.payload() == objective.payload()
    assert shifted.digest == objective.digest
    encoded = json.dumps(
        objective.payload(),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert objective.digest == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    ("mode", "currency", "equities"),
    [
        ("independent_symbol", "USDT", (True,)),
        ("independent_symbol", "USDT", (math.nan,)),
        ("independent_symbol", "USDT", (math.inf,)),
        ("independent_symbol", "USDT", (0.0,)),
        ("shared_portfolio", "USDT", (100.0, 100.0)),
        ("unknown", "USDT", (100.0,)),
        ("independent_symbol", " ", (100.0,)),
        ("independent_symbol", "USDT", (1e308, 1e308)),
    ],
)
def test_invalid_capital_contract_is_rejected(
    mode: object, currency: object, equities: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CapitalContract(mode, currency, equities)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_start", START.replace(tzinfo=None)),
        ("evaluation_stop_exclusive", START),
        ("evaluation_stop_exclusive", START - timedelta(seconds=1)),
        ("terminal_valuation", "reset"),
        ("maximum_drawdown", True),
        ("maximum_drawdown", math.nan),
        ("economics_digest", "x" * 64),
    ],
)
def test_invalid_objective_declaration_is_rejected(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _objective(**{field: value})


def test_invalid_endpoint_or_increment_numbers_fail_closed() -> None:
    with pytest.raises((TypeError, ValueError)):
        _objective().net_profit_rate((True,))
    with pytest.raises((TypeError, ValueError)):
        _objective().net_profit_rate((100.0,), net_external_cash_flows=(math.inf,))
    with pytest.raises((TypeError, ValueError)):
        net_equity_increment(100.0, 105.0, initial_capital=True)
    with pytest.raises((TypeError, ValueError)):
        net_equity_increment(
            100.0, 105.0, initial_capital=100.0, net_external_cash_flow=math.nan
        )
