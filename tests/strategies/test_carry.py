from datetime import datetime

import numpy as np
import pytest

from trade_rl.strategies import carry


def test_equal_base_units_size_from_only_observed_prices() -> None:
    assert hasattr(carry, "FundingCarryBot")
    bot = carry.FundingCarryBot(carry.CarryConfig())
    target = bot.decide(
        timestamp=datetime(2023, 1, 1),
        prices=np.array([100.0, 102.0, 20.0, 21.0]),
        quantities=np.zeros(4),
        equity=10_000.0,
        drawdown=0.0,
    )
    np.testing.assert_allclose(target, [12.254, -12.254, 59.523, -59.523])


def test_hold_preserves_quantity_and_unmatched_fill_stops_irreversibly() -> None:
    bot = carry.FundingCarryBot(carry.CarryConfig())
    arguments = dict(
        timestamp=datetime(2023, 1, 1),
        prices=np.array([100.0, 100.0]),
        equity=1000.0,
        drawdown=0.0,
    )
    initial = bot.decide(quantities=np.zeros(2), **arguments)
    arguments["timestamp"] = datetime(2023, 1, 2)
    arguments["prices"] = np.array([120.0, 120.0])
    np.testing.assert_array_equal(bot.decide(quantities=initial, **arguments), initial)
    np.testing.assert_array_equal(
        bot.decide(quantities=np.array([2.5, -2.0]), **arguments), np.zeros(2)
    )
    assert bot.stop_reason == "unmatched_hedge"
    np.testing.assert_array_equal(bot.decide(quantities=np.zeros(2), **arguments), 0)


@pytest.mark.parametrize("value", [True, float("nan"), 0.0, 1.1])
def test_config_rejects_invalid_gross(value: float) -> None:
    with pytest.raises(ValueError):
        carry.CarryConfig(gross_budget=value)


def test_sized_quantity_uses_the_canonical_decimal_lot() -> None:
    bot = carry.FundingCarryBot(carry.CarryConfig())
    target = bot.decide(
        timestamp=datetime(2023, 1, 1),
        prices=np.full(2, 100000.0),
        quantities=np.zeros(2),
        equity=3960.0,
        drawdown=0.0,
    )
    np.testing.assert_array_equal(target, [0.009, -0.009])


def test_only_new_calendar_month_resizes_and_drawdown_stop_is_permanent() -> None:
    bot = carry.FundingCarryBot(carry.CarryConfig())
    old = bot.decide(
        timestamp=datetime(2023, 1, 31),
        prices=np.full(2, 100.0),
        quantities=np.zeros(2),
        equity=4000.0,
        drawdown=0.0,
    )
    new = bot.decide(
        timestamp=datetime(2023, 2, 1),
        prices=np.full(2, 200.0),
        quantities=old,
        equity=4800.0,
        drawdown=0.0,
    )
    np.testing.assert_array_equal(new, [6, -6])
    stopped = bot.decide(
        timestamp=datetime(2023, 2, 2),
        prices=np.full(2, 200.0),
        quantities=new,
        equity=4300.0,
        drawdown=0.1,
    )
    np.testing.assert_array_equal(stopped, [0, 0])
    assert bot.stop_reason == "maximum_drawdown"
    recovered = bot.decide(
        timestamp=datetime(2023, 3, 1),
        prices=np.full(2, 200.0),
        quantities=np.zeros(2),
        equity=5000.0,
        drawdown=0.0,
    )
    np.testing.assert_array_equal(recovered, [0, 0])
