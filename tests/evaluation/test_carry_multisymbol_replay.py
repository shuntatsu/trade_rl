"""G2 independent oracle for the multi-pair carry replay (synthetic, hand-computed).

The expected values below are derived from the frozen contract on paper (quantity =
notional / price, per-leg fee plus adverse execution, funding on the short
perpetual) with exact rational arithmetic. Nothing here calls production sizing,
cost or funding code to build the expectation.
"""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.integrations.test_binance_carry import series
from trade_rl.evaluation import carry
from trade_rl.integrations.binance.carry import (
    assemble_carry_dataset,
    assemble_multisymbol_carry_dataset,
)

UNIVERSE = ("AAAUSDT", "BBBUSDT", "CCCUSDT")


def priced_pair(price: float, funding_multiplier: float):
    spot, perp = series(perpetual=False), series(perpetual=True)
    flat = np.full(48, price)
    spot = replace(spot, open=flat, high=flat, low=flat, close=flat)
    perp = replace(
        perp,
        open=flat,
        high=flat,
        low=flat,
        close=flat,
        funding_rate=perp.funding_rate * funding_multiplier,
    )
    return spot, perp


def test_three_pair_replay_matches_the_hand_computed_ledger() -> None:
    # Funding settles in bars 7 and 15 (bar index = completed hour). The hold is
    # entered at bar 6 and exited at bar 18, so both events accrue.
    pairs = {
        "AAAUSDT": priced_pair(100.0, 2.0),  # +0.0002 per event
        "BBBUSDT": priced_pair(50.0, 1.0),  # +0.0001 per event
        "CCCUSDT": priced_pair(250.0, -1.0),  # -0.0001 per event, the short pays
    }
    dataset = assemble_multisymbol_carry_dataset(pairs, universe=UNIVERSE)
    result = carry.replay_carry(
        dataset, start_index=5, stop_index=18, initial_capital=6000
    )

    capital = Fraction(6000)
    gross, legs = Fraction(1, 2), 6
    leg_notional = capital * gross / legs  # 500 per leg, 3000 gross in total
    prices = {"AAAUSDT": 100, "BBBUSDT": 50, "CCCUSDT": 250}
    rates = {
        "AAAUSDT": Fraction(2, 10_000),
        "BBBUSDT": Fraction(1, 10_000),
        "CCCUSDT": Fraction(-1, 10_000),
    }
    spot_cost = Fraction(10, 10_000) + Fraction(5, 10_000)  # 10bp fee + 5bp adverse
    perp_cost = Fraction(5, 10_000) + Fraction(5, 10_000)  # 5bp fee + 5bp adverse
    quantities, expected_cost, expected_funding = {}, Fraction(0), Fraction(0)
    for symbol, price in prices.items():
        quantity = leg_notional / price
        quantities[symbol] = quantity
        # entry and exit each trade one spot and one perpetual leg of 500 USDT
        expected_cost += 2 * leg_notional * (spot_cost + perp_cost)
        expected_funding += 2 * quantity * price * rates[symbol]
    assert expected_cost == Fraction(15, 2) and expected_funding == Fraction(1, 5)

    sorted_symbols = sorted(prices)
    hold = [float(quantities[s]) * sign for s in sorted_symbols for sign in (1, -1)]
    assert hold == [5.0, -5.0, 10.0, -10.0, 2.0, -2.0]
    np.testing.assert_array_equal(result["quantities"][0], hold)
    np.testing.assert_array_equal(result["quantities"][-1], np.zeros(6))
    assert result["total_cost"] == pytest.approx(float(expected_cost))
    assert result["funding_pnl"] == pytest.approx(float(expected_funding))
    final_equity = capital - expected_cost + expected_funding
    assert result["equity"][-1] == pytest.approx(float(final_equity))
    assert float(final_equity) == pytest.approx(5992.7)
    assert result["terminal_flat"] and result["execution_valid"]
    assert result["stop_reason"] is None
    assert np.prod(1 + np.asarray(result["returns"])) * 6000 == pytest.approx(5992.7)


def test_pair_count_contract_is_even_bounded_and_keeps_the_btc_eth_cases() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    assert carry.MAX_CARRY_PAIRS == 11
    for symbols in (("BTCUSDT",), ("BTCUSDT", "ETHUSDT")):
        dataset = assemble_carry_dataset({s: (spot, perp) for s in symbols})
        carry.replay_carry(dataset, start_index=5, stop_index=9, initial_capital=4000)
    eleven = tuple(f"S{i:02d}USDT" for i in range(11))
    wide = assemble_multisymbol_carry_dataset(
        {s: (spot, perp) for s in eleven}, universe=eleven
    )
    result = carry.replay_carry(wide, start_index=5, stop_index=9, initial_capital=4000)
    assert wide.n_symbols == 22 and result["terminal_flat"]
    odd = replace(wide, symbols=wide.symbols[:-1] + ("S10USDT:odd",))
    with pytest.raises(ValueError, match="alternating"):
        carry.replay_carry(odd, start_index=5, stop_index=9, initial_capital=4000)


def test_pair_count_limits_agree_between_assembler_and_replay() -> None:
    from trade_rl.integrations.binance import carry as source

    assert source.MAX_MULTISYMBOL_PAIRS == carry.MAX_CARRY_PAIRS
