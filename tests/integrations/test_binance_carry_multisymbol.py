from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from tests.integrations.test_binance_carry import series
from trade_rl.integrations.binance import carry

BTC_ETH_GOLDEN = {
    # Recorded from the unmodified assembler before the multi-symbol change, so the
    # existing BTC/ETH protocols stay byte-identical (dataset identity covers every
    # array and the identity payload).
    "btc": "28a27a77901c298826a2fb0495505beb43b184a4de4b98258780e1c266f34ec4",
    "btc_eth_stress": (
        "c95586155a0b282513982691f265bb0346b5bda3ebe830ac29a35730ab7b9810"
    ),
    "btc_eth_funded": (
        "73a686be1c36ac56a0913a199cb2e8ccbd5b445c61692028ef429947c366ba02"
    ),
}
UNIVERSE = ("SOLUSDT", "BNBUSDT", "XRPUSDT")


def test_btc_eth_path_is_byte_identical_to_the_recorded_identities() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    funded = replace(perp, funding_rate=perp.funding_rate * 10)
    assert (
        carry.assemble_carry_dataset({"BTCUSDT": (spot, perp)}).dataset_id
        == BTC_ETH_GOLDEN["btc"]
    )
    assert (
        carry.assemble_carry_dataset(
            {"BTCUSDT": (spot, perp), "ETHUSDT": (spot, perp)},
            cost_multiplier=2.0,
            capacity_multiplier=0.1,
        ).dataset_id
        == BTC_ETH_GOLDEN["btc_eth_stress"]
    )
    assert (
        carry.assemble_carry_dataset(
            {"BTCUSDT": (spot, funded), "ETHUSDT": (spot, funded)}
        ).dataset_id
        == BTC_ETH_GOLDEN["btc_eth_funded"]
    )


def test_legacy_assembler_still_refuses_every_other_symbol() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    with pytest.raises(ValueError, match="BTCUSDT"):
        carry.assemble_carry_dataset({"SOLUSDT": (spot, perp)})


def test_multisymbol_assembly_keeps_the_btc_eth_economics_per_pair() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    pairs = {symbol: (spot, perp) for symbol in UNIVERSE}
    dataset = carry.assemble_multisymbol_carry_dataset(pairs, universe=UNIVERSE)
    assert dataset.symbols == (
        "BNBUSDT:spot",
        "BNBUSDT:perp",
        "SOLUSDT:spot",
        "SOLUSDT:perp",
        "XRPUSDT:spot",
        "XRPUSDT:perp",
    )
    assert dataset.identity_verified
    legacy = carry.assemble_carry_dataset({"BTCUSDT": (spot, perp)})
    for name in ("taker_fee_rate", "maker_fee_rate", "spread_rate", "lot_size"):
        np.testing.assert_array_equal(
            dataset.resolved_array(name)[:, :2], legacy.resolved_array(name)
        )
    np.testing.assert_array_equal(
        dataset.resolved_array("max_participation_rate"),
        np.full(dataset.close.shape, 0.01),
    )
    np.testing.assert_array_equal(
        dataset.resolved_array("borrow_available")[0], [False, True] * len(UNIVERSE)
    )
    np.testing.assert_array_equal(dataset.funding_rate[:, ::2], 0)
    assert dataset.funding_rate[:, 1::2].sum() == pytest.approx(0.0006 * 3)


def test_multisymbol_identity_binds_contract_universe_and_stress() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    pairs = {"SOLUSDT": (spot, perp)}
    base = carry.assemble_multisymbol_carry_dataset(pairs, universe=UNIVERSE)
    other_universe = carry.assemble_multisymbol_carry_dataset(
        pairs, universe=(*UNIVERSE, "ADAUSDT")
    )
    costly = carry.assemble_multisymbol_carry_dataset(
        pairs, universe=UNIVERSE, cost_multiplier=2.0
    )
    assert len({base.dataset_id, other_universe.dataset_id, costly.dataset_id}) == 3
    legacy = carry.assemble_carry_dataset({"BTCUSDT": (spot, perp)})
    assert legacy.dataset_id != base.dataset_id


@pytest.mark.parametrize(
    "case",
    ["undeclared", "empty", "duplicate_universe", "bad_symbol", "too_many", "one_leg"],
)
def test_multisymbol_assembly_fails_closed(case: str) -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    pairs: dict = {"SOLUSDT": (spot, perp)}
    universe = UNIVERSE
    if case == "undeclared":
        pairs = {"DOGEUSDT": (spot, perp)}
    elif case == "empty":
        pairs = {}
    elif case == "duplicate_universe":
        universe = ("SOLUSDT", "SOLUSDT")
    elif case == "bad_symbol":
        universe = ("sol-usdt",)
        pairs = {"sol-usdt": (spot, perp)}
    elif case == "too_many":
        universe = tuple(f"A{i}USDT" for i in range(12))
        pairs = {symbol: (spot, perp) for symbol in universe}
    elif case == "one_leg":
        pairs = {"SOLUSDT": (spot,)}
    with pytest.raises(ValueError):
        carry.assemble_multisymbol_carry_dataset(pairs, universe=universe)


def test_multisymbol_assembly_keeps_every_source_guard() -> None:
    spot, perp = series(perpetual=False), series(perpetual=True)
    with pytest.raises(ValueError, match="funding"):
        carry.assemble_multisymbol_carry_dataset(
            {"SOLUSDT": (spot, spot)}, universe=UNIVERSE
        )
    delayed = replace(spot, available_at=spot.available_at + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="available"):
        carry.assemble_multisymbol_carry_dataset(
            {"SOLUSDT": (delayed, perp)}, universe=UNIVERSE
        )
    with pytest.raises(ValueError, match="multipliers"):
        carry.assemble_multisymbol_carry_dataset(
            {"SOLUSDT": (spot, perp)}, universe=UNIVERSE, cost_multiplier=0.0
        )
    with pytest.raises(ValueError, match="reduce"):
        carry.assemble_multisymbol_carry_dataset(
            {"SOLUSDT": (spot, perp)}, universe=UNIVERSE, capacity_multiplier=2.0
        )


def perp_rows(opens):
    return [[ms, 100.0, 100.0, 100.0, 100.0, 5, ms + 3599999, 1000] for ms in opens]


def test_perpetual_parser_is_strict_and_aligns_funding_to_completed_bars() -> None:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2023, 1, 1, 12, tzinfo=UTC)
    opens = [int(start.timestamp() * 1000) + h * 3_600_000 for h in range(12)]
    events = [(opens[8], 0.0001)]  # 08:00:00 settles the bar closing at 08:00
    data = carry.parse_carry_perp_rows(
        perp_rows(opens), events, start_time=start, end_time=end
    )
    assert len(data.timestamps) == 12 and bool(data.tradable.all())
    assert data.funding_event_count.tolist() == [0] * 7 + [1] + [0] * 4
    assert data.funding_rate[7] == pytest.approx(0.0001)
    np.testing.assert_array_equal(data.available_at, data.timestamps)
    for defect in (opens[:4] + opens[5:], opens + [opens[-1]], opens[::-1]):
        with pytest.raises(ValueError):
            carry.parse_carry_perp_rows(
                perp_rows(defect), events, start_time=start, end_time=end
            )
