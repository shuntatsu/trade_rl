"""USD-M contract-type admission for exchange-rule extraction."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from trade_rl.integrations.market_store.contracts import StoreMarket
from trade_rl.integrations.market_store.rules import (
    USDS_M_PERPETUAL_CONTRACT_TYPES,
    ExchangeInfoError,
    parse_symbol_rules,
)


def _payload(symbol: str, **extra: object) -> bytes:
    item: dict[str, object] = {
        "symbol": symbol,
        "status": "TRADING",
        "onboardDate": 1_765_440_300_000,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }
    item.update(extra)
    return json.dumps({"symbols": [item]}).encode()


def test_admitted_contract_types_are_exactly_the_two_perpetual_kinds() -> None:
    assert USDS_M_PERPETUAL_CONTRACT_TYPES == frozenset(
        {"PERPETUAL", "TRADIFI_PERPETUAL"}
    )


@pytest.mark.parametrize("contract_type", ["PERPETUAL", "TRADIFI_PERPETUAL"])
def test_usds_m_perpetual_kinds_are_parsed_with_exact_rules(
    contract_type: str,
) -> None:
    (rule,) = parse_symbol_rules(
        _payload("XAUUSDT", contractType=contract_type),
        market=StoreMarket.USDS_M,
        symbols=["XAUUSDT"],
    )
    assert rule.symbol == "XAUUSDT"
    assert rule.trading_status == "TRADING"
    assert rule.tick_size == Decimal("0.01")
    assert rule.lot_size == Decimal("0.001")
    assert rule.min_notional == Decimal(5)


@pytest.mark.parametrize(
    "contract_type",
    [
        "CURRENT_QUARTER",
        "NEXT_QUARTER",
        "perpetual",
        "TRADIFI_PERPETUAL ",
        "",
        None,
    ],
)
def test_usds_m_non_perpetual_or_inexact_contract_types_are_refused(
    contract_type: str | None,
) -> None:
    with pytest.raises(ExchangeInfoError, match="not a USD-M perpetual"):
        parse_symbol_rules(
            _payload("BTCUSDT_261225", contractType=contract_type),
            market=StoreMarket.USDS_M,
            symbols=["BTCUSDT_261225"],
        )


@pytest.mark.parametrize("contract_type", [["PERPETUAL"], {"PERPETUAL": 1}, 1])
def test_usds_m_non_string_contract_types_are_refused_not_crashed(
    contract_type: object,
) -> None:
    with pytest.raises(ExchangeInfoError, match="not a USD-M perpetual"):
        parse_symbol_rules(
            _payload("BTCUSDT", contractType=contract_type),
            market=StoreMarket.USDS_M,
            symbols=["BTCUSDT"],
        )


def test_usds_m_missing_contract_type_is_refused() -> None:
    with pytest.raises(ExchangeInfoError, match="not a USD-M perpetual"):
        parse_symbol_rules(
            _payload("BTCUSDT"), market=StoreMarket.USDS_M, symbols=["BTCUSDT"]
        )


def test_spot_rules_do_not_consult_contract_type() -> None:
    (rule,) = parse_symbol_rules(
        _payload("BTCUSDT"), market=StoreMarket.SPOT, symbols=["BTCUSDT"]
    )
    assert rule.symbol == "BTCUSDT"
