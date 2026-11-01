from __future__ import annotations

import ast
import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tests.evaluation import paper_ledger_oracle as oracle
from tests.evaluation.paper_ledger_oracle import (
    FundingSettlement,
    InstrumentSource,
    PaperLedger,
    RawDepth,
    SourceObservation,
    SourceSide,
)
from tests.integrations.test_binance_forward import PublicFeed
from tests.integrations.test_binance_forward_rules import RuleFeed
from trade_rl.evaluation.paper.engine import EvidenceRef, PaperEngine, PaperSettings
from trade_rl.integrations.binance.forward import capture_forward_snapshot
from trade_rl.integrations.binance.forward_rules import capture_forward_rules

BASE = datetime(2026, 9, 18, tzinfo=UTC)
CAPITAL = Decimal("10000")
LOT = Decimal("0.001")
TARGET = (Decimal("12.5"), Decimal("-12.5"), Decimal("6.25"), Decimal("-6.25"))
SETTLEMENTS = (
    FundingSettlement(
        "BTCUSDT", 1, BASE + timedelta(seconds=1), Decimal("0.01"), Decimal("100")
    ),
    FundingSettlement(
        "ETHUSDT", 3, BASE + timedelta(seconds=1), Decimal("-0.02"), Decimal("200")
    ),
    FundingSettlement(
        "BTCUSDT", 1, BASE + timedelta(seconds=3), Decimal("0.01"), Decimal("103")
    ),
    FundingSettlement(
        "ETHUSDT", 3, BASE + timedelta(seconds=3), Decimal("-0.02"), Decimal("201")
    ),
)


def _levels(
    values: tuple[tuple[str, str], ...], *, deep: bool = False
) -> tuple[tuple[Decimal, Decimal], ...]:
    return tuple(
        (Decimal(price), Decimal(quantity) * (Decimal("10") if deep else Decimal("1")))
        for price, quantity in values
    )


def source_observation(
    second: int, *, deep: bool = False, asymmetric: bool = False
) -> SourceObservation:
    size = "60" if not deep else "1000"
    btc_perpetual_size = "40" if asymmetric else size
    eth_size = "30" if not deep else "1000"
    btc = RawDepth(
        bids=_levels(
            (("100.2", btc_perpetual_size), ("100", btc_perpetual_size)),
            deep=deep,
        ),
        asks=_levels(
            (("100.3", btc_perpetual_size), ("100.5", btc_perpetual_size)),
            deep=deep,
        ),
    )
    eth = RawDepth(
        bids=_levels((("200.2", eth_size), ("200", eth_size)), deep=deep),
        asks=_levels((("200.3", eth_size), ("200.5", eth_size)), deep=deep),
    )
    btc_spot = RawDepth(
        bids=_levels((("99", size), ("98.5", size)), deep=deep),
        asks=_levels((("101", size), ("101.5", size)), deep=deep),
    )
    eth_spot = RawDepth(
        bids=_levels((("198", eth_size), ("197", eth_size)), deep=deep),
        asks=_levels((("202", eth_size), ("203", eth_size)), deep=deep),
    )
    instruments = (
        InstrumentSource(
            "BTCUSDT",
            SourceSide.SPOT,
            btc_spot,
            None,
            LOT,
            Decimal("0.001"),
            Decimal("100"),
            Decimal("5"),
            Decimal("10"),
        ),
        InstrumentSource(
            "BTCUSDT",
            SourceSide.PERPETUAL,
            btc,
            Decimal("100"),
            LOT,
            Decimal("0.001"),
            Decimal("100"),
            Decimal("50"),
            Decimal("5"),
        ),
        InstrumentSource(
            "ETHUSDT",
            SourceSide.SPOT,
            eth_spot,
            None,
            LOT,
            Decimal("0.001"),
            Decimal("100"),
            Decimal("5"),
            Decimal("10"),
        ),
        InstrumentSource(
            "ETHUSDT",
            SourceSide.PERPETUAL,
            eth,
            Decimal("200"),
            LOT,
            Decimal("0.001"),
            Decimal("100"),
            Decimal("50"),
            Decimal("5"),
        ),
    )
    at = BASE + timedelta(seconds=second)
    settled = tuple(event for event in SETTLEMENTS if event.at <= at)
    return SourceObservation(at, instruments, settled, Decimal("5"), Decimal("0.1"))


class LiteralFeed(PublicFeed):
    """Build public-response bytes from the literal source fixture."""

    def __init__(self, source: SourceObservation):
        super().__init__()
        self.source = source

    def _request_bytes(self, url, *, max_response_bytes=None):
        payload = json.loads(
            super()._request_bytes(url, max_response_bytes=max_response_bytes)
        )
        milliseconds = int(self.source.at.timestamp() * 1000)
        symbol = "ETHUSDT" if "ETHUSDT" in url else "BTCUSDT"
        if isinstance(payload, dict) and "serverTime" in payload:
            payload["serverTime"] = milliseconds
        elif isinstance(payload, dict) and "bids" in payload:
            index = 2 if symbol == "ETHUSDT" else 0
            if "fapi" in url:
                index += 1
            depth = self.source.instruments[index].depth
            payload["bids"] = [
                [str(price), str(quantity)] for price, quantity in depth.bids
            ]
            payload["asks"] = [
                [str(price), str(quantity)] for price, quantity in depth.asks
            ]
            if "fapi" in url:
                payload.update(E=milliseconds, T=milliseconds)
        elif isinstance(payload, dict) and "markPrice" in payload:
            index = 3 if symbol == "ETHUSDT" else 1
            mark = self.source.instruments[index].mark
            assert mark is not None
            payload.update(
                markPrice=str(mark),
                indexPrice=str(mark),
                lastFundingRate="0",
                nextFundingTime=milliseconds + 28_800_000,
                time=milliseconds,
            )
        elif isinstance(payload, list):
            matching = [
                event for event in self.source.settlements if event.symbol == symbol
            ]
            payload = [
                {
                    "symbol": symbol,
                    "fundingTime": int(
                        (BASE - timedelta(seconds=1)).timestamp() * 1000
                    ),
                    "fundingRate": "0",
                    "markPrice": str(
                        self.source.instruments[1 if symbol == "BTCUSDT" else 3].mark
                    ),
                },
                *[
                    {
                        "symbol": event.symbol,
                        "fundingTime": int(event.at.timestamp() * 1000),
                        "fundingRate": str(event.rate),
                        "markPrice": str(event.mark),
                    }
                    for event in matching
                ],
            ]
        return json.dumps(payload).encode()


def _capture_market(root: Path, name: str, source: SourceObservation) -> EvidenceRef:
    capture_forward_snapshot(
        root / name,
        transport=LiteralFeed(source),
        clock=lambda: source.at,
        monotonic=lambda: 0.0,
    )
    raw = (root / name / "snapshot.json").read_bytes()
    return EvidenceRef(name, hashlib.sha256(raw).hexdigest())


def _capture_rules(root: Path) -> EvidenceRef:
    directory = root / "rules"
    capture_forward_rules(
        directory,
        transport=RuleFeed(),
        clock=lambda: BASE,
        monotonic=lambda: 0.0,
    )
    raw = (directory / "rules.json").read_bytes()
    return EvidenceRef("rules", hashlib.sha256(raw).hexdigest())


def _orders(lots: tuple[int, int, int, int]) -> list[dict[str, int]]:
    return [
        {"symbol_index": index, "lot_count": lot_count}
        for index, lot_count in enumerate(lots)
        if lot_count
    ]


def _assert_close(actual: float, expected: Decimal) -> None:
    tolerance = max(2e-9, 2e-12 * abs(float(expected)))
    assert abs(actual - float(expected)) <= tolerance


def _assert_account(actual: dict, ledger: PaperLedger) -> None:
    expected = ledger.account_state()
    for field in (
        "cash",
        "equity",
        "peak_value",
        "maximum_drawdown",
        "total_cost",
        "funding_pnl",
        "turnover_total",
        "collateral",
    ):
        _assert_close(actual[field], expected[field])
    assert actual["fill_count"] == expected["fill_count"]
    assert actual["exact_quantities"] == expected["exact_quantities"]
    for got, want in zip(actual["quantities"], expected["quantities"], strict=True):
        _assert_close(got, Decimal(str(want)))
    for got, want in zip(actual["mark_prices"], expected["mark_prices"], strict=True):
        _assert_close(got, Decimal(str(want)))
    futures_requirement = sum(
        (
            abs(expected["quantities"][index]) * expected["mark_prices"][index]
            for index in (1, 3)
        ),
        Decimal("0"),
    )
    assert expected["collateral"] >= futures_requirement


def _assert_fills(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        for field in (
            "symbol_index",
            "lot_count",
            "filled_lots",
            "unfilled_lots",
            "reason",
        ):
            assert got[field] == want[field]
        if want["price"] is None:
            assert got["price"] is None
        else:
            _assert_close(got["price"], want["price"])
        _assert_close(got["fee"], want["fee"])


def test_oracle_has_no_production_accounting_or_strategy_imports():
    tree = ast.parse(Path(oracle.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert all(
        name.split(".", 1)[0]
        in {
            "__future__",
            "dataclasses",
            "datetime",
            "decimal",
            "enum",
            "fractions",
            "typing",
        }
        for name in imported
    )


def test_literal_source_to_ledger_partial_funding_and_terminal_close(tmp_path):
    settings = PaperSettings(
        start_at=BASE - timedelta(seconds=1),
        close_at=BASE + timedelta(seconds=4),
        initial_capital=float(CAPITAL),
        spot_fee_bps=10,
        perpetual_fee_bps=5,
        adverse_bps=5,
        depth_fraction=0.1,
    )
    protocol = PaperEngine.initialize(
        tmp_path / "paper",
        settings=settings,
        study={"purpose": "synthetic ledger oracle"},
    )
    engine = PaperEngine(tmp_path / "paper", expected_protocol_sha256=protocol)
    rule = _capture_rules(tmp_path / "paper")
    ledger = PaperLedger(initial_capital=CAPITAL)

    source = source_observation(0)
    ledger.observe(source)
    decision = engine.decide(
        "entry-decision",
        market=_capture_market(tmp_path / "paper", "m0", source),
        rules=rule,
        at=BASE + timedelta(seconds=0.1),
    )["event"]["payload"]["result"]
    assert decision["orders"] == _orders((12_500, -12_500, 6_250, -6_250))
    assert decision["pending"] is True
    _assert_account(decision["account"], ledger)

    # The settlement timestamp equals this execution timestamp. It is applied
    # before the newly requested positions exist, so both held quantities are 0.
    source = source_observation(1)
    funding = ledger.observe(source)
    first_lots = (12_500, -12_500, 6_250, -6_250)
    expected_fills = ledger.execute(source, first_lots)
    first = engine.execute(
        "entry-execution",
        market=_capture_market(tmp_path / "paper", "m1", source),
        at=source.at,
    )["event"]["payload"]["result"]
    assert [row["amount"] for row in funding] == [0.0, 0.0]
    assert all(row["quantity"] == "0" for row in first["funding"])
    _assert_fills(first["fills"], expected_fills)
    assert [row["filled_lots"] for row in first["fills"]] == [
        12_000,
        -12_000,
        6_000,
        -6_000,
    ]
    assert [row["unfilled_lots"] for row in first["fills"]] == [500, -500, 250, -250]
    for actual_price, expected_price in zip(
        (row["price"] for row in first["fills"]),
        (
            Decimal("101.300625"),
            Decimal("100.04995"),
            Decimal("202.60125"),
            Decimal("199.99995"),
        ),
        strict=True,
    ):
        _assert_close(actual_price, expected_price)
    for actual_fee, expected_fee in zip(
        (row["fee"] for row in first["fills"]),
        (
            Decimal("1.2156075"),
            Decimal("0.6002997"),
            Decimal("1.2156075"),
            Decimal("0.59999985"),
        ),
        strict=True,
    ):
        _assert_close(actual_fee, expected_fee)
    _assert_close(first["account"]["cash"], Decimal("9965.75258545"))
    assert first["pending"] is False
    assert first["terminal_flat"] is False
    _assert_account(first["account"], ledger)

    source = source_observation(2)
    ledger.observe(source)
    retry_lots = ledger.lot_delta(TARGET)
    assert retry_lots == (500, -500, 250, -250)
    retry_decision = engine.decide(
        "retry-decision",
        market=_capture_market(tmp_path / "paper", "m2", source),
        rules=rule,
        at=source.at + timedelta(seconds=0.1),
    )["event"]["payload"]["result"]
    assert retry_decision["orders"] == _orders(retry_lots)
    assert retry_decision["pending"] is True
    _assert_account(retry_decision["account"], ledger)

    # Both funding events settle at the same instant as the retry execution.
    # Only the prior partial holdings (12 BTC / 6 ETH) earn/pay funding;
    # the additional 0.5 / 0.25 lots are filled after settlement.
    source = source_observation(3)
    funding = ledger.observe(source)
    expected_fills = ledger.execute(source, retry_lots)
    second = engine.execute(
        "retry-execution",
        market=_capture_market(tmp_path / "paper", "m3", source),
        at=source.at,
    )["event"]["payload"]["result"]
    actual_funding = second["funding"]
    assert [row["symbol"] for row in actual_funding] == ["BTCUSDT", "ETHUSDT"]
    assert [row["quantity"] for row in actual_funding] == ["-12", "-6"]
    assert actual_funding[0]["amount"] > 0 > actual_funding[1]["amount"]
    assert [row["amount"] for row in actual_funding] == [12.36, -24.12]
    assert [row["amount"] for row in funding] == pytest.approx(
        [actual_funding[0]["amount"], actual_funding[1]["amount"]],
        rel=2e-12,
        abs=2e-9,
    )
    _assert_fills(second["fills"], expected_fills)
    assert all(row["unfilled_lots"] == 0 for row in second["fills"])
    assert second["pending"] is False
    assert tuple(ledger.quantities) == TARGET
    _assert_account(second["account"], ledger)

    # Reaching close requests actual zero targets; it does not erase positions.
    source = source_observation(4)
    ledger.observe(source)
    close_lots = ledger.lot_delta((Decimal(0),) * 4)
    close_decision = engine.decide(
        "terminal-decision",
        market=_capture_market(tmp_path / "paper", "m4", source),
        rules=rule,
        at=source.at + timedelta(seconds=0.1),
    )["event"]["payload"]["result"]
    assert close_lots == (-12_500, 12_500, -6_250, 6_250)
    assert close_decision["stop_reason"] == "terminal_close"
    assert close_decision["orders"] == _orders(close_lots)
    assert close_decision["terminal_flat"] is False
    assert close_decision["pending"] is True
    _assert_account(close_decision["account"], ledger)

    source = source_observation(5, deep=True)
    ledger.observe(source)
    expected_fills = ledger.execute(source, close_lots)
    closed = engine.execute(
        "terminal-execution",
        market=_capture_market(tmp_path / "paper", "m5", source),
        at=source.at,
    )["event"]["payload"]["result"]
    _assert_fills(closed["fills"], expected_fills)
    assert all(row["unfilled_lots"] == 0 for row in closed["fills"])
    assert closed["pending"] is False
    assert closed["terminal_flat"] is True
    _assert_account(closed["account"], ledger)
    assert closed["account"]["fill_count"] == 12


def test_asymmetric_partial_uses_actual_positions_for_funding_and_recovery_close(
    tmp_path,
):
    root = tmp_path / "paper"
    settings = PaperSettings(
        start_at=BASE - timedelta(seconds=1),
        close_at=BASE + timedelta(seconds=10),
        initial_capital=float(CAPITAL),
        spot_fee_bps=10,
        perpetual_fee_bps=5,
        adverse_bps=5,
        depth_fraction=0.1,
    )
    protocol = PaperEngine.initialize(
        root, settings=settings, study={"purpose": "synthetic asymmetric ledger"}
    )
    engine = PaperEngine(root, expected_protocol_sha256=protocol)
    rule = _capture_rules(root)
    ledger = PaperLedger(initial_capital=CAPITAL)
    requested = (12_500, -12_500, 6_250, -6_250)

    source = source_observation(0)
    ledger.observe(source)
    decision = engine.decide(
        "entry-decision",
        market=_capture_market(root, "m0", source),
        rules=rule,
        at=source.at + timedelta(seconds=0.1),
    )["event"]["payload"]["result"]
    assert decision["orders"] == _orders(requested)

    source = source_observation(1, asymmetric=True)
    ledger.observe(source)
    expected_fills = ledger.execute(source, requested)
    partial = engine.execute(
        "entry-execution",
        market=_capture_market(root, "m1", source),
        at=source.at,
    )["event"]["payload"]["result"]
    _assert_fills(partial["fills"], expected_fills)
    assert [row["filled_lots"] for row in partial["fills"]] == [
        12_000,
        -8_000,
        6_000,
        -6_000,
    ]
    assert partial["stop_reason"] == "unmatched_hedge"
    assert partial["pending"] is False
    assert partial["account"]["exact_quantities"] == ["12", "-8", "6", "-6"]
    _assert_account(partial["account"], ledger)

    source = source_observation(3, asymmetric=True)
    funding = ledger.observe(source)
    close_lots = ledger.lot_delta((Decimal(0),) * 4)
    decision = engine.decide(
        "recovery-decision",
        market=_capture_market(root, "m3", source),
        rules=rule,
        at=source.at + timedelta(seconds=0.1),
    )["event"]["payload"]["result"]
    assert decision["stop_reason"] == "unmatched_hedge"
    assert decision["orders"] == _orders(close_lots)
    assert close_lots == (-12_000, 8_000, -6_000, 6_000)
    assert [row["quantity"] for row in decision["funding"]] == ["-8", "-6"]
    assert [row["amount"] for row in decision["funding"]] == [8.24, -24.12]
    assert [row["amount"] for row in funding] == [8.24, -24.12]
    _assert_account(decision["account"], ledger)

    source = source_observation(4, deep=True)
    ledger.observe(source)
    expected_fills = ledger.execute(source, close_lots)
    recovered = engine.execute(
        "recovery-execution",
        market=_capture_market(root, "m4", source),
        at=source.at,
    )["event"]["payload"]["result"]
    _assert_fills(recovered["fills"], expected_fills)
    assert recovered["pending"] is False
    assert recovered["terminal_flat"] is True
    _assert_account(recovered["account"], ledger)
