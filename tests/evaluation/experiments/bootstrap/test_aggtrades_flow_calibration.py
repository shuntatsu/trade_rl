from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from trade_rl.evaluation.experiments.bootstrap.agg_flow_calibration import (
    AggTradesArchiveCalibration,
    build_aggtrades_flow_calibration_result,
    evaluate_aggtrades_archive,
)
from trade_rl.evaluation.experiments.bootstrap.agg_flow_capacity import (
    canonical_m2_aggtrades_flow_capacity_protocol,
)
from trade_rl.integrations.binance.agg_trades import BinanceAggTradesSeries


def _series(
    *,
    timestamps_ms: list[int],
    prices: list[float],
    quantities: list[float],
    buyer_is_maker: list[bool],
    source: str,
) -> BinanceAggTradesSeries:
    size = len(timestamps_ms)
    return BinanceAggTradesSeries(
        aggregate_trade_ids=np.arange(size, dtype=np.int64),
        prices=np.asarray(prices, dtype=np.float64),
        quantities=np.asarray(quantities, dtype=np.float64),
        first_trade_ids=np.arange(size, dtype=np.int64),
        last_trade_ids=np.arange(size, dtype=np.int64),
        timestamps=np.asarray(timestamps_ms, dtype="datetime64[ms]").astype(
            "datetime64[ns]"
        ),
        buyer_is_maker=np.asarray(buyer_is_maker, dtype=np.bool_),
        source_uri=source,
        raw_payload_sha256="a" * 64,
        raw_payload_size_bytes=123,
        header_present=False,
    )


def _planned_url(symbol: str, day: datetime) -> str:
    return (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
    )


def _accepted_record(
    symbol: str,
    day: datetime,
    *,
    fractions: tuple[float, ...] = (0.02,) * 24,
) -> AggTradesArchiveCalibration:
    return AggTradesArchiveCalibration(
        symbol=symbol,
        day=day,
        url=_planned_url(symbol, day),
        accepted=True,
        rejection_reason=None,
        raw_payload_sha256="b" * 64,
        raw_payload_size_bytes=100,
        checksum_text=("b" * 64) + f"  {symbol}-aggTrades-{day:%Y-%m-%d}.zip",
        checksum_verified=True,
        header_present=False,
        row_count=10,
        first_timestamp=day,
        last_timestamp=day.replace(hour=23, minute=59),
        hourly_fractions=fractions,
    )


def _rejected_record(symbol: str, day: datetime) -> AggTradesArchiveCalibration:
    return AggTradesArchiveCalibration(
        symbol=symbol,
        day=day,
        url=_planned_url(symbol, day),
        accepted=False,
        rejection_reason="archive_missing",
        raw_payload_sha256=None,
        raw_payload_size_bytes=None,
        checksum_text=None,
        checksum_verified=False,
        header_present=None,
        row_count=None,
        first_timestamp=None,
        last_timestamp=None,
        hourly_fractions=(),
    )


def test_archive_flow_uses_fixed_five_second_bins_and_both_taker_sides() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()
    day = datetime(2021, 1, 1, tzinfo=UTC)
    base_ms = int(day.timestamp() * 1_000)
    url = _planned_url("BTCUSDT", day)
    series = _series(
        timestamps_ms=[
            base_ms,
            base_ms + 4_999,
            base_ms + 5_000,
            base_ms + 1_000,
            base_ms + 3_000,
            base_ms + 5_001,
        ],
        prices=[10.0] * 6,
        quantities=[10.0, 5.0, 12.0, 8.0, 2.0, 9.0],
        buyer_is_maker=[False, False, False, True, True, True],
        source=url,
    )

    record = evaluate_aggtrades_archive(
        protocol,
        symbol="BTCUSDT",
        day=day,
        url=url,
        series=series,
        checksum_text=("a" * 64) + "  archive.zip",
        checksum_verified=True,
    )

    assert record.accepted is True
    assert record.row_count == 6
    assert record.hourly_fractions == pytest.approx((100.0 / 460.0,))


def test_archive_flow_rejects_wrong_day_and_post_evaluation_timestamp() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()
    day = datetime(2022, 12, 1, tzinfo=UTC)
    url = _planned_url("BTCUSDT", day)
    post_evaluation_ms = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    series = _series(
        timestamps_ms=[post_evaluation_ms],
        prices=[10.0],
        quantities=[1.0],
        buyer_is_maker=[False],
        source=url,
    )

    with pytest.raises(ValueError, match="requested UTC day|evaluation boundary"):
        evaluate_aggtrades_archive(
            protocol,
            symbol="BTCUSDT",
            day=day,
            url=url,
            series=series,
            checksum_text=("a" * 64) + "  archive.zip",
            checksum_verified=True,
        )


def test_result_uses_exact_registered_rank_and_cap_without_interpolation() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()
    records: list[AggTradesArchiveCalibration] = []
    for symbol in protocol.symbols:
        for day in protocol.planned_days:
            fractions = tuple((index + 1) / 10_000 for index in range(24))
            records.append(_accepted_record(symbol, day, fractions=fractions))

    result = build_aggtrades_flow_calibration_result(protocol, records)

    assert result.status == "PASS"
    for summary in result.symbol_summaries:
        assert summary.accepted_days == 24
        assert summary.accepted_days_2021 == 12
        assert summary.accepted_days_2022 == 12
        assert summary.valid_hours == 576
        assert summary.rank == protocol.lower_tail_rank(576) == 57
        # 24 values repeated once per month; indices 0..23 each occur 24 times.
        assert summary.q10 == pytest.approx(0.0003)
        assert summary.symbol_cap == pytest.approx(0.000075)


def test_registered_coverage_failure_invalidates_without_fallback() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()
    records: list[AggTradesArchiveCalibration] = []
    for symbol in protocol.symbols:
        for index, day in enumerate(protocol.planned_days):
            if symbol == "BTCUSDT" and index < 5:
                records.append(_rejected_record(symbol, day))
            else:
                records.append(_accepted_record(symbol, day))

    result = build_aggtrades_flow_calibration_result(protocol, records)

    assert result.status == "INVALID"
    btc = next(item for item in result.symbol_summaries if item.symbol == "BTCUSDT")
    assert btc.accepted_days == 19
    assert btc.q10 is None
    assert btc.symbol_cap is None
    assert any("BTCUSDT" in failure and "accepted_days" in failure for failure in result.failures)


def test_result_requires_exact_planned_roster_and_is_content_addressed() -> None:
    protocol = canonical_m2_aggtrades_flow_capacity_protocol()
    records = [
        _accepted_record(symbol, day)
        for symbol in protocol.symbols
        for day in protocol.planned_days
    ]
    result = build_aggtrades_flow_calibration_result(protocol, records)
    payload = result.to_payload()

    assert payload["result_digest"] == result.digest
    assert len(result.digest) == 64
    assert payload["protocol_digest"] == protocol.digest
    assert payload["planned_urls"] == list(protocol.planned_urls)
    forbidden = {
        "returns",
        "pnl",
        "strategy",
        "candidate_result",
        "experiment_result",
    }
    assert forbidden.isdisjoint(payload)

    with pytest.raises(ValueError, match="planned roster"):
        build_aggtrades_flow_calibration_result(protocol, records[:-1])

    with pytest.raises(ValueError, match="planned roster|duplicate"):
        build_aggtrades_flow_calibration_result(protocol, records + [records[0]])
