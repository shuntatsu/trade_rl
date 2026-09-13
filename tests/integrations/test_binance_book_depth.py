from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from trade_rl.integrations.binance import (
    BinanceBookDepthSeries,
    BinanceMarket,
    BinanceTransportError,
    parse_vision_book_depth_archive,
    plan_vision_book_depth_urls,
    validate_book_depth_reference_alignment,
    vision_book_depth_url,
)

_EXPECTED_BANDS = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)


def _zip_csv(
    rows: list[str], *, name: str = "BTCUSDT-bookDepth-2025-05-19.csv"
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, "\n".join(rows) + "\n")
    return buffer.getvalue()


def _snapshot(timestamp: str, *, price: float = 100.0) -> list[str]:
    rows: list[str] = []
    for band in _EXPECTED_BANDS:
        distance = abs(band)
        depth = float(distance * 10)
        average_price = price * (1.0 + band * 0.005)
        notional = depth * average_price
        rows.append(f"{timestamp},{band},{depth:.8f},{notional:.8f}")
    return rows


def _payload(*snapshots: list[str]) -> bytes:
    rows = ["timestamp,percentage,depth,notional"]
    for snapshot in snapshots:
        rows.extend(snapshot)
    return _zip_csv(rows)


def _drop_last_band(rows: list[str]) -> list[str]:
    return rows[:-1]


def _duplicate_last_band(rows: list[str]) -> list[str]:
    return [*rows, rows[-1]]


def _break_cumulative_depth(rows: list[str]) -> list[str]:
    changed = list(rows)
    changed[1] = changed[1].replace(",40.00000000,", ",1.00000000,")
    return changed


def _make_depth_negative(rows: list[str]) -> list[str]:
    changed = list(rows)
    changed[0] = changed[0].replace(
        ",50.00000000,",
        ",-1.00000000,",
    )
    return changed


def test_book_depth_url_and_plan_are_usds_m_daily_only() -> None:
    day = datetime(2025, 5, 19, tzinfo=UTC)

    assert vision_book_depth_url("usds-m", "BTCUSDT", day) == (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2025-05-19.zip"
    )
    assert plan_vision_book_depth_urls(
        "usds-m",
        "BTCUSDT",
        day + timedelta(hours=8),
        day + timedelta(days=2, hours=1),
    ) == (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2025-05-19.zip",
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2025-05-20.zip",
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2025-05-21.zip",
    )

    for market in (BinanceMarket.SPOT, BinanceMarket.COIN_M):
        with pytest.raises(ValueError, match="USD-M"):
            vision_book_depth_url(market, "BTCUSDT", day)


def test_book_depth_parser_returns_immutable_deterministic_evidence() -> None:
    payload = _payload(
        _snapshot("2025-05-19 11:07:31", price=100.0),
        _snapshot("2025-05-19 11:08:01", price=101.0),
    )
    source = (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        "BTCUSDT/BTCUSDT-bookDepth-2025-05-19.zip"
    )

    series = parse_vision_book_depth_archive(payload, source=source)

    assert isinstance(series, BinanceBookDepthSeries)
    assert series.percentage_bands == _EXPECTED_BANDS
    np.testing.assert_array_equal(
        series.timestamps,
        np.array(
            ["2025-05-19T11:07:31", "2025-05-19T11:08:01"],
            dtype="datetime64[ns]",
        ),
    )
    np.testing.assert_array_equal(series.available_at, series.timestamps)
    assert series.depth.shape == (2, 10)
    assert series.notional.shape == (2, 10)
    assert series.implied_average_price.shape == (2, 10)
    assert series.source_uri == source
    assert series.raw_payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert series.raw_payload_size_bytes == len(payload)
    assert series.timestamps.flags.writeable is False
    assert series.depth.flags.writeable is False
    assert series.notional.flags.writeable is False
    assert series.implied_average_price.flags.writeable is False
    with pytest.raises(ValueError):
        series.depth[0, 0] = 1.0
    with pytest.raises(FrozenInstanceError):
        series.source_uri = "changed"  # type: ignore[misc]


def test_book_depth_series_rejects_nat_time_evidence() -> None:
    series = parse_vision_book_depth_archive(
        _payload(_snapshot("2025-05-19 11:07:31")),
        source="fixture",
    )
    nat = np.array(["NaT"], dtype="datetime64[ns]")

    with pytest.raises(ValueError, match="timestamps.*NaT"):
        replace(series, timestamps=nat)
    with pytest.raises(ValueError, match="available_at.*NaT"):
        replace(series, available_at=nat)


def test_book_depth_parser_requires_exact_header() -> None:
    payload = _zip_csv(
        [
            "timestamp,percentage,notional,depth",
            *_snapshot("2025-05-19 11:07:31"),
        ]
    )

    with pytest.raises(BinanceTransportError, match="header"):
        parse_vision_book_depth_archive(payload, source="fixture")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_drop_last_band, "bands"),
        (_duplicate_last_band, "duplicate"),
        (_break_cumulative_depth, "nondecreasing"),
        (_make_depth_negative, "non-negative"),
    ],
)
def test_book_depth_parser_rejects_structurally_invalid_snapshots(
    mutate: Callable[[list[str]], list[str]],
    match: str,
) -> None:
    rows = _snapshot("2025-05-19 11:07:31")
    payload = _payload(mutate(rows))

    with pytest.raises(BinanceTransportError, match=match):
        parse_vision_book_depth_archive(payload, source="fixture")


def test_book_depth_parser_rejects_nonfinite_values() -> None:
    rows = _snapshot("2025-05-19 11:07:31")
    rows[0] = rows[0].rsplit(",", 1)[0] + ",nan"

    with pytest.raises(BinanceTransportError, match="finite"):
        parse_vision_book_depth_archive(_payload(rows), source="fixture")


def test_book_depth_parser_rejects_out_of_order_snapshot_timestamps() -> None:
    payload = _payload(
        _snapshot("2025-05-19 11:08:01"),
        _snapshot("2025-05-19 11:07:31"),
    )

    with pytest.raises(BinanceTransportError, match="strictly increasing"):
        parse_vision_book_depth_archive(payload, source="fixture")


def test_book_depth_parser_rejects_bid_ask_implied_price_crossing() -> None:
    rows = _snapshot("2025-05-19 11:07:31")
    # Make the closest ask average price lower than the closest bid average price
    # while preserving cumulative depth/notional monotonicity on each side.
    ask_index = _EXPECTED_BANDS.index(1)
    rows[ask_index] = "2025-05-19 11:07:31,1,10.00000000,900.00000000"
    for band in (2, 3, 4, 5):
        index = _EXPECTED_BANDS.index(band)
        depth = float(band * 10)
        rows[index] = f"2025-05-19 11:07:31,{band},{depth:.8f},{depth * 90.0:.8f}"

    with pytest.raises(BinanceTransportError, match="bid.*ask|side"):
        parse_vision_book_depth_archive(_payload(rows), source="fixture")


def test_reference_alignment_rejects_known_style_price_misalignment() -> None:
    # Structurally valid snapshot modelled after the public-data anomaly pattern:
    # its implied prices are around 82k-85k while the contemporaneous reference is 103k.
    rows = [
        "2025-05-19 11:07:31,-5,7885.27500000,647368074.36920000",
        "2025-05-19 11:07:31,-4,7178.45300000,590768470.27090000",
        "2025-05-19 11:07:31,-3,5443.14900000,450287742.99870000",
        "2025-05-19 11:07:31,-2,3989.69900000,331452387.75030000",
        "2025-05-19 11:07:31,-1,2294.32400000,191369991.21780000",
        "2025-05-19 11:07:31,1,2097.41000000,176607765.20950000",
        "2025-05-19 11:07:31,2,3485.14100000,294629128.47080000",
        "2025-05-19 11:07:31,3,4813.86500000,408742768.20550000",
        "2025-05-19 11:07:31,4,5699.71200000,485628858.29740000",
        "2025-05-19 11:07:31,5,6491.51500000,555012399.28440000",
    ]
    series = parse_vision_book_depth_archive(_payload(rows), source="fixture")

    with pytest.raises(BinanceTransportError, match="reference-price alignment"):
        validate_book_depth_reference_alignment(
            series,
            np.array([103_000.0]),
            reference_available_at=series.timestamps,
            max_relative_deviation_rate=0.10,
        )


def test_reference_alignment_accepts_causal_reference_within_explicit_bound() -> None:
    series = parse_vision_book_depth_archive(
        _payload(_snapshot("2025-05-19 11:07:31", price=100.0)),
        source="fixture",
    )

    validate_book_depth_reference_alignment(
        series,
        np.array([100.0]),
        reference_available_at=series.timestamps,
        max_relative_deviation_rate=0.10,
    )
