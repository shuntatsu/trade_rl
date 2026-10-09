from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import pytest

from trade_rl.integrations.binance import (
    BinanceMarket,
    BinancePublicTransport,
    BinanceTransportError,
    BinanceTransportMode,
)


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def _funding_archive(payload: str) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("funding.csv", payload)
    return archive.getvalue()


def test_legacy_rest_funding_rates_keep_pair_shape_without_a_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    settlement_time = _ms(datetime(2026, 6, 1, tzinfo=UTC))
    monkeypatch.setattr(
        transport,
        "_request_json",
        lambda _url: [{"fundingTime": settlement_time, "fundingRate": "0.01"}],
    )

    rates, source = transport.load_funding_rates(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        start_ms=settlement_time,
        end_ms=settlement_time + 1,
        mode=BinanceTransportMode.REST,
    )

    assert rates == [(settlement_time, 0.01)]
    assert source == "rest"


def test_rest_funding_events_preserve_exchange_settlement_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    settlement_time = _ms(datetime(2026, 6, 1, tzinfo=UTC))
    monkeypatch.setattr(
        transport,
        "_request_json",
        lambda _url: [
            {
                "fundingTime": settlement_time,
                "fundingRate": "0.01",
                "markPrice": "95.0",
            }
        ],
    )

    events, source = transport.load_funding_events(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        start_ms=settlement_time,
        end_ms=settlement_time + 1,
        mode=BinanceTransportMode.REST,
    )

    assert events == [(settlement_time, 0.01, 95.0)]
    assert source == "rest"


def test_rest_event_api_fails_closed_when_settlement_mark_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    settlement_time = _ms(datetime(2026, 6, 1, tzinfo=UTC))
    monkeypatch.setattr(
        transport,
        "_request_json",
        lambda _url: [{"fundingTime": settlement_time, "fundingRate": "0.01"}],
    )

    with pytest.raises(BinanceTransportError, match="settlement mark price"):
        transport.load_funding_events(
            market=BinanceMarket.USDS_M,
            symbol="BTCUSDT",
            start_ms=settlement_time,
            end_ms=settlement_time + 1,
            mode=BinanceTransportMode.REST,
        )


def test_vision_missing_mark_uses_exact_time_without_overwriting_archive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settlement_time = _ms(datetime(2026, 6, 1, 8, tzinfo=UTC))
    already_marked_time = _ms(datetime(2026, 6, 1, 16, tzinfo=UTC))
    payload = _funding_archive(
        "calc_time,last_funding_rate,markPrice\n"
        f"{settlement_time},0.01,\n{already_marked_time},-0.02,123.0\n"
    )
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    monkeypatch.setattr(transport, "_request_bytes", lambda _url: payload)
    monkeypatch.setattr(
        transport,
        "_request_json",
        lambda _url: [
            {
                "fundingTime": settlement_time,
                "fundingRate": "0.99",
                "markPrice": "95.0",
            },
            {
                "fundingTime": already_marked_time,
                "fundingRate": "0.88",
                "markPrice": "777.0",
            },
        ],
    )

    events, source = transport.load_funding_events(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        start_ms=_ms(datetime(2026, 6, 1, tzinfo=UTC)),
        end_ms=_ms(datetime(2026, 7, 1, tzinfo=UTC)),
        mode=BinanceTransportMode.VISION,
    )

    assert events == [
        (settlement_time, 0.01, 95.0),
        (already_marked_time, -0.02, 123.0),
    ]
    assert source == "vision+rest-marks"


@pytest.mark.parametrize("timestamp_offset_ms", (-1, 1))
def test_vision_missing_mark_rejects_rest_mark_at_a_different_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    timestamp_offset_ms: int,
) -> None:
    settlement_time = _ms(datetime(2026, 6, 1, 8, tzinfo=UTC))
    payload = _funding_archive(f"calc_time,last_funding_rate\n{settlement_time},0.01\n")
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    monkeypatch.setattr(transport, "_request_bytes", lambda _url: payload)
    monkeypatch.setattr(
        transport,
        "_request_json",
        lambda _url: [
            {
                "fundingTime": settlement_time + timestamp_offset_ms,
                "fundingRate": "0.02",
                "markPrice": "95.0",
            }
        ],
    )

    with pytest.raises(
        BinanceTransportError, match="settlement mark price is unavailable"
    ):
        transport.load_funding_events(
            market=BinanceMarket.USDS_M,
            symbol="BTCUSDT",
            start_ms=_ms(datetime(2026, 6, 1, tzinfo=UTC)),
            end_ms=_ms(datetime(2026, 7, 1, tzinfo=UTC)),
            mode=BinanceTransportMode.VISION,
        )


@pytest.mark.parametrize("mark_source", ("archive", "rest_completion"))
@pytest.mark.parametrize("invalid_mark", ("0", "nan", "inf"))
def test_vision_funding_refuses_invalid_original_or_completed_marks(
    monkeypatch: pytest.MonkeyPatch,
    mark_source: str,
    invalid_mark: str,
) -> None:
    settlement_time = _ms(datetime(2026, 6, 1, 8, tzinfo=UTC))
    archive_mark = invalid_mark if mark_source == "archive" else ""
    payload = _funding_archive(
        f"calc_time,last_funding_rate,markPrice\n{settlement_time},0.01,{archive_mark}\n"
    )
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    monkeypatch.setattr(transport, "_request_bytes", lambda _url: payload)
    monkeypatch.setattr(
        transport,
        "_request_json",
        lambda _url: [
            {
                "fundingTime": settlement_time,
                "fundingRate": "0.99",
                "markPrice": invalid_mark,
            }
        ],
    )

    # Vision parsing and REST wrapping have different historical exception types.
    with pytest.raises(
        (BinanceTransportError, ValueError), match="settlement mark price"
    ):
        transport.load_funding_events(
            market=BinanceMarket.USDS_M,
            symbol="BTCUSDT",
            start_ms=_ms(datetime(2026, 6, 1, tzinfo=UTC)),
            end_ms=_ms(datetime(2026, 7, 1, tzinfo=UTC)),
            mode=BinanceTransportMode.VISION,
        )


@pytest.mark.parametrize("mark_header", ("markPrice", "mark_price"))
def test_vision_funding_csv_parser_preserves_settlement_mark(
    monkeypatch: pytest.MonkeyPatch,
    mark_header: str,
) -> None:
    settlement_time = _ms(datetime(2026, 6, 1, 8, tzinfo=UTC))
    payload = _funding_archive(
        f"calc_time,last_funding_rate,{mark_header}\n{settlement_time},0.01,95.0\n"
    )
    transport = BinancePublicTransport(max_attempts=1, retry_backoff_seconds=0.0)
    monkeypatch.setattr(transport, "_request_bytes", lambda _url: payload)

    events, source = transport.load_funding_events(
        market=BinanceMarket.USDS_M,
        symbol="BTCUSDT",
        start_ms=_ms(datetime(2026, 6, 1, tzinfo=UTC)),
        end_ms=_ms(datetime(2026, 7, 1, tzinfo=UTC)),
        mode=BinanceTransportMode.VISION,
    )

    assert events == [(settlement_time, 0.01, 95.0)]
    assert source == "vision"
