"""Failures keep the type a collector needs to classify them (no network).

The transport chains its final error to the underlying HTTP or socket error, and a
freshness rejection of one capture has its own ``ValueError`` subtype.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from datetime import timedelta

import pytest

from tests.integrations.test_binance_forward import NOW, PublicFeed
from trade_rl.evaluation.paper.collection_policy import classify_failure
from trade_rl.integrations.binance.forward import (
    ForwardFreshnessError,
    capture_forward_snapshot,
)
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import BinanceTransportError


def _fail_with(monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    def urlopen(*args, **kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)


@pytest.mark.parametrize(
    ("code", "expected"),
    [(429, "D"), (451, "D"), (503, "T"), (404, "I"), (400, "I")],
)
def test_the_transport_error_is_chained_to_the_http_status(
    monkeypatch: pytest.MonkeyPatch, code: int, expected: str
) -> None:
    http = urllib.error.HTTPError("https://example.test", code, "err", {}, None)  # type: ignore[arg-type]
    _fail_with(monkeypatch, http)
    transport = BinancePublicTransport(timeout_seconds=1, max_attempts=1)

    with pytest.raises(BinanceTransportError) as raised:
        transport._request_bytes("https://example.test/x", max_response_bytes=1024)

    assert raised.value.__cause__ is http
    assert classify_failure(raised.value) == expected


@pytest.mark.parametrize(
    "error", [TimeoutError("timed out"), urllib.error.URLError("unreachable")]
)
def test_a_timeout_or_connection_error_is_chained_and_transient(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    _fail_with(monkeypatch, error)
    transport = BinancePublicTransport(timeout_seconds=1, max_attempts=1)

    with pytest.raises(BinanceTransportError) as raised:
        transport._request_bytes("https://example.test/x", max_response_bytes=1024)

    assert raised.value.__cause__ is error
    assert classify_failure(raised.value) == "T"


class SlowFeed(PublicFeed):
    def __init__(self, clock_box: list) -> None:
        super().__init__()
        self.clock_box = clock_box

    def _request_bytes(self, url, *, max_response_bytes=None):
        self.clock_box[0] += timedelta(
            seconds=6
        )  # beyond the five-second request bound
        return super()._request_bytes(url, max_response_bytes=max_response_bytes)


def test_an_overlong_request_is_a_freshness_error_and_still_a_value_error(tmp_path):
    now = [NOW]

    def clock():
        return now[0]

    with pytest.raises(ForwardFreshnessError) as raised:
        capture_forward_snapshot(
            tmp_path / "capture",
            transport=SlowFeed(now),
            clock=clock,
            monotonic=lambda: (now[0] - NOW).total_seconds(),
        )

    assert isinstance(raised.value, ValueError)
    assert classify_failure(raised.value) == "T"
    assert (tmp_path / "capture" / "failure.json").is_file()
