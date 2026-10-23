from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from trade_rl.integrations.binance import forward, forward_rules
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import BinanceTransportError

_NOW = datetime(2026, 9, 18, tzinfo=UTC)
_MS = int(_NOW.timestamp() * 1000)


class _Response:
    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.payload = payload
        self.headers = {} if headers is None else headers
        self.chunk_size = chunk_size
        self.offset = 0
        self.read_sizes: list[int] = []

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            end = len(self.payload)
        else:
            amount = size if self.chunk_size is None else min(size, self.chunk_size)
            end = min(len(self.payload), self.offset + amount)
        result = self.payload[self.offset : end]
        self.offset = end
        return result


def _patch_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    response_factory: Callable[[urllib.request.Request], _Response],
) -> list[str]:
    timeouts: list[float] = []

    def urlopen(request: urllib.request.Request, *, timeout: float) -> _Response:
        timeouts.append(timeout)
        return response_factory(request)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return timeouts


def test_bounded_transport_accepts_exact_limit_after_short_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(b"12345678", chunk_size=3)
    timeouts = _patch_urlopen(monkeypatch, lambda _request: response)

    result = BinancePublicTransport(max_attempts=1)._request_bytes(
        "https://fapi.binance.com/fapi/v1/time", max_response_bytes=8
    )

    assert result == b"12345678"
    assert response.read_sizes == [9, 6, 3, 1]
    assert all(0 < size <= 9 for size in response.read_sizes)
    assert timeouts == [30.0]


def test_bounded_transport_streams_large_exact_limit_in_fixed_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 70 * 1024
    payload = b"x" * limit
    response = _Response(payload)
    _patch_urlopen(monkeypatch, lambda _request: response)

    result = BinancePublicTransport(max_attempts=1)._request_bytes(
        "https://fapi.binance.com/fapi/v1/time", max_response_bytes=limit
    )

    assert result == payload
    assert response.read_sizes == [64 * 1024, 6 * 1024 + 1, 1]
    assert all(0 < size <= 64 * 1024 for size in response.read_sizes)


@pytest.mark.parametrize(
    "headers",
    [{"Content-Length": "1"}, {}],
    ids=["underreported-length", "missing-length"],
)
def test_bounded_transport_rejects_actual_oversize_with_untrusted_length(
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
) -> None:
    response = _Response(b"0123456789", headers=headers, chunk_size=2)
    timeouts = _patch_urlopen(monkeypatch, lambda _request: response)

    with pytest.raises(BinanceTransportError, match="maximum of 8 bytes"):
        BinancePublicTransport(max_attempts=3, retry_backoff_seconds=0)._request_bytes(
            "https://fapi.binance.com/fapi/v1/time", max_response_bytes=8
        )

    assert response.read_sizes == [9, 7, 5, 3, 1]
    assert all(0 < size <= 9 for size in response.read_sizes)
    assert timeouts == [30.0]


def test_bounded_transport_rejects_declared_oversize_without_reading_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _Response(b"12345678", headers={"Content-Length": "9"}, chunk_size=2)
    _patch_urlopen(monkeypatch, lambda _request: response)

    with pytest.raises(BinanceTransportError, match="maximum of 8 bytes"):
        BinancePublicTransport(max_attempts=1)._request_bytes(
            "https://fapi.binance.com/fapi/v1/time", max_response_bytes=8
        )

    assert response.read_sizes == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, sys.maxsize])
def test_bounded_transport_requires_a_positive_integer_limit(
    monkeypatch: pytest.MonkeyPatch, limit: object
) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid limit reached the network"),
    )

    with pytest.raises(ValueError, match="max_response_bytes"):
        BinancePublicTransport()._request_bytes(
            "https://fapi.binance.com/fapi/v1/time",
            max_response_bytes=limit,  # type: ignore[arg-type]
        )


def test_forward_snapshot_oversize_writes_durable_failure_without_partial_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(forward, "_MAX_RESPONSE_BYTES", 8)
    responses: list[_Response] = []

    def response_factory(request: urllib.request.Request) -> _Response:
        assert request.full_url == "https://data-api.binance.vision/api/v3/time"
        response = _Response(b"123456789", headers={"Content-Length": "1"})
        responses.append(response)
        return response

    timeouts = _patch_urlopen(monkeypatch, response_factory)
    root = tmp_path / "capture"

    with pytest.raises(BinanceTransportError, match="maximum of 8 bytes"):
        forward.capture_forward_snapshot(
            root, clock=lambda: _NOW, monotonic=lambda: 1.0
        )

    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["eligible"] is False
    assert failure["error_type"] == "BinanceTransportError"
    assert "maximum of 8 bytes" in failure["error"]
    assert failure["completed_responses"] == []
    assert not (root / "spot_time.raw").exists()
    assert responses[0].read_sizes == [9]
    assert timeouts == [4.0]


def test_forward_snapshot_keeps_completed_raw_when_later_response_is_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(forward, "_MAX_RESPONSE_BYTES", 64)
    responses: list[_Response] = []

    def response_factory(_request: urllib.request.Request) -> _Response:
        if responses:
            payload = b"x" * 65
            headers = {"Content-Length": "1"}
        else:
            payload = json.dumps({"serverTime": _MS}).encode()
            headers = {}
        response = _Response(payload, headers=headers)
        responses.append(response)
        return response

    timeouts = _patch_urlopen(monkeypatch, response_factory)
    root = tmp_path / "capture"

    with pytest.raises(BinanceTransportError, match="maximum of 64 bytes"):
        forward.capture_forward_snapshot(
            root, clock=lambda: _NOW, monotonic=lambda: 1.0
        )

    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["eligible"] is False
    assert [record["label"] for record in failure["completed_responses"]] == [
        "spot_time"
    ]
    assert (root / "spot_time.raw").read_bytes() == json.dumps(
        {"serverTime": _MS}
    ).encode()
    assert not (root / "futures_time.raw").exists()
    assert responses[0].read_sizes == [65, 36]
    assert responses[1].read_sizes == [65]
    assert timeouts == [4.0, 4.0]


def test_forward_rules_oversize_writes_durable_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(forward_rules, "_MAX_RESPONSE_BYTES", 8)
    responses: list[_Response] = []

    def response_factory(_request: urllib.request.Request) -> _Response:
        response = _Response(b"123456789", headers={"Content-Length": "1"})
        responses.append(response)
        return response

    timeouts = _patch_urlopen(monkeypatch, response_factory)
    root = tmp_path / "rules"

    with pytest.raises(BinanceTransportError, match="maximum of 8 bytes"):
        forward_rules.capture_forward_rules(
            root, clock=lambda: _NOW, monotonic=lambda: 1.0
        )

    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["eligible"] is False
    assert failure["error_type"] == "BinanceTransportError"
    assert failure["completed_responses"] == []
    assert not (root / "spot_rules.raw").exists()
    assert responses[0].read_sizes == [9]
    assert timeouts == [4.0]


def test_forward_rules_keeps_completed_raw_when_later_response_is_oversize(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(forward_rules, "_MAX_RESPONSE_BYTES", 64)
    responses: list[_Response] = []

    def response_factory(_request: urllib.request.Request) -> _Response:
        payload = b"completed metadata" if not responses else b"x" * 65
        headers = {} if not responses else {"Content-Length": "1"}
        response = _Response(payload, headers=headers)
        responses.append(response)
        return response

    timeouts = _patch_urlopen(monkeypatch, response_factory)
    root = tmp_path / "rules"

    with pytest.raises(BinanceTransportError, match="maximum of 64 bytes"):
        forward_rules.capture_forward_rules(
            root, clock=lambda: _NOW, monotonic=lambda: 1.0
        )

    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["eligible"] is False
    assert [record["label"] for record in failure["completed_responses"]] == [
        "spot_rules"
    ]
    assert (root / "spot_rules.raw").read_bytes() == b"completed metadata"
    assert not (root / "perpetual_rules.raw").exists()
    assert responses[0].read_sizes == [65, 47]
    assert responses[1].read_sizes == [65]
    assert timeouts == [4.0, 4.0]


def test_forward_capture_preserves_429_without_reading_error_body_or_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class ErrorBody:
        def __init__(self) -> None:
            self.read_sizes: list[int] = []

        def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            return b"rate limit response body"

        def close(self) -> None:
            return None

    body = ErrorBody()
    attempts = 0

    def urlopen(_request: urllib.request.Request, *, timeout: float) -> None:
        nonlocal attempts
        attempts += 1
        assert timeout == 4.0
        raise urllib.error.HTTPError(
            "https://data-api.binance.vision/api/v3/time",
            429,
            "too many requests",
            {},
            body,
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    root = tmp_path / "capture"

    with pytest.raises(BinanceTransportError, match="HTTP Error 429"):
        forward.capture_forward_snapshot(
            root, clock=lambda: _NOW, monotonic=lambda: 1.0
        )

    failure = json.loads((root / "failure.json").read_text(encoding="utf-8"))
    assert failure["eligible"] is False
    assert failure["error_type"] == "BinanceTransportError"
    assert "HTTP Error 429" in failure["error"]
    assert failure["completed_responses"] == []
    assert attempts == 1
    assert body.read_sizes == []
