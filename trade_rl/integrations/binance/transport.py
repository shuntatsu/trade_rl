"""Bounded public Binance REST and Vision transport."""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.integrations.binance.cache import validate_cached_vision_payload
from trade_rl.integrations.binance.metadata import (
    BinanceExchangeInfoSnapshot,
    _mutable_json_object,
)
from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    BinanceTransportMode,
    _aware_utc,
    _finite_float,
    _market,
    _mode,
)
from trade_rl.integrations.binance.vision import (
    _VISION_ROOT,
    _csv_rows_from_zip,
    _interval_ms,
    _iter_months,
    _looks_like_header,
    _normalize_epoch_ms,
    plan_vision_kline_urls,
    vision_funding_url,
)

_REST_BASE = {
    "spot": "https://api.binance.com",
    "usds-m": "https://fapi.binance.com",
    "coin-m": "https://dapi.binance.com",
}

_REST_KLINES = {
    "spot": "/api/v3/klines",
    "usds-m": "/fapi/v1/klines",
    "coin-m": "/dapi/v1/klines",
}

_REST_EXCHANGE_INFO = {
    "spot": "/api/v3/exchangeInfo",
    "usds-m": "/fapi/v1/exchangeInfo",
    "coin-m": "/dapi/v1/exchangeInfo",
}

_REST_FUNDING = {
    "usds-m": "/fapi/v1/fundingRate",
    "coin-m": "/dapi/v1/fundingRate",
}

_USER_AGENT = "trade-rl/0.3 public-market-data"


class BinancePublicTransport:
    """Bounded public HTTP transport for REST and Binance Vision archives."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.25,
        cache_root: str | Path | None = None,
        allow_network: bool = True,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not math.isfinite(retry_backoff_seconds) or retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if not isinstance(allow_network, bool):
            raise ValueError("allow_network must be a boolean")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.allow_network = allow_network

    def _vision_cache_path(self, url: str) -> Path | None:
        if self.cache_root is None or not url.startswith(f"{_VISION_ROOT}/"):
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_root / digest[:2] / f"{digest}.bin"

    def _validated_cached_vision_payload(self, url: str, cache_path: Path) -> bytes:
        return validate_cached_vision_payload(url, cache_path)

    def _write_vision_cache(
        self,
        *,
        url: str,
        cache_path: Path,
        payload: bytes,
        etag: str | None,
        last_modified: str | None,
    ) -> None:
        evidence = {
            "acquired_at": datetime.now(UTC).isoformat(),
            "downloader": _USER_AGENT,
            "etag": etag,
            "last_modified": last_modified,
            "schema_version": "binance_vision_raw_cache_v1",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "url": url,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path = cache_path.with_suffix(".json")
        binary_temporary = cache_path.with_suffix(".bin.tmp")
        evidence_temporary = evidence_path.with_suffix(".json.tmp")
        binary_temporary.write_bytes(payload)
        evidence_temporary.write_text(
            json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        binary_temporary.replace(cache_path)
        evidence_temporary.replace(evidence_path)

    def _request_bytes(self, url: str) -> bytes:
        cache_path = self._vision_cache_path(url)
        if cache_path is not None and cache_path.is_file():
            return self._validated_cached_vision_payload(url, cache_path)
        if not self.allow_network:
            raise BinanceTransportError(
                f"network access is disabled for uncached source: {url}"
            )
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        last_error: BaseException | None = None
        for attempt in range(self.max_attempts):
            try:
                with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS endpoints
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    payload = response.read()
                    headers = getattr(response, "headers", {})
                    header_get = getattr(headers, "get", lambda _name: None)
                    etag = header_get("ETag")
                    last_modified = header_get("Last-Modified")
                if not payload:
                    raise BinanceTransportError(
                        f"Binance returned an empty response for {url}"
                    )
                if cache_path is not None:
                    self._write_vision_cache(
                        url=url,
                        cache_path=cache_path,
                        payload=payload,
                        etag=None if etag is None else str(etag),
                        last_modified=(
                            None if last_modified is None else str(last_modified)
                        ),
                    )
                return payload
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code == 404:
                    break
                if error.code < 500 and error.code not in {418, 429, 451}:
                    break
            except (TimeoutError, urllib.error.URLError) as error:
                last_error = error
            if attempt + 1 < self.max_attempts and self.retry_backoff_seconds > 0.0:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        detail = "unknown transport error" if last_error is None else str(last_error)
        raise BinanceTransportError(f"Binance request failed for {url}: {detail}")

    def _request_json(self, url: str) -> object:
        payload = self._request_bytes(url)
        try:
            return json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BinanceTransportError(
                f"Binance returned invalid JSON for {url}"
            ) from error

    @staticmethod
    def _query_url(base: str, path: str, values: Mapping[str, object]) -> str:
        return f"{base}{path}?{urllib.parse.urlencode(values)}"

    def _load_rest_klines(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[object]]:
        rows: list[list[object]] = []
        cursor = start_ms
        step = _interval_ms(interval)
        while cursor < end_ms:
            url = self._query_url(
                _REST_BASE[market.value],
                _REST_KLINES[market.value],
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1_000,
                },
            )
            payload = self._request_json(url)
            if not isinstance(payload, list):
                raise BinanceTransportError("Binance kline response must be a list")
            chunk: list[list[object]] = []
            for item in payload:
                if not isinstance(item, list):
                    raise BinanceTransportError("Binance kline row must be a list")
                chunk.append(item)
            if not chunk:
                break
            rows.extend(chunk)
            last_open = _normalize_epoch_ms(chunk[-1][0])
            next_cursor = last_open + step
            if next_cursor <= cursor:
                raise BinanceTransportError(
                    "Binance REST kline pagination did not advance"
                )
            cursor = next_cursor
            if len(chunk) < 1_000:
                break
        return rows

    def _load_vision_klines(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[list[object]]:
        result: list[list[object]] = []
        start_time = datetime.fromtimestamp(start_ms / 1_000, tz=UTC)
        end_time = datetime.fromtimestamp(end_ms / 1_000, tz=UTC)
        for url in plan_vision_kline_urls(
            market, symbol, interval, start_time, end_time
        ):
            rows = _csv_rows_from_zip(self._request_bytes(url), source=url)
            if rows and _looks_like_header(rows[0]):
                rows = rows[1:]
            for row in rows:
                if len(row) < 8:
                    raise BinanceTransportError(
                        f"Binance Vision kline row is short: {url}"
                    )
                open_ms = _normalize_epoch_ms(row[0])
                if start_ms <= open_ms < end_ms:
                    result.append(list(row))
        return result

    def load_klines(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> tuple[list[list[object]], str]:
        resolved_market = _market(market)
        resolved_mode = _mode(mode)
        if resolved_mode is BinanceTransportMode.REST:
            return (
                self._load_rest_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "rest",
            )
        if resolved_mode is BinanceTransportMode.VISION:
            return (
                self._load_vision_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "vision",
            )
        try:
            return (
                self._load_rest_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "rest",
            )
        except BinanceTransportError:
            return (
                self._load_vision_klines(
                    market=resolved_market,
                    symbol=symbol,
                    interval=interval,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "vision",
            )

    def _load_rest_funding(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[tuple[int, float, float | None]]:
        if market is BinanceMarket.SPOT:
            return []
        cursor = start_ms
        result: list[tuple[int, float, float | None]] = []
        while cursor < end_ms:
            url = self._query_url(
                _REST_BASE[market.value],
                _REST_FUNDING[market.value],
                {
                    "symbol": symbol,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1_000,
                },
            )
            payload = self._request_json(url)
            if not isinstance(payload, list):
                raise BinanceTransportError("Binance funding response must be a list")
            chunk: list[tuple[int, float, float | None]] = []
            for item in payload:
                if not isinstance(item, dict):
                    raise BinanceTransportError("Binance funding row must be an object")
                try:
                    timestamp = _normalize_epoch_ms(item.get("fundingTime"))
                    rate = _finite_float(item.get("fundingRate"), field="funding rate")
                    mark_value = item.get("markPrice")
                    mark_price = (
                        None
                        if mark_value is None or mark_value == ""
                        else _finite_float(
                            mark_value,
                            field="funding settlement mark price",
                        )
                    )
                except (TypeError, ValueError) as error:
                    raise BinanceTransportError(
                        f"Binance funding row is invalid: {error}"
                    ) from error
                if mark_price is not None and mark_price <= 0.0:
                    raise BinanceTransportError(
                        "Binance funding settlement mark price must be positive"
                    )
                if start_ms <= timestamp <= end_ms:
                    chunk.append((timestamp, rate, mark_price))
            result.extend(chunk)
            if len(chunk) < 1_000:
                break
            next_cursor = chunk[-1][0] + 1
            if next_cursor <= cursor:
                raise BinanceTransportError(
                    "Binance funding pagination did not advance"
                )
            cursor = next_cursor
        return result

    def _load_vision_funding(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[tuple[int, float, float | None]]:
        if market is BinanceMarket.SPOT:
            return []
        result: list[tuple[int, float, float | None]] = []
        for month in _iter_months(start_ms, end_ms):
            url = vision_funding_url(market, symbol, month)
            rows = _csv_rows_from_zip(self._request_bytes(url), source=url)
            if not rows:
                continue
            if not _looks_like_header(rows[0]):
                raise BinanceTransportError(
                    f"Binance Vision funding archive lacks a supported header: {url}"
                )
            header = {name.strip(): index for index, name in enumerate(rows[0])}
            if {"calc_time", "last_funding_rate"}.issubset(header):
                time_field = "calc_time"
                rate_field = "last_funding_rate"
            elif {"fundingTime", "fundingRate"}.issubset(header):
                time_field = "fundingTime"
                rate_field = "fundingRate"
            else:
                raise BinanceTransportError(
                    f"Binance Vision funding header is unsupported: {tuple(header)}"
                )
            mark_field = next(
                (name for name in ("markPrice", "mark_price") if name in header),
                None,
            )
            for row in rows[1:]:
                try:
                    timestamp = _normalize_epoch_ms(row[header[time_field]])
                    rate = _finite_float(row[header[rate_field]], field="funding rate")
                    mark_text = None if mark_field is None else row[header[mark_field]]
                    mark_price = (
                        None
                        if mark_text is None or not mark_text.strip()
                        else _finite_float(
                            mark_text,
                            field="funding settlement mark price",
                        )
                    )
                    if mark_price is not None and mark_price <= 0.0:
                        raise BinanceTransportError(
                            "Binance funding settlement mark price must be positive"
                        )
                except IndexError as error:
                    raise BinanceTransportError(
                        f"Binance Vision funding row is short: {url}"
                    ) from error
                if start_ms <= timestamp <= end_ms:
                    result.append((timestamp, rate, mark_price))
        return result

    def _load_vision_funding_range(
        self,
        *,
        market: BinanceMarket,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> tuple[list[tuple[int, float, float | None]], str]:
        end_time = datetime.fromtimestamp(end_ms / 1_000, tz=UTC)
        trailing_month_start = end_time.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        trailing_month_start_ms = int(trailing_month_start.timestamp() * 1_000)
        if end_ms == trailing_month_start_ms:
            return (
                self._load_vision_funding(
                    market=market,
                    symbol=symbol,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
                "vision",
            )

        vision_end_ms = max(start_ms, trailing_month_start_ms)
        result: list[tuple[int, float, float | None]] = []
        sources: list[str] = []
        if start_ms < vision_end_ms:
            result.extend(
                self._load_vision_funding(
                    market=market,
                    symbol=symbol,
                    start_ms=start_ms,
                    end_ms=vision_end_ms - 1,
                )
            )
            sources.append("vision")
        result.extend(
            self._load_rest_funding(
                market=market,
                symbol=symbol,
                start_ms=vision_end_ms,
                end_ms=end_ms,
            )
        )
        sources.append("rest")
        return result, "+".join(sources)

    def _complete_funding_marks(
        self,
        *,
        events: Sequence[tuple[int, float] | tuple[int, float, float | None]],
        market: BinanceMarket,
        symbol: str,
        start_ms: int,
        end_ms: int,
        source: str,
    ) -> tuple[list[tuple[int, float, float]], str]:
        normalized = [
            (event[0], event[1], None if len(event) == 2 else event[2])
            for event in events
        ]
        if all(mark is not None for _, _, mark in normalized):
            complete: list[tuple[int, float, float]] = []
            for timestamp, rate, mark in normalized:
                if mark is None:
                    raise AssertionError("funding mark completeness changed")
                if not math.isfinite(mark) or mark <= 0.0:
                    raise BinanceTransportError(
                        "Binance funding settlement mark price must be positive"
                    )
                complete.append((timestamp, rate, mark))
            return complete, source

        rest_events = self._load_rest_funding(
            market=market,
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        marks: dict[int, float] = {}
        for event in rest_events:
            if len(event) == 3 and event[2] is not None:
                marks[event[0]] = event[2]
        completed: list[tuple[int, float, float]] = []
        for timestamp, rate, mark_price in normalized:
            resolved_mark = (
                mark_price if mark_price is not None else marks.get(timestamp)
            )
            if resolved_mark is None:
                raise BinanceTransportError(
                    "Binance funding settlement mark price is unavailable"
                )
            if not math.isfinite(resolved_mark) or resolved_mark <= 0.0:
                raise BinanceTransportError(
                    "Binance funding settlement mark price must be positive"
                )
            completed.append((timestamp, rate, float(resolved_mark)))
        return completed, f"{source}+rest-marks"

    @staticmethod
    def _require_funding_marks(
        events: Sequence[tuple[int, float, float | None]],
    ) -> list[tuple[int, float, float]]:
        marked: list[tuple[int, float, float]] = []
        for timestamp, rate, mark_price in events:
            if mark_price is None:
                raise BinanceTransportError(
                    "Binance funding settlement mark price is unavailable"
                )
            if not math.isfinite(mark_price) or mark_price <= 0.0:
                raise BinanceTransportError(
                    "Binance funding settlement mark price must be positive"
                )
            marked.append((timestamp, rate, mark_price))
        return marked

    def load_funding_events(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        start_ms: int,
        end_ms: int,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> tuple[list[tuple[int, float, float]], str]:
        """Load funding settlements with the mark used by the exchange."""

        resolved_market = _market(market)
        if resolved_market is BinanceMarket.SPOT:
            return [], "spot:no-funding"
        resolved_mode = _mode(mode)
        if resolved_mode is BinanceTransportMode.REST:
            events = self._load_rest_funding(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return self._require_funding_marks(events), "rest"
        if resolved_mode is BinanceTransportMode.VISION:
            events, source = self._load_vision_funding_range(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return self._complete_funding_marks(
                events=events,
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                source=source,
            )
        try:
            events = self._load_rest_funding(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return self._require_funding_marks(events), "rest"
        except BinanceTransportError:
            events, source = self._load_vision_funding_range(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return self._complete_funding_marks(
                events=events,
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                source=source,
            )

    def load_funding_rates(
        self,
        *,
        market: BinanceMarket | str,
        symbol: str,
        start_ms: int,
        end_ms: int,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> tuple[list[tuple[int, float]], str]:
        """Load legacy timestamp/rate pairs without requiring settlement marks."""

        resolved_market = _market(market)
        if resolved_market is BinanceMarket.SPOT:
            return [], "spot:no-funding"
        resolved_mode = _mode(mode)
        if resolved_mode is BinanceTransportMode.REST:
            events = self._load_rest_funding(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return [(timestamp, rate) for timestamp, rate, _ in events], "rest"
        if resolved_mode is BinanceTransportMode.VISION:
            events, source = self._load_vision_funding_range(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return [(timestamp, rate) for timestamp, rate, _ in events], source
        try:
            events = self._load_rest_funding(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return [(timestamp, rate) for timestamp, rate, _ in events], "rest"
        except BinanceTransportError:
            events, source = self._load_vision_funding_range(
                market=resolved_market,
                symbol=symbol,
                start_ms=start_ms,
                end_ms=end_ms,
            )
            return [(timestamp, rate) for timestamp, rate, _ in events], source

    def load_exchange_information(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
    ) -> tuple[dict[str, object], str]:
        snapshot = self.load_exchange_information_snapshot(
            market=market,
            mode=mode,
        )
        return _mutable_json_object(snapshot.payload), "rest"

    def load_exchange_information_snapshot(
        self,
        *,
        market: BinanceMarket | str,
        mode: BinanceTransportMode | str = BinanceTransportMode.AUTO,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> BinanceExchangeInfoSnapshot:
        resolved_market = _market(market)
        resolved_mode = _mode(mode)
        if resolved_mode is BinanceTransportMode.VISION:
            raise BinanceTransportError(
                "Binance Vision does not publish exchange metadata; provide static metadata"
            )
        primary_url = (
            f"{_REST_BASE[resolved_market.value]}"
            f"{_REST_EXCHANGE_INFO[resolved_market.value]}"
        )
        url = primary_url
        try:
            raw_payload = self._request_bytes(primary_url)
        except BinanceTransportError:
            if resolved_market is not BinanceMarket.USDS_M or not self.allow_network:
                raise
            url = "https://www.binance.com/fapi/v1/exchangeInfo"
            raw_payload = self._request_bytes(url)
        try:
            payload = json.loads(raw_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BinanceTransportError(
                f"Binance returned invalid JSON for {url}"
            ) from error
        if not isinstance(payload, dict):
            raise BinanceTransportError(
                "Binance exchange information must be an object"
            )
        return BinanceExchangeInfoSnapshot(
            payload=dict(payload),
            raw_payload=raw_payload,
            source_uri=url,
            retrieved_at=_aware_utc(clock(), field="retrieved_at"),
            raw_payload_sha256=hashlib.sha256(raw_payload).hexdigest(),
        )


__all__ = ["BinancePublicTransport"]
