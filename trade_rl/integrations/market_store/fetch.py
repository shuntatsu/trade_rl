"""The only network client of the market-data store: a bounded Vision downloader.

It is imported lazily by the CLI and constructed only after ``--allow-network``; a
dry run (``plan``, or ``ingest`` without ``--allow-network``) never imports it. It
fetches only ``https://data.binance.vision/data/`` URLs.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

from trade_rl.integrations.market_store.ingest import FetchResult

VISION_PREFIX = "https://data.binance.vision/data/"
USER_AGENT = "trade-rl/0.3 market-store public-market-data"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_RETRY_STATUS = frozenset({418, 429, 451, 500, 502, 503, 504})


class VisionFetcher:
    """Bounded HTTPS GET with a few retries; 404 is a result, not an error."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if timeout_seconds <= 0 or max_attempts <= 0 or max_bytes <= 0:
            raise ValueError("timeout, attempts and byte bound must be positive")
        self._timeout = timeout_seconds
        self._attempts = max_attempts
        self._backoff = retry_backoff_seconds
        self._max_bytes = max_bytes

    def fetch(self, url: str) -> FetchResult:
        if not url.startswith(VISION_PREFIX):
            raise ValueError("only official Binance Vision URLs may be fetched")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last_status = 0
        for attempt in range(self._attempts):
            try:
                with urllib.request.urlopen(  # noqa: S310 - fixed HTTPS prefix
                    request, timeout=self._timeout
                ) as response:
                    body = response.read(self._max_bytes + 1)
                    if len(body) > self._max_bytes:
                        raise RuntimeError(
                            f"response exceeds {self._max_bytes} bytes: {url}"
                        )
                    return FetchResult(status=int(response.status), body=body)
            except urllib.error.HTTPError as error:
                last_status = int(error.code)
                if last_status == 404 or last_status not in _RETRY_STATUS:
                    return FetchResult(status=last_status, body=None)
            except (TimeoutError, urllib.error.URLError):
                last_status = 0
            if attempt + 1 < self._attempts and self._backoff > 0:
                time.sleep(self._backoff * (2**attempt))
        return FetchResult(status=last_status, body=None)


__all__ = ["VISION_PREFIX", "VisionFetcher"]
