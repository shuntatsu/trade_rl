"""The only network client of the market-data store: a bounded Vision downloader.

It is imported lazily by the CLI and constructed only after ``--allow-network``; a
dry run (``plan``, or ``ingest`` without ``--allow-network``) never imports it.

Defense in depth at the lowest layer: before ANY network call, :meth:`VisionFetcher.fetch`
parses the URL itself (:func:`~trade_rl.integrations.market_store.archive_url.parse_archive_url`)
and refuses everything that is not a recognised Vision archive or ``.CHECKSUM`` path
whose period lies entirely before the FS1 warm-up start. Only a fetcher constructed
with an Issue #810 warm-up authorization URL may also take periods entirely before the
sealed window start; nothing can take a URL touching the sealed window, and nothing is
decided from caller-declared metadata. Redirects are never followed.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from trade_rl.integrations.market_store.archive_url import (
    VISION_PREFIX,
    check_url_fetchable,
)
from trade_rl.integrations.market_store.embargo import EmbargoPolicy
from trade_rl.integrations.market_store.ingest import FetchResult

USER_AGENT = "trade-rl/0.3 market-store public-market-data"
DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_RETRY_STATUS = frozenset({418, 429, 451, 500, 502, 503, 504})


class RedirectRefused(urllib.error.URLError):
    """A Vision response tried to redirect; redirects are never followed."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        # returning None makes urllib raise the 3xx as an HTTPError instead of
        # following it, so the next URL is never requested
        return None


Opener = Callable[..., Any]


class VisionFetcher:
    """Bounded HTTPS GET with a few retries; 404 is a result, not an error."""

    def __init__(
        self,
        *,
        warmup_authorization_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
        max_bytes: int = DEFAULT_MAX_BYTES,
        _opener: Opener | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_attempts <= 0 or max_bytes <= 0:
            raise ValueError("timeout, attempts and byte bound must be positive")
        # validates the URL shape; the boundaries themselves are not parameters
        self._policy = EmbargoPolicy(warmup_authorization_url=warmup_authorization_url)
        self._timeout = timeout_seconds
        self._attempts = max_attempts
        self._backoff = retry_backoff_seconds
        self._max_bytes = max_bytes
        self._open = (
            _opener
            if _opener is not None
            else urllib.request.build_opener(_NoRedirect).open
        )

    def fetch(self, url: str) -> FetchResult:
        # decided from the URL alone, before any network call
        check_url_fetchable(url, self._policy)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last_status = 0
        for attempt in range(self._attempts):
            try:
                with self._open(request, timeout=self._timeout) as response:
                    final_url = (
                        response.geturl() if hasattr(response, "geturl") else url
                    )
                    if final_url != url:
                        # defense in depth: even a followed redirect is re-checked
                        check_url_fetchable(str(final_url), self._policy)
                        raise RedirectRefused(
                            f"response for {url} came from a different URL {final_url}"
                        )
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
            except RedirectRefused:
                raise
            except (TimeoutError, urllib.error.URLError):
                last_status = 0
            if attempt + 1 < self._attempts and self._backoff > 0:
                time.sleep(self._backoff * (2**attempt))
        return FetchResult(status=last_status, body=None)


__all__ = ["VISION_PREFIX", "RedirectRefused", "VisionFetcher"]
