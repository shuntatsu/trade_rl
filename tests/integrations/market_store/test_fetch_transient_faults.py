"""Transient transport faults are retried and then reported as status 0.

A connection reset, a body cut short (``IncompleteRead``) or another ``OSError`` must
not escape as an unclassified error: the carry screen maps status 0 to a resumable
operational halt. A redirect is still refused outright.
"""

from __future__ import annotations

import http.client

import pytest

from tests.integrations.market_store.test_url_gate import (
    SPOT,
    Opener,
    Response,
    fetcher_with,
)
from trade_rl.integrations.market_store.fetch import RedirectRefused, VisionFetcher
from trade_rl.integrations.market_store.ingest import RESPONSE_TOO_LARGE_STATUS

URL = f"{SPOT}-2021-01.zip"


def raising(error: BaseException, *, times: int):
    state = {"left": times}

    def respond(url: str) -> Response:
        if state["left"]:
            state["left"] -= 1
            raise error
        return Response(url)

    return respond


@pytest.mark.parametrize(
    "error",
    [
        ConnectionResetError("reset by peer"),
        TimeoutError("timed out"),
        OSError("network is unreachable"),
        http.client.IncompleteRead(b"par", 10),
        http.client.RemoteDisconnected("closed"),
    ],
)
def test_a_transient_fault_that_clears_is_retried_to_success(error) -> None:
    opener = Opener(raising(error, times=2))
    assert fetcher_with(opener).fetch(URL).status == 200
    assert len(opener.calls) == 3


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset"), http.client.IncompleteRead(b"", 5)],
)
def test_a_persistent_transient_fault_is_status_zero_not_an_exception(error) -> None:
    opener = Opener(raising(error, times=99))
    result = fetcher_with(opener).fetch(URL)
    assert (result.status, result.body) == (0, None)
    assert len(opener.calls) == 3


def test_a_redirect_is_still_raised_not_retried() -> None:
    def redirect(url: str) -> Response:
        return Response(f"{SPOT}-2021-02.zip")

    opener = Opener(redirect)
    with pytest.raises(RedirectRefused):
        fetcher_with(opener).fetch(URL)
    assert len(opener.calls) == 1


def test_an_oversize_response_is_a_status_not_an_unclassified_error() -> None:
    opener = Opener(lambda url: Response(url, body=b"x" * 11))
    fetcher = VisionFetcher(retry_backoff_seconds=0.0, max_bytes=10, _opener=opener)
    result = fetcher.fetch(URL)
    assert (result.status, result.body) == (RESPONSE_TOO_LARGE_STATUS, None)
    # deterministic, so it is not retried
    assert len(opener.calls) == 1


def test_a_response_at_the_byte_bound_is_still_accepted() -> None:
    opener = Opener(lambda url: Response(url, body=b"x" * 10))
    fetcher = VisionFetcher(retry_backoff_seconds=0.0, max_bytes=10, _opener=opener)
    assert fetcher.fetch(URL).status == 200
