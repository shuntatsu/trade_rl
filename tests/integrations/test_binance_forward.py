import json
from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.integrations.binance import forward

NOW = datetime(2026, 9, 18, 0, tzinfo=UTC)
MS = int(NOW.timestamp() * 1000)


class PublicFeed:
    def __init__(self, defect=None):
        self.urls = []
        self.defect = defect

    def _request_bytes(self, url):
        self.urls.append(url)
        symbol = "ETHUSDT" if "ETHUSDT" in url else "BTCUSDT"
        if url.endswith("/time"):
            payload = {"serverTime": MS}
        elif "/depth?" in url:
            payload = {
                "lastUpdateId": 1,
                "bids": [["100", "10"], ["99", "20"]],
                "asks": [["101", "11"], ["102", "20"]],
            }
            if "fapi" in url:
                payload.update(E=MS, T=MS)
            if self.defect == "crossed":
                payload["asks"][0][0] = "99"
            if self.defect == "negative":
                payload["bids"][0][1] = "-1"
            if self.defect == "unordered":
                payload["bids"].reverse()
            if self.defect == "stale" and "fapi" in url:
                payload["E"] = MS - 6000
            if self.defect == "future" and "fapi" in url:
                payload["E"] = MS + 2000
        elif "/premiumIndex?" in url:
            payload = {
                "symbol": symbol,
                "markPrice": "100.5",
                "indexPrice": "100.4",
                "lastFundingRate": "0.0001",
                "nextFundingTime": MS + 28800000,
                "time": MS,
            }
            if self.defect == "symbol":
                payload["symbol"] = "WRONG"
        else:
            payload = [
                {
                    "symbol": symbol,
                    "fundingRate": "-0.0001",
                    "fundingTime": MS - 1000,
                    "markPrice": "100.5",
                }
            ]
            if self.defect == "funding_future":
                payload[0]["fundingTime"] = MS + 2000
            if self.defect == "duplicate":
                payload *= 2
        if self.defect == "unavailable" and "ETHUSDT" in url:
            raise RuntimeError("feed unavailable")
        return json.dumps(payload).encode()


def test_capture_keeps_raw_evidence_and_does_not_treat_quoted_funding_as_settled(
    tmp_path,
):
    feed = PublicFeed()
    snapshot = forward.capture_forward_snapshot(
        tmp_path / "capture", transport=feed, clock=lambda: NOW, monotonic=lambda: 1.0
    )
    assert snapshot["eligible"] is True and snapshot["production_eligible"] is False
    assert len(feed.urls) == 10 and all(
        "order?" not in url and "signature" not in url for url in feed.urls
    )
    assert (
        snapshot["market"]["BTCUSDT"]["settled_funding"][0]["fundingRate"] == "-0.0001"
    )
    assert snapshot["market"]["BTCUSDT"]["mark_quote"]["lastFundingRate"] == "0.0001"
    assert (
        snapshot["market"]["BTCUSDT"]["spot_depth_time_basis"] == "local_receipt_only"
    )
    assert len(snapshot["responses"]) == 10
    assert len(list((tmp_path / "capture").glob("*.raw"))) == 10
    assert (tmp_path / "capture/snapshot.json").is_file()
    with pytest.raises(FileExistsError):
        forward.capture_forward_snapshot(
            tmp_path / "capture",
            transport=feed,
            clock=lambda: NOW,
            monotonic=lambda: 1.0,
        )
    assert len(feed.urls) == 10


@pytest.mark.parametrize(
    "defect",
    [
        "crossed",
        "negative",
        "unordered",
        "stale",
        "future",
        "symbol",
        "funding_future",
        "duplicate",
        "unavailable",
    ],
)
def test_bad_or_partial_market_evidence_never_publishes_eligible_snapshot(
    tmp_path, defect
):
    with pytest.raises((ValueError, RuntimeError)):
        forward.capture_forward_snapshot(
            tmp_path / "capture",
            transport=PublicFeed(defect),
            clock=lambda: NOW,
            monotonic=lambda: 1.0,
        )
    assert not (tmp_path / "capture/snapshot.json").exists()
    assert (tmp_path / "capture/failure.json").is_file()
    assert list((tmp_path / "capture").glob("*.raw"))


def test_slow_response_and_clock_jump_are_rejected(tmp_path):
    ticks = iter([0.0, 0.0, 6.0])
    with pytest.raises(ValueError, match="duration"):
        forward.capture_forward_snapshot(
            tmp_path / "slow",
            transport=PublicFeed(),
            clock=lambda: NOW,
            monotonic=lambda: next(ticks),
        )
    clock = iter([NOW, NOW, NOW - timedelta(seconds=1)])
    with pytest.raises(ValueError, match="clock"):
        forward.capture_forward_snapshot(
            tmp_path / "clock",
            transport=PublicFeed(),
            clock=lambda: next(clock),
            monotonic=lambda: 1.0,
        )


def test_clock_cannot_move_backwards_between_separate_requests(tmp_path):
    clock = iter(
        [
            NOW,
            NOW,
            NOW + timedelta(seconds=1),
            NOW + timedelta(seconds=0.5),
            NOW + timedelta(seconds=0.5),
        ]
    )
    ticks = iter([0.0, 0.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="clock"):
        forward.capture_forward_snapshot(
            tmp_path / "capture",
            transport=PublicFeed(),
            clock=lambda: next(clock),
            monotonic=lambda: next(ticks),
        )


def test_capture_wide_monotonic_span_prevents_hidden_clock_adjustments(tmp_path):
    class Clock:
        calls = 0

        def __call__(self):
            self.calls += 1
            return NOW + timedelta(seconds=(self.calls - 1) * 0.1)

    class Ticks:
        calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls == 1:
                return 0.0
            return ((self.calls - 2) // 2) * 60 + ((self.calls - 2) % 2) * 0.1

    with pytest.raises(ValueError, match="span|clock"):
        forward.capture_forward_snapshot(
            tmp_path / "capture",
            transport=PublicFeed(),
            clock=Clock(),
            monotonic=Ticks(),
        )


def test_quotes_must_still_be_fresh_when_the_whole_snapshot_completes(tmp_path):
    times = [0.0] + [t for i in range(10) for t in (i * 0.6, i * 0.6 + 0.1)]
    clock = iter(times)
    ticks = iter(times)
    with pytest.raises(ValueError, match="stale"):
        forward.capture_forward_snapshot(
            tmp_path / "capture",
            transport=PublicFeed(),
            clock=lambda: NOW + timedelta(seconds=next(clock)),
            monotonic=lambda: next(ticks),
        )
