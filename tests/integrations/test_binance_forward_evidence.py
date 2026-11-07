import hashlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

from tests.integrations.test_binance_forward import MS, NOW, PublicFeed
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper import account
from trade_rl.integrations.binance.forward import (
    MAX_CONSUME_LATENCY_MS,
    capture_forward_snapshot,
)
from trade_rl.integrations.binance.forward_evidence import read_forward_snapshot


def captured(tmp_path):
    root = tmp_path / "capture"
    snapshot = capture_forward_snapshot(
        root, transport=PublicFeed(), clock=lambda: NOW, monotonic=lambda: 1.0
    )
    return root, snapshot


def write(root, snapshot):
    (root / "snapshot.json").write_bytes(canonical_json_bytes(snapshot))


def test_reader_reconstructs_raw_source_and_can_bind_parent_digest(tmp_path):
    root, snapshot = captured(tmp_path)
    digest = hashlib.sha256((root / "snapshot.json").read_bytes()).hexdigest()
    assert read_forward_snapshot(root, expected_sha256=digest) == snapshot
    assert read_forward_snapshot(root, as_of=NOW + timedelta(seconds=5)) == snapshot
    with pytest.raises(ValueError, match="digest"):
        read_forward_snapshot(root, expected_sha256="0" * 64)


def test_reader_preserves_the_original_twenty_level_historical_profile(tmp_path):
    root, snapshot = captured(tmp_path)
    snapshot["schema"] = "binance_forward_market_snapshot_v1"
    snapshot.pop("depth_limit")
    for row in snapshot["responses"]:
        if "/depth?" in row["url"]:
            row["url"] = row["url"].replace("limit=100", "limit=20")
            (root / (row["label"] + ".json")).write_bytes(canonical_json_bytes(row))
    write(root, snapshot)
    assert read_forward_snapshot(root) == snapshot


@pytest.mark.parametrize("defect", ["schema_only", "depth_limit", "depth_type"])
def test_reader_rejects_relabelled_depth_profiles(tmp_path, defect):
    root, snapshot = captured(tmp_path)
    if defect == "schema_only":
        snapshot["schema"] = "binance_forward_market_snapshot_v1"
        snapshot.pop("depth_limit")
    else:
        snapshot["depth_limit"] = 20 if defect == "depth_limit" else 100.0
    write(root, snapshot)
    with pytest.raises(ValueError):
        read_forward_snapshot(root)


# Request schedule (label, start ms, end ms after capture start) of a slow but
# legal capture: it spans 5.62 s, its oldest quote (the BTCUSDT premiumIndex
# ``time``) is 4.48 s old at completion, and two ETHUSDT depth requests take
# 1.45 s each. Quote times are offsets from capture start in milliseconds.
SLOW_SCHEDULE = [
    ("spot_time", 0, 100),
    ("futures_time", 100, 200),
    ("BTCUSDT_spot_depth", 200, 1200),
    ("BTCUSDT_perp_depth", 1200, 1300),
    ("BTCUSDT_mark", 1300, 1400),
    ("BTCUSDT_funding", 1400, 1500),
    ("ETHUSDT_spot_depth", 1500, 2950),
    ("ETHUSDT_perp_depth", 2950, 4400),
    ("ETHUSDT_mark", 4400, 4500),
    ("ETHUSDT_funding", 4500, 5620),
]
SLOW_QUOTE_OFFSETS = {
    "BTCUSDT_perp_depth": 1250,
    "BTCUSDT_mark": 1140,
    "ETHUSDT_perp_depth": 4350,
    "ETHUSDT_mark": 4450,
}
COMPLETED = NOW + timedelta(milliseconds=5620)


class SlowTimeline:
    """Synthetic wall and monotonic clocks driven by the request schedule."""

    def __init__(self):
        self.ms = 0

    def clock(self):
        return NOW + timedelta(milliseconds=self.ms)

    def monotonic(self):
        return self.ms / 1000


class SlowFeed(PublicFeed):
    def __init__(self, timeline):
        super().__init__()
        self.timeline = timeline
        self.pending = iter(SLOW_SCHEDULE)

    def _request_bytes(self, url, *, max_response_bytes=None):
        label, start, end = next(self.pending)
        assert self.timeline.ms == start
        payload = json.loads(super()._request_bytes(url))
        offset = SLOW_QUOTE_OFFSETS.get(label)
        if label.endswith("_time"):
            payload["serverTime"] = MS + start
        elif label.endswith("_perp_depth"):
            payload.update(E=MS + offset, T=MS + offset)
        elif label.endswith("_mark"):
            payload["time"] = MS + offset
        self.timeline.ms = end
        return json.dumps(payload).encode()


def slow_captured(tmp_path):
    root = tmp_path / "slow"
    timeline = SlowTimeline()
    snapshot = capture_forward_snapshot(
        root,
        transport=SlowFeed(timeline),
        clock=timeline.clock,
        monotonic=timeline.monotonic,
    )
    return root, snapshot


def test_slow_capture_fixture_has_the_incident_timing(tmp_path):
    root, snapshot = slow_captured(tmp_path)
    assert snapshot["capture_elapsed_seconds"] == 5.62
    assert read_forward_snapshot(root) == snapshot
    mark = snapshot["market"]["BTCUSDT"]["mark_quote"]["time"]
    assert COMPLETED.timestamp() * 1000 - mark == 4480


def test_snapshot_accepted_by_its_own_capture_is_not_stale_a_moment_later(tmp_path):
    # Regression: the 5 s per-quote bound was re-applied at consumption time,
    # so a 4.48 s old quote on a slow 5.62 s capture failed 0.6 s after
    # completion although capture had accepted it.
    root, snapshot = slow_captured(tmp_path)
    consumed = COMPLETED + timedelta(milliseconds=600)
    assert read_forward_snapshot(root, as_of=consumed) == snapshot


def test_consumption_latency_is_bounded_but_inclusive(tmp_path):
    root, snapshot = slow_captured(tmp_path)
    limit = timedelta(milliseconds=MAX_CONSUME_LATENCY_MS)
    assert read_forward_snapshot(root, as_of=COMPLETED) == snapshot
    assert read_forward_snapshot(root, as_of=COMPLETED + limit) == snapshot
    with pytest.raises(ValueError, match="consumed too late"):
        read_forward_snapshot(root, as_of=COMPLETED + limit + timedelta(milliseconds=1))


def test_consumption_cannot_precede_capture_completion(tmp_path):
    root, _ = slow_captured(tmp_path)
    with pytest.raises(ValueError, match="has not been received"):
        read_forward_snapshot(root, as_of=COMPLETED - timedelta(milliseconds=1))


def test_consume_latency_matches_the_paper_decision_window():
    # The paper window is a literal inside trade_rl/evaluation/paper
    # (PaperAccount.execute, supervisor._execute_pending). That sealed code is
    # deliberately not refactored here, so this guard pins the literal in the
    # source text on both sides instead of importing a shared constant.
    assert MAX_CONSUME_LATENCY_MS == 10_000
    paper = Path(account.__file__).parent
    assert "(at - decision_at).total_seconds() > 10:" in (
        paper / "account.py"
    ).read_text(encoding="utf-8")
    assert "(at - decision_at).total_seconds() <= 10:" in (
        paper / "supervisor.py"
    ).read_text(encoding="utf-8")


def test_reader_still_rejects_quote_older_than_five_seconds_at_completion(tmp_path):
    root, snapshot = slow_captured(tmp_path)
    # Age the BTCUSDT mark quote to 5.07 s at completion (0.85 s at its receipt,
    # so only the completion-time bound can catch it) and re-hash the evidence.
    record = snapshot["responses"][4]
    assert record["label"] == "BTCUSDT_mark"
    mark = snapshot["market"]["BTCUSDT"]["mark_quote"]
    mark["time"] = MS + 550
    raw = json.dumps(mark).encode()
    record["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    record["size_bytes"] = len(raw)
    (root / "BTCUSDT_mark.raw").write_bytes(raw)
    (root / "BTCUSDT_mark.json").write_bytes(canonical_json_bytes(record))
    write(root, snapshot)
    with pytest.raises(ValueError, match="stale"):
        read_forward_snapshot(root)
    with pytest.raises(ValueError, match="stale"):
        read_forward_snapshot(root, as_of=COMPLETED + timedelta(milliseconds=1))


@pytest.mark.parametrize(
    "defect", ["raw", "sidecar", "summary", "roster", "url", "time", "span", "failure"]
)
def test_tampered_evidence_fails_closed(tmp_path, defect):
    root, snapshot = captured(tmp_path)
    record = snapshot["responses"][2]
    if defect == "raw":
        (root / (record["label"] + ".raw")).write_bytes(b"{}")
    elif defect == "sidecar":
        (root / (record["label"] + ".json")).write_text("{}")
    elif defect == "summary":
        snapshot["market"]["BTCUSDT"]["spot_depth"]["bids"][0][0] = "50"
    elif defect == "roster":
        snapshot["responses"].pop()
    elif defect in ("url", "time", "span"):
        key, value = {
            "url": ("url", "https://wrong.example/depth"),
            "time": ("requested_at", (NOW - timedelta(seconds=1)).isoformat()),
            "span": ("capture_elapsed_seconds", 600.0),
        }[defect]
        record[key] = value
        (root / (record["label"] + ".json")).write_bytes(canonical_json_bytes(record))
    else:
        (root / "failure.json").write_text("{}")
    write(root, snapshot)
    with pytest.raises(ValueError):
        read_forward_snapshot(root)


def test_rehashed_malformed_raw_cannot_hide_behind_matching_summary(tmp_path):
    root, snapshot = captured(tmp_path)
    record = snapshot["responses"][3]
    snapshot["market"]["BTCUSDT"]["perpetual_depth"]["asks"][0][0] = "99"
    raw = canonical_json_bytes(snapshot["market"]["BTCUSDT"]["perpetual_depth"])
    record["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    record["size_bytes"] = len(raw)
    (root / (record["label"] + ".raw")).write_bytes(raw)
    (root / (record["label"] + ".json")).write_bytes(canonical_json_bytes(record))
    write(root, snapshot)
    with pytest.raises(ValueError, match="crossed"):
        read_forward_snapshot(root)


def test_duplicate_json_keys_are_not_accepted(tmp_path):
    root, _ = captured(tmp_path)
    raw = (root / "snapshot.json").read_text()
    (root / "snapshot.json").write_text(
        raw.replace('"eligible":true', '"eligible":false,"eligible":true')
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_forward_snapshot(root)
