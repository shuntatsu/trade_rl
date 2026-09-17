import hashlib
from datetime import timedelta

import pytest

from tests.integrations.test_binance_forward import NOW, PublicFeed
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.integrations.binance.forward import capture_forward_snapshot
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


@pytest.mark.parametrize("seconds", [-0.001, 5.001])
def test_execution_consumption_cannot_use_future_or_stale_snapshot(tmp_path, seconds):
    root, _ = captured(tmp_path)
    with pytest.raises(ValueError):
        read_forward_snapshot(root, as_of=NOW + timedelta(seconds=seconds))


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
