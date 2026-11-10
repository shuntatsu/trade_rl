"""Capture timestamps must never make a later fetch visible at an earlier as-of."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.integrations.cross_asset.test_store_fetch import FakeTransport, _fred
from trade_rl.integrations.cross_asset.catalog import series_by_key
from trade_rl.integrations.cross_asset.fetch import fetch_all
from trade_rl.integrations.cross_asset.store import (
    CrossAssetStore,
    latest_records,
    manifest_digest,
    parse_utc,
)

SECOND = datetime(2025, 7, 10, 12, 0, 0, tzinfo=UTC)


def _fetch(store: CrossAssetStore, when: datetime, value: str) -> dict[str, object]:
    payload = _fred("DGS10", [("2025-07-08", value)])
    return fetch_all(
        store,
        (series_by_key("DGS10"),),
        FakeTransport({"id=DGS10": payload}),
        now=lambda: when,
        sleep=lambda _delay: None,
    )[0]


def test_new_capture_preserves_microseconds_and_never_leaks_to_earlier_asof(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path)
    captured = SECOND + timedelta(microseconds=900_000)
    record = _fetch(store, captured, "4.90")
    assert record["fetched_at"] == "2025-07-10T12:00:00.900000Z"
    assert record["captured_at"] == record["fetched_at"]
    records = store.read_manifest()
    assert latest_records(records, as_of=SECOND + timedelta(microseconds=100_000)) == {}
    assert (
        latest_records(records, as_of=captured)["DGS10"]["sha256"] == record["sha256"]
    )


def test_two_revisions_in_one_second_have_distinct_ordered_vintages(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path)
    first = _fetch(store, SECOND + timedelta(microseconds=250_000), "4.90")
    later = _fetch(store, SECOND + timedelta(microseconds=750_000), "4.95")
    assert first["path"] != later["path"]
    records = store.read_manifest()
    between = latest_records(records, as_of=SECOND + timedelta(microseconds=500_000))
    assert between["DGS10"]["sha256"] == first["sha256"]
    assert latest_records(records)["DGS10"]["sha256"] == later["sha256"]


def test_old_second_only_capture_is_conservatively_unknown_during_that_second() -> None:
    legacy = {
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "captured_at": "2025-07-10T12:00:00Z",
        "sha256": "a" * 64,
    }
    # Legacy writers truncated actual subsecond capture time. The exact instant
    # is unknowable; never claim this row was fetched at the start of the second.
    assert latest_records([legacy], as_of=SECOND) == {}
    assert (
        latest_records([legacy], as_of=SECOND + timedelta(microseconds=900_000)) == {}
    )
    assert latest_records([legacy], as_of=SECOND + timedelta(seconds=1)) == {
        "DGS10": legacy
    }


def test_capture_time_is_part_of_the_feature_input_provenance() -> None:
    record = {
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00.250000Z",
        "captured_at": "2025-07-10T12:00:00.250000Z",
        "sha256": "a" * 64,
    }
    later = {
        **record,
        "fetched_at": "2025-07-10T12:00:00.750000Z",
        "captured_at": "2025-07-10T12:00:00.750000Z",
    }
    assert manifest_digest({"DGS10": record}) != manifest_digest({"DGS10": later})
    # A same-byte recheck does not change the vintage's input provenance:
    # its fetched_at changes, while the original captured_at remains fixed.
    recheck = {**record, "status": "unchanged", "fetched_at": later["fetched_at"]}
    assert manifest_digest({"DGS10": record}) == manifest_digest({"DGS10": recheck})


def test_new_exact_second_capture_is_known_at_that_second(tmp_path: Path) -> None:
    store = CrossAssetStore(tmp_path)
    record = _fetch(store, SECOND, "4.90")
    assert record["fetched_at"] == "2025-07-10T12:00:00.000000Z"
    assert latest_records(store.read_manifest(), as_of=SECOND)["DGS10"] == record


def test_future_unchanged_recheck_cannot_be_selected_before_its_fetch_time() -> None:
    original = {
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00.000000Z",
        "captured_at": "2025-07-10T12:00:00.000000Z",
        "sha256": "a" * 64,
    }
    recheck = {
        **original,
        "status": "unchanged",
        "fetched_at": "2025-07-12T12:00:00.000000Z",
    }
    assert latest_records([original, recheck], as_of=SECOND + timedelta(seconds=1)) == {
        "DGS10": original
    }
    assert latest_records([original, recheck], as_of=SECOND + timedelta(days=3)) == {
        "DGS10": recheck
    }


@pytest.mark.parametrize(
    "invalid",
    [
        "2025-07-10T12:00:00+09:00Z",
        "2025-07-10T12:00:00+09:00",
        "2025-07-10T12:00:00",
    ],
)
def test_parse_utc_refuses_noncanonical_zulu_or_naive_timestamps(invalid: str) -> None:
    with pytest.raises(ValueError, match="UTC timestamp"):
        parse_utc(invalid)


def test_asof_cutoff_requires_timezone_aware_datetime() -> None:
    record = {
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00.000000Z",
        "captured_at": "2025-07-10T12:00:00.000000Z",
        "sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="as_of must be timezone-aware"):
        latest_records([record], as_of=datetime(2025, 7, 10, 12, 0))


def test_legacy_recheck_uses_first_capture_when_same_raw_file_has_multiple_ok_rows(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path)
    first = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/same.csv",
        "sha256": "a" * 64,
    }
    second = {**first, "fetched_at": "2025-07-11T12:00:00Z"}
    recheck = {
        **first,
        "status": "unchanged",
        "fetched_at": "2025-07-12T12:00:00Z",
    }
    (tmp_path / "manifest.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in (first, second, recheck)
        ),
        encoding="utf-8",
    )
    loaded = store.read_manifest()
    # All legacy references to the same immutable raw bytes share the earliest
    # captured_at, not merely an unchanged recheck of that raw path.
    assert loaded[0]["captured_at"] == first["fetched_at"]
    assert loaded[1]["captured_at"] == first["fetched_at"]
    assert loaded[2]["captured_at"] == first["fetched_at"]
    assert (
        latest_records(loaded, as_of=SECOND + timedelta(days=1, seconds=1))["DGS10"][
            "fetched_at"
        ]
        == second["fetched_at"]
    )
