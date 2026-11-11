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


def test_existing_explicit_capture_time_beats_later_fetched_at(
    tmp_path: Path,
) -> None:
    store = CrossAssetStore(tmp_path)
    original_capture = "2025-07-10T12:00:00.250000Z"
    modern = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-11T12:00:00.250000Z",
        "captured_at": original_capture,
        "path": "raw/fred/DGS10/retained.csv",
        "sha256": "a" * 64,
    }
    legacy_recheck = {
        **modern,
        "status": "unchanged",
        "fetched_at": "2025-07-12T12:00:00Z",
    }
    del legacy_recheck["captured_at"]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in (modern, legacy_recheck)
        ),
        encoding="utf-8",
    )
    found = store.read_manifest()
    assert found[0]["captured_at"] == original_capture
    assert found[1]["captured_at"] == original_capture


@pytest.mark.parametrize("missing_field", ["path", "sha256"])
def test_usable_manifest_record_without_raw_identity_fails_closed(
    tmp_path: Path, missing_field: str
) -> None:
    store = CrossAssetStore(tmp_path)
    invalid = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/raw.csv",
        "sha256": "a" * 64,
    }
    del invalid[missing_field]
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(invalid, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw provenance"):
        store.read_manifest()


def test_unchanged_from_other_series_cannot_inherit_capture_authority(
    tmp_path: Path,
) -> None:
    """Sharing raw path and checksum never proves a different series had an origin."""

    origin = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/original.csv",
        "sha256": "a" * 64,
    }
    foreign = {
        **origin,
        "key": "DGS2",
        "status": "unchanged",
        "fetched_at": "2025-07-11T12:00:00Z",
    }
    store = CrossAssetStore(tmp_path)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in (origin, foreign)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="without captured_at"):
        store.read_manifest()


def test_distinct_series_cannot_share_legacy_origin_clock(
    tmp_path: Path,
) -> None:
    """The first capture for B is B's own, not A's earlier matching raw reference."""

    origin = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/legacy-shared/file.csv",
        "sha256": "b" * 64,
    }
    second = {**origin, "key": "DGS2", "fetched_at": "2025-07-11T12:00:00Z"}
    recheck = {
        **second,
        "status": "unchanged",
        "fetched_at": "2025-07-12T12:00:00Z",
    }
    store = CrossAssetStore(tmp_path)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in (origin, second, recheck)
        ),
        encoding="utf-8",
    )
    loaded = store.read_manifest()
    assert loaded[0]["captured_at"] == origin["fetched_at"]
    assert loaded[1]["captured_at"] == second["fetched_at"]
    assert loaded[2]["captured_at"] == second["fetched_at"]


def test_legacy_multiple_raw_origins_resolve_independently_of_record_order(
    tmp_path: Path,
) -> None:
    """One series may have two vintages; neither may borrow the other's capture."""

    a = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/a.csv",
        "sha256": "a" * 64,
    }
    a_late = {**a, "fetched_at": "2025-07-12T12:00:00Z"}
    b = {
        **a,
        "fetched_at": "2025-07-11T12:00:00Z",
        "path": "raw/fred/DGS10/b.csv",
        "sha256": "b" * 64,
    }
    a_recheck = {**a, "fetched_at": "2025-07-13T12:00:00Z", "status": "unchanged"}
    b_recheck = {**b, "fetched_at": "2025-07-14T12:00:00Z", "status": "unchanged"}
    # Manifest order does not define the earliest captured_at: both vintages
    # must find their own minimum even when an earlier receipt is listed last.
    rows = (a_late, b, a, b_recheck, a_recheck)
    store = CrossAssetStore(tmp_path)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows),
        encoding="utf-8",
    )
    found = store.read_manifest()
    for index in (0, 2, 4):
        assert found[index]["captured_at"] == a["fetched_at"]
    for index in (1, 3):
        assert found[index]["captured_at"] == b["fetched_at"]
    at_a = latest_records(found, as_of=SECOND + timedelta(seconds=1))
    assert at_a["DGS10"]["sha256"] == a["sha256"]
    at_b = latest_records(found, as_of=SECOND + timedelta(days=1, seconds=1))
    assert at_b["DGS10"]["sha256"] == b["sha256"]


@pytest.mark.parametrize(
    "invalid_digest",
    [
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "z" * 64,
        "a" * 32 + " " + "a" * 31,
    ],
)
def test_raw_origin_requires_canonical_sha256(
    tmp_path: Path, invalid_digest: str
) -> None:
    store = CrossAssetStore(tmp_path)
    record = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/a.csv",
        "sha256": invalid_digest,
    }
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="raw provenance"):
        store.read_manifest()


def test_legacy_duplicate_ok_reversed_order_still_uses_true_first_capture(
    tmp_path: Path,
) -> None:
    early = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/a.csv",
        "sha256": "a" * 64,
    }
    late = {**early, "fetched_at": "2025-07-11T12:00:00Z"}
    unchanged = {
        **early,
        "status": "unchanged",
        "fetched_at": "2025-07-12T12:00:00Z",
    }
    # The earlier ok row occurs *after* the later ok row in the manifest.
    store = CrossAssetStore(tmp_path)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in (late, early, unchanged)
        ),
        encoding="utf-8",
    )
    resolved = store.read_manifest()
    assert all(row["captured_at"] == early["fetched_at"] for row in resolved)


@pytest.mark.parametrize("bad_key", ["", "  "])
def test_usable_manifest_rejects_blank_series_key(tmp_path: Path, bad_key: str) -> None:
    store = CrossAssetStore(tmp_path)
    record = {
        "schema": "cross_asset_manifest_v1",
        "key": bad_key,
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/raw.csv",
        "sha256": "a" * 64,
    }
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw provenance"):
        store.read_manifest()


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value"),
    [
        ("key", " DGS10 "),
        ("key", "DGS10 "),
        ("path", " "),
        ("path", " raw/fred/DGS10/a.csv"),
        ("sha256", " "),
        ("sha256", " " + "a" * 64),
    ],
)
def test_usable_manifest_rejects_noncanonical_whitespace_in_raw_provenance(
    tmp_path: Path, invalid_field: str, invalid_value: str
) -> None:
    record = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/a.csv",
        "sha256": "a" * 64,
    }
    record[invalid_field] = invalid_value
    store = CrossAssetStore(tmp_path)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="raw provenance"):
        store.read_manifest()


def test_unchanged_recheck_cannot_claim_a_future_original_capture(
    tmp_path: Path,
) -> None:
    """An ok row written after an unchanged receipt cannot explain that receipt."""

    origin = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-12T12:00:00Z",
        "path": "raw/fred/DGS10/original.csv",
        "sha256": "a" * 64,
    }
    premature = {
        **origin,
        "status": "unchanged",
        "fetched_at": "2025-07-11T12:00:00Z",
    }
    store = CrossAssetStore(tmp_path)
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in (premature, origin)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="capture origin.*after"):
        store.read_manifest()


def test_legacy_first_capture_is_independent_of_duplicate_ok_record_order(
    tmp_path: Path,
) -> None:
    early = {
        "schema": "cross_asset_manifest_v1",
        "key": "DGS10",
        "status": "ok",
        "fetched_at": "2025-07-10T12:00:00Z",
        "path": "raw/fred/DGS10/a.csv",
        "sha256": "a" * 64,
    }
    late = {**early, "fetched_at": "2025-07-11T12:00:00Z"}
    unchanged = {
        **early,
        "status": "unchanged",
        "fetched_at": "2025-07-12T12:00:00Z",
    }
    store = CrossAssetStore(tmp_path)
    # Earliest first, latest second: a naive last-write-wins origin map is wrong.
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in (early, late, unchanged)),
        encoding="utf-8",
    )
    resolved = store.read_manifest()
    assert all(row["captured_at"] == early["fetched_at"] for row in resolved)
