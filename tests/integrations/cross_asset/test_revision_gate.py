"""Revision look-ahead fences: per-series gating, overlay refusal, legacy rows."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.integrations.cross_asset.support import (
    FETCHED_AT,
    FakeTransport,
    clock,
    fred_payload,
    random_walk,
    seed_store,
    weekdays,
)
from tests.integrations.cross_asset.test_dataset_overlay import _dataset
from trade_rl.integrations.cross_asset.catalog import CATALOG, series_by_key
from trade_rl.integrations.cross_asset.dataset import with_cross_asset_global_features
from trade_rl.integrations.cross_asset.features import FeatureFrame
from trade_rl.integrations.cross_asset.fetch import fetch_all
from trade_rl.integrations.cross_asset.loader import load_cross_asset_features
from trade_rl.integrations.cross_asset.observations import load_obs_series
from trade_rl.integrations.cross_asset.store import (
    CrossAssetStore,
    latest_records,
)

NON_LOW = tuple(spec.key for spec in CATALOG if not spec.default_member)
GRID = _dataset().timestamps


def _prefixes(frame: FeatureFrame) -> set[str]:
    return {name.split("__")[0] for name in frame.names if "__" in name}


def test_default_frame_has_no_columns_from_series_that_are_not_low(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    frame = load_cross_asset_features(store.root, GRID)
    assert _prefixes(frame).isdisjoint(NON_LOW)
    assert {"DEXJPUS", "T10Y2Y", "VIXCLS", "NIKKEI225"} <= _prefixes(frame)
    assert frame.opt_in_series == ()
    assert set(frame.series_present).isdisjoint(NON_LOW)
    assert len(frame.column_sources) == len(frame.names)
    used = {key for sources in frame.column_sources for key in sources}
    assert used.isdisjoint(NON_LOW)


def test_opting_in_adds_exactly_that_series_and_records_it(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    default = load_cross_asset_features(store.root, GRID)
    opted = load_cross_asset_features(store.root, GRID, opt_in_series=("SP500",))
    assert _prefixes(opted) - _prefixes(default) == {"SP500"}
    assert opted.opt_in_series == ("SP500",)
    assert "SP500" in opted.series_present
    column = opted.column("SP500__ret1")
    assert opted.column_sources[column] == ("SP500",)


def test_composite_sources_name_the_series_actually_used(tmp_path: Path) -> None:
    store = seed_store(tmp_path)
    default = load_cross_asset_features(store.root, GRID)
    sources = default.column_sources[default.column("dollar_strength")]
    assert set(sources) == {"DEXJPUS", "DEXCHUS", "DEXUSEU", "DEXUSUK", "DEXUSAL"}
    opted = load_cross_asset_features(
        store.root, GRID, opt_in_series=("DTWEXBGS", "SP500", "NASDAQCOM")
    )
    assert "DTWEXBGS" in opted.column_sources[opted.column("dollar_strength")]
    equity = opted.column_sources[opted.column("equity_momentum20")]
    assert {"SP500", "NASDAQCOM"} <= set(equity)
    rotation = opted.column_sources[opted.column("rotation_equity_minus_dollar")]
    assert set(rotation) >= set(equity)


def _hand_frame(
    names: tuple[str, ...],
    sources: tuple[tuple[str, ...], ...],
    opt_in: tuple[str, ...] = (),
) -> FeatureFrame:
    n = GRID.size
    return FeatureFrame(
        feature_set="cross_asset_features_v1",
        definitions_digest="d" * 64,
        input_digest="i" * 64,
        names=names,
        decision_ns=GRID.astype("datetime64[ns]").astype(np.int64),
        values=np.ones((n, len(names))),
        available=np.ones((n, len(names)), dtype=np.bool_),
        staleness_hours=np.ones((n, len(names))),
        series_present=("SP500",),
        opt_in_series=opt_in,
        column_sources=sources,
    )


def test_overlay_refuses_a_non_low_series_that_was_not_opted_in() -> None:
    frame = _hand_frame(("SP500__ret1",), (("SP500",),))
    with pytest.raises(ValueError, match="not opted in"):
        with_cross_asset_global_features(
            _dataset(),
            frame,
            feature_names=("SP500__ret1",),
            accept_revision_hindsight=True,
        )


def test_overlay_refuses_a_composite_built_from_a_non_opted_series() -> None:
    frame = _hand_frame(("equity_momentum20",), (("NASDAQCOM", "NIKKEI225"),))
    with pytest.raises(ValueError, match="not opted in"):
        with_cross_asset_global_features(
            _dataset(), frame, feature_names=("equity_momentum20",)
        )


def test_overlay_refuses_frames_without_source_provenance() -> None:
    frame = _hand_frame(("DEXJPUS__ret1",), ())
    with pytest.raises(ValueError, match="provenance"):
        with_cross_asset_global_features(
            _dataset(), frame, feature_names=("DEXJPUS__ret1",)
        )
    unknown = _hand_frame(("ZZZ__ret1",), (("ZZZ",),))
    with pytest.raises(ValueError, match="unknown"):
        with_cross_asset_global_features(
            _dataset(), unknown, feature_names=("ZZZ__ret1",)
        )


def test_opted_in_revised_series_need_an_explicit_hindsight_acceptance(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    frame = load_cross_asset_features(
        store.root, GRID, opt_in_series=("SP500", "NASDAQCOM")
    )
    base = _dataset()
    for names in (("SP500__ret1",), ("equity_momentum20",), ("NASDAQCOM__sm20",)):
        with pytest.raises(ValueError, match="accept_revision_hindsight"):
            with_cross_asset_global_features(base, frame, feature_names=names)
    accepted = with_cross_asset_global_features(
        base, frame, feature_names=("SP500__ret1",), accept_revision_hindsight=True
    )
    payload = json.loads(accepted.identity_payload_json or "{}")["cross_asset_overlay"]
    assert payload["schema"] == "cross_asset_global_overlay_v3"
    assert payload["strict_point_in_time_verified"] is False
    assert payload["latest_vintage_series"] == ["SP500"]
    assert payload["accept_revision_hindsight"] is True
    assert payload["revision_hindsight_series"] == ["SP500"]
    assert payload["opt_in_series"] == ["NASDAQCOM", "SP500"]
    assert payload["vintage_basis"] == "latest_vintage_at_capture_not_first_release"
    assert accepted.identity_verified


def test_low_risk_features_need_retrospective_acceptance_and_are_recorded_as_such(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    frame = load_cross_asset_features(store.root, GRID)
    names = ("dollar_strength", "DEXJPUS__ret5")
    with pytest.raises(ValueError, match="accept_revision_hindsight"):
        with_cross_asset_global_features(_dataset(), frame, feature_names=names)
    combined = with_cross_asset_global_features(
        _dataset(), frame, feature_names=names, accept_revision_hindsight=True
    )
    payload = json.loads(combined.identity_payload_json or "{}")["cross_asset_overlay"]
    assert payload["schema"] == "cross_asset_global_overlay_v3"
    assert payload["accept_revision_hindsight"] is True
    assert payload["strict_point_in_time_verified"] is False
    assert "DEXJPUS" in payload["latest_vintage_series"]
    assert payload["revision_hindsight_series"] == []
    assert payload["vintage_basis"] == "latest_vintage_at_capture_not_first_release"


def test_a_later_revision_of_a_non_low_series_cannot_reach_a_default_dataset(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    names = ("dollar_strength", "curve_slope", "VIXCLS__ret1")
    before = load_cross_asset_features(store.root, GRID)
    dataset_before = with_cross_asset_global_features(
        _dataset(), before, feature_names=names, accept_revision_hindsight=True
    )
    days = weekdays(420)
    revised = random_walk(420, seed=0)
    revised[355:385] *= np.linspace(1.0, 1.6, 30)  # around the June 2024 grid
    fetch_all(
        store,
        (series_by_key("SP500"),),
        FakeTransport({"id=SP500": fred_payload("SP500", days, revised)}),
        now=clock(FETCHED_AT + timedelta(days=3)),
        sleep=lambda _s: None,
    )
    after = load_cross_asset_features(store.root, GRID)
    assert before.names == after.names
    assert np.array_equal(before.values, after.values, equal_nan=True)
    dataset_after = with_cross_asset_global_features(
        _dataset(), after, feature_names=names, accept_revision_hindsight=True
    )
    assert dataset_after.dataset_id != dataset_before.dataset_id  # input digest moved
    assert np.array_equal(dataset_after.global_features, dataset_before.global_features)
    # The revised series is reachable only through an explicit, recorded opt-in.
    opted_before = load_cross_asset_features(
        store.root,
        GRID,
        opt_in_series=("SP500",),
        as_of=FETCHED_AT + timedelta(hours=1),
    )
    opted_after = load_cross_asset_features(store.root, GRID, opt_in_series=("SP500",))
    column = opted_after.column("SP500__ret1")
    assert not np.array_equal(
        opted_before.values[:, column], opted_after.values[:, column], equal_nan=True
    )
    with pytest.raises(ValueError, match="accept_revision_hindsight"):
        with_cross_asset_global_features(
            _dataset(), opted_after, feature_names=("SP500__ret1",)
        )


def _legacy_manifest(root: Path, *, with_origin: bool) -> CrossAssetStore:
    """A manifest written before ``captured_at`` existed (re-check restamped)."""

    store = seed_store(root)
    records = store.read_manifest()
    legacy = []
    for record in records:
        record = {k: v for k, v in record.items() if k != "captured_at"}
        legacy.append(record)
    key = "DEXJPUS"
    origin = next(r for r in legacy if r["key"] == key)
    recheck = dict(origin)
    recheck.update(status="unchanged", fetched_at="2025-07-20T12:00:00Z")
    if not with_origin:
        legacy = [r for r in legacy if r["key"] != key]
        recheck["path"] = "raw/fred/DEXJPUS/missing.csv"
    legacy.append(recheck)
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in legacy
        ),
        encoding="utf-8",
    )
    return CrossAssetStore(root)


def test_legacy_unchanged_rows_resolve_to_the_origin_capture_time(
    tmp_path: Path,
) -> None:
    store = _legacy_manifest(tmp_path, with_origin=True)
    records = store.read_manifest()
    rechecks = [r for r in records if r["status"] == "unchanged"]
    assert len(rechecks) == 1
    origin = next(r for r in records if r["key"] == "DEXJPUS" and r["status"] == "ok")
    assert rechecks[0]["fetched_at"] == "2025-07-20T12:00:00Z"
    assert rechecks[0]["captured_at"] == origin["fetched_at"]  # not the re-check
    latest = latest_records(records)
    assert latest["DEXJPUS"]["captured_at"] == origin["fetched_at"]
    # The in-progress filter therefore uses the original capture time.
    obs = load_obs_series(store, {"DEXJPUS": latest["DEXJPUS"]})
    assert obs["DEXJPUS"].dates.size > 0


def test_legacy_unchanged_rows_without_an_origin_fail_closed(tmp_path: Path) -> None:
    store = _legacy_manifest(tmp_path, with_origin=False)
    with pytest.raises(ValueError, match="captured_at"):
        store.read_manifest()
