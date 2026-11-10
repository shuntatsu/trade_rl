"""All captured latest-vintage series remain retrospective, even when revisions are rare."""

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
from trade_rl.integrations.cross_asset.catalog import series_by_key
from trade_rl.integrations.cross_asset.dataset import with_cross_asset_global_features
from trade_rl.integrations.cross_asset.fetch import fetch_all
from trade_rl.integrations.cross_asset.loader import load_cross_asset_features


def test_even_low_revision_risk_needs_explicit_latest_vintage_ack(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    dataset = _dataset()
    frame = load_cross_asset_features(store.root, dataset.timestamps)
    assert series_by_key("VIXCLS").revision_risk == "low"

    snapshot_id = dataset.dataset_id
    snapshot_identity = dataset.identity_payload_json
    snapshot_features = dataset.global_features.copy()
    for acknowledgement in (False, None, 1, "True"):
        with pytest.raises(ValueError, match="accept_revision_hindsight"):
            with_cross_asset_global_features(
                dataset,
                frame,
                feature_names=("VIXCLS__ret1",),
                accept_revision_hindsight=acknowledgement,  # type: ignore[arg-type]
            )
        assert dataset.dataset_id == snapshot_id
        assert dataset.identity_payload_json == snapshot_identity
        np.testing.assert_array_equal(dataset.global_features, snapshot_features)

    accepted = with_cross_asset_global_features(
        dataset,
        frame,
        feature_names=("VIXCLS__ret1",),
        accept_revision_hindsight=True,
    )
    identity = json.loads(accepted.identity_payload_json or "{}")
    info = identity["cross_asset_overlay"]
    assert info["strict_point_in_time_verified"] is False
    assert info["latest_vintage_series"] == ["VIXCLS"]
    assert info["revision_hindsight_series"] == []  # low risk is not zero risk
    assert info["accept_revision_hindsight"] is True
    assert accepted.identity_verified


def test_low_risk_later_revision_cannot_pass_as_unacknowledged_pit(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    dataset = _dataset()
    first = load_cross_asset_features(store.root, dataset.timestamps)
    days = weekdays(420)
    revised = random_walk(420, seed=44)
    revised[355:385] *= np.linspace(1.0, 1.6, 30)
    fetch_all(
        store,
        (series_by_key("VIXCLS"),),
        FakeTransport({"id=VIXCLS": fred_payload("VIXCLS", days, revised)}),
        now=clock(FETCHED_AT + timedelta(days=3)),
        sleep=lambda _s: None,
    )
    later = load_cross_asset_features(store.root, dataset.timestamps)
    col = first.column("VIXCLS__ret1")
    assert not np.array_equal(
        first.values[:, col], later.values[:, col], equal_nan=True
    )
    for frame in (first, later):
        with pytest.raises(ValueError, match="accept_revision_hindsight"):
            with_cross_asset_global_features(
                dataset, frame, feature_names=("VIXCLS__ret1",)
            )


def test_even_unrevised_fx_composite_is_retrospective_by_default(
    tmp_path: Path,
) -> None:
    store = seed_store(tmp_path)
    dataset = _dataset()
    frame = load_cross_asset_features(store.root, dataset.timestamps)
    names = ("dollar_strength",)
    with pytest.raises(ValueError, match="accept_revision_hindsight"):
        with_cross_asset_global_features(dataset, frame, feature_names=names)
    accepted = with_cross_asset_global_features(
        dataset, frame, feature_names=names, accept_revision_hindsight=True
    )
    payload = json.loads(accepted.identity_payload_json or "{}")["cross_asset_overlay"]
    expected_members = set(frame.column_sources[frame.column("dollar_strength")])
    assert expected_members == {"DEXJPUS", "DEXCHUS", "DEXUSEU", "DEXUSUK", "DEXUSAL"}
    assert payload["schema"] == "cross_asset_global_overlay_v3"
    assert payload["strict_point_in_time_verified"] is False
    assert payload["latest_vintage_series"] == sorted(expected_members)
    assert payload["revision_hindsight_series"] == []
    assert payload["accept_revision_hindsight"] is True
    np.testing.assert_array_equal(
        accepted.global_features[:, -1],
        np.where(
            frame.available[:, frame.column("dollar_strength")],
            frame.values[:, frame.column("dollar_strength")],
            0.0,
        ).astype(np.float32),
    )
    assert accepted.identity_verified
