"""Causal ADX seed admission, publication and legacy identity compatibility."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from tests.data.test_long_native_alignment import _source
from trade_rl.data.build.config import load_market_build_request
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features import calculate_feature_events
from trade_rl.data.features import multitimeframe as mtf


def _events(spec, close, present):
    n = len(close)
    return calculate_feature_events(
        spec,
        open_price=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.ones(n),
        funding_rate=np.zeros(n),
        funding_available=np.zeros(n, dtype=bool),
        row_present=present,
        active=np.ones(n, dtype=bool),
    )


@pytest.mark.parametrize("period", [1, 2, 5, 14, 52])
@pytest.mark.parametrize("direction", [1, -1])
def test_causal_adx_seed_does_not_need_a_future_row(period, direction):
    seed_rows = 2 * period - 1
    close = 1000.0 + direction * np.arange(seed_rows + 2, dtype=float)
    present = np.ones(len(close), dtype=bool)
    spec = FeatureSpec("adx", FeatureKind.ADX_CAUSAL, lookback=period)
    legacy = replace(spec, kind=FeatureKind.ADX)
    values, valid, starts = _events(spec, close[:seed_rows], present[:seed_rows])
    assert valid.tolist() == [False] * (seed_rows - 1) + [True]
    assert starts.tolist() == [-1] * (seed_rows - 1) + [0]
    # A one-row Wilder seed has zero directional movement. Larger monotone
    # seeds have only one nonzero direction and therefore DX/100 exactly one.
    assert values[-1] == (0.0 if period == 1 else 1.0)
    # #1570 pinned the legacy kind to an extra-row admission here. Per the
    # owner decision of 2026-11-28 (all research is reset; no legacy-compatible
    # mode) the legacy kind now uses the same causal 2p-1 seed.
    legacy_seed = _events(legacy, close[:seed_rows], present[:seed_rows])
    for expected, actual in zip((values, valid, starts), legacy_seed, strict=True):
        np.testing.assert_array_equal(expected, actual)
    full = _events(spec, close, present)
    for expected, actual in zip((values, valid, starts), full, strict=True):
        np.testing.assert_array_equal(expected, actual[:seed_rows])
    # Complete long legacy segments keep their exact historical arithmetic.
    for expected, actual in zip(_events(legacy, close, present), full, strict=True):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize("period", [1, 2, 5, 14, 52])
def test_causal_adx_every_prefix_and_future_mask_mutation(period):
    rng = np.random.default_rng(period)
    n = 6 * period + 8
    close = 100.0 + np.cumsum(rng.uniform(-1.0, 1.0, n))
    present = np.ones(n, dtype=bool)
    present[2 * period + 1] = False
    spec = FeatureSpec("adx", FeatureKind.ADX_CAUSAL, lookback=period)
    full = _events(spec, close, present)
    for end in range(n + 1):
        changed = close.copy()
        changed[end:] *= 2.0
        changed_present = present.copy()
        changed_present[end:] = False
        prefix = _events(spec, close[:end], present[:end])
        mutated = _events(spec, changed, changed_present)
        for original, short, alternate in zip(full, prefix, mutated, strict=True):
            np.testing.assert_array_equal(original[:end], short)
            np.testing.assert_array_equal(original[:end], alternate[:end])


def test_causal_adx_publication_binds_all_rows_after_each_gap():
    times = np.datetime64("2024-01-02", "ns") + np.arange(12) * np.timedelta64(1, "h")
    receipts = times.copy()
    receipts[1] = times[4] + np.timedelta64(1, "ns")
    raw = _source(times, receipts)
    tradable = raw.tradable.copy()
    tradable[5] = False
    raw = replace(raw, tradable=tradable)
    spec = FeatureSpec(
        "adx", FeatureKind.ADX_CAUSAL, lookback=2, max_staleness_hours=24.0
    )
    assert spec.requires_complete_native_rows
    contract = InstrumentContract("SYNTH")
    _, valid, published = mtf._native_events(spec, raw, contract)
    assert np.flatnonzero(valid).tolist() == [2, 3, 4, 8, 9, 10, 11]
    for t in np.flatnonzero(valid):
        start = 0 if t < 5 else 6
        assert published[t] == max(receipts[start : t + 1])
    _, available, _, _ = mtf.align_native_feature(
        spec, raw, contract, times, np.ones(len(times), dtype=bool), timeframe="1h"
    )
    assert not available[:5].any()
    assert available[5]
    assert available[8:].all()


@pytest.mark.parametrize("lookback", [True, 0, -1, 2.0, np.int64(2)])
def test_causal_adx_requires_positive_native_lookback(lookback):
    with pytest.raises(ValueError):
        FeatureSpec("adx", FeatureKind.ADX_CAUSAL, lookback=lookback)


def test_causal_adx_config_roundtrip_binds_distinct_kind(tmp_path):
    spec = FeatureSpec("adx", FeatureKind.ADX_CAUSAL, lookback=14)
    legacy = replace(spec, kind=FeatureKind.ADX)
    expected = legacy.canonical_payload() | {"kind": "adx_causal_v1"}
    assert spec.canonical_payload() == expected
    assert legacy.kind.value == "adx"
    assert spec.kind.value == "adx_causal_v1"
    assert not legacy.requires_complete_native_rows
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps(
            {
                "source_root": ".",
                "base_timeframe": "1h",
                "features": [spec.canonical_payload()],
                "instruments": [
                    {"symbol": "SYNTH", "listed_at": "2024-01-01T00:00:00Z"}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_market_build_request(path).config.features == (spec,)


def test_causal_adx_native_dataset_identity_and_artifact_roundtrip(tmp_path):
    from tests.data.test_native_spot_builder import _fixture
    from trade_rl.data.artifacts.codec import (
        load_dataset_files,
        write_market_dataset_files,
    )
    from trade_rl.data.build.native_spot import build_native_fixed_spot_dataset

    args, _ = _fixture(7)
    group = args["feature_groups"][0]
    spec = FeatureSpec(
        "adx", FeatureKind.ADX_CAUSAL, lookback=2, max_staleness_hours=200.0
    )

    def build(feature):
        return build_native_fixed_spot_dataset(
            **(args | {"feature_groups": (replace(group, specs=(feature,)),)})
        )

    dataset = build(spec)
    legacy = build(replace(spec, kind=FeatureKind.ADX))
    assert dataset.feature_available.all()
    np.testing.assert_array_equal(dataset.features, legacy.features)
    np.testing.assert_array_equal(dataset.feature_available, legacy.feature_available)
    assert dataset.feature_config_digest != legacy.feature_config_digest
    assert dataset.dataset_id != legacy.dataset_id
    write_market_dataset_files(tmp_path / "artifact", dataset)
    restored = load_dataset_files(tmp_path / "artifact")
    assert restored.identity_verified and restored.dataset_id == dataset.dataset_id
    np.testing.assert_array_equal(restored.features, dataset.features)
