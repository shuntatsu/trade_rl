"""Opt-in hook that appends cross-asset features to a dataset's global features.

``MarketDataset.global_features`` already reaches the PPO observation (with an
availability mask) and the portfolio transform fits its own scale on the training
window, so the overlay needs no change to shared feature code: it returns a new
dataset whose first ``g`` global columns are byte-identical to the input and
whose identity binds the overlay definition and input digests. Callers that never
call this function get exactly the datasets they had before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.cross_asset.features import FeatureFrame
from trade_rl.integrations.cross_asset.normalization import FeatureNormalizer

GLOBAL_PREFIX = "xa_"
MISSING_REASON_CROSS_ASSET = 2
OVERLAY_SCHEMA = "cross_asset_global_overlay_v1"


def with_cross_asset_global_features(
    dataset: MarketDataset,
    frame: FeatureFrame,
    *,
    feature_names: Sequence[str],
    normalizer: FeatureNormalizer | None = None,
) -> MarketDataset:
    """Return ``dataset`` with ``xa_<name>`` columns appended to global features.

    ``frame`` must have been built on exactly ``dataset.timestamps``. Unavailable
    entries are stored as 0.0 with ``global_feature_available=False`` so the
    policy sees an explicit mask rather than a fabricated value.
    """

    names = tuple(feature_names)
    if not names:
        raise ValueError("select at least one cross-asset feature")
    if len(set(names)) != len(names):
        raise ValueError("cross-asset feature names must be unique")
    expected = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    if not np.array_equal(frame.decision_ns, expected):
        raise ValueError("frame decision times must equal the dataset timestamps")
    source = frame if normalizer is None else normalizer.transform(frame)
    chosen = source.select(names)
    added_names = tuple(f"{GLOBAL_PREFIX}{name}" for name in names)
    collisions = set(added_names) & set(dataset.global_feature_names)
    if collisions:
        raise ValueError(f"global feature names already present: {sorted(collisions)}")

    usable = chosen.available
    values = np.where(usable, chosen.values, 0.0).astype(np.float32)
    staleness = np.where(usable, chosen.staleness_hours, 0.0).astype(np.float32)
    reasons = np.where(usable, 0, MISSING_REASON_CROSS_ASSET).astype(np.int16)

    identity = (
        parse_identity_json(dataset.identity_payload_json)
        if dataset.identity_payload_json is not None
        else {}
    )
    identity["cross_asset_overlay"] = {
        "schema": OVERLAY_SCHEMA,
        "feature_set": frame.feature_set,
        "definitions_digest": frame.definitions_digest,
        "input_digest": frame.input_digest,
        "names": list(names),
        "normalizer_digest": None if normalizer is None else normalizer.digest(),
        "series_present": list(frame.series_present),
        "opt_in_series": list(frame.opt_in_series),
    }
    combined = replace(
        dataset,
        dataset_id="0" * 64,
        identity_payload_json=None,
        global_features=np.concatenate([dataset.global_features, values], axis=1),
        global_feature_names=(*dataset.global_feature_names, *added_names),
        global_feature_available=np.concatenate(
            [dataset.resolved_array("global_feature_available"), usable], axis=1
        ),
        global_feature_staleness_hours=np.concatenate(
            [dataset.resolved_array("global_feature_staleness_hours"), staleness],
            axis=1,
        ),
        global_feature_missing_reason=np.concatenate(
            [dataset.resolved_array("global_feature_missing_reason"), reasons], axis=1
        ),
    )
    return combined.with_content_identity(identity)


__all__ = [
    "GLOBAL_PREFIX",
    "MISSING_REASON_CROSS_ASSET",
    "with_cross_asset_global_features",
]
