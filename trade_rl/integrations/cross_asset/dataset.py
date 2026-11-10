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
from trade_rl.integrations.cross_asset.catalog import series_by_key
from trade_rl.integrations.cross_asset.features import FeatureFrame
from trade_rl.integrations.cross_asset.normalization import FeatureNormalizer

GLOBAL_PREFIX = "xa_"
MISSING_REASON_CROSS_ASSET = 2
OVERLAY_SCHEMA = "cross_asset_global_overlay_v2"
VINTAGE_BASIS = "latest_vintage_at_capture_not_first_release"


def _hindsight_series(chosen: FeatureFrame, opted: tuple[str, ...]) -> set[str]:
    """Series behind the chosen columns that are not low revision risk.

    Raises when provenance is missing, a series is unknown, or a non-low series
    was not opted in on the frame.
    """

    if len(chosen.column_sources) != len(chosen.names):
        raise ValueError("frame has no source provenance for its columns")
    hindsight: set[str] = set()
    for name, sources in zip(chosen.names, chosen.column_sources, strict=True):
        if not sources:
            raise ValueError(f"feature {name} has no source provenance")
        for key in sources:
            try:
                spec = series_by_key(key)
            except KeyError as error:
                raise ValueError(f"feature {name} uses unknown series {key}") from error
            if spec.default_member:
                continue
            if key not in opted:
                raise ValueError(
                    f"feature {name} uses {key} (revision risk {spec.revision_risk}) "
                    "which was not opted in"
                )
            hindsight.add(key)
    return hindsight


def with_cross_asset_global_features(
    dataset: MarketDataset,
    frame: FeatureFrame,
    *,
    feature_names: Sequence[str],
    normalizer: FeatureNormalizer | None = None,
    accept_revision_hindsight: bool = False,
) -> MarketDataset:
    """Return ``dataset`` with ``xa_<name>`` columns appended to global features.

    Fail-closed fences: every selected column must carry source provenance;
    a column fed by a series whose revision risk is not ``low`` is refused unless
    that series was opted in on the frame, and even then only with
    ``accept_revision_hindsight=True`` because values are the newest vintage, not
    first releases. The flag, the opt-in set and the hindsight series are recorded
    in the dataset identity.

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
    hindsight = _hindsight_series(chosen, frame.opt_in_series)
    if hindsight and not accept_revision_hindsight:
        raise ValueError(
            "features use series with revision hindsight "
            f"{sorted(hindsight)}; pass accept_revision_hindsight=True to accept "
            "that they carry newest-vintage values"
        )
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
        "vintage_basis": VINTAGE_BASIS,
        "accept_revision_hindsight": bool(accept_revision_hindsight),
        "revision_hindsight_series": sorted(hindsight),
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
    "VINTAGE_BASIS",
    "with_cross_asset_global_features",
]
