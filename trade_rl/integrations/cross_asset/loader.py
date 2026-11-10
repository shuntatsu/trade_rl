"""Load cross-asset features aligned to an arbitrary UTC decision grid."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from trade_rl.data.contracts import timeframe_hours
from trade_rl.integrations.cross_asset.features import FeatureFrame, compute_frame
from trade_rl.integrations.cross_asset.observations import load_obs_series
from trade_rl.integrations.cross_asset.store import (
    CrossAssetStore,
    latest_records,
    manifest_digest,
)

DEFAULT_ROOT = Path("data") / "cross_asset"


def regular_grid(start: datetime, end: datetime, timeframe: str) -> np.ndarray:
    """Bar-close timestamps ``start, start+step, ...`` strictly before ``end``."""

    for value in (start, end):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("grid bounds must be timezone-aware")
    step = np.timedelta64(int(round(timeframe_hours(timeframe) * 3600)), "s")
    first = np.datetime64(start.astimezone(UTC).replace(tzinfo=None), "ns")
    last = np.datetime64(end.astimezone(UTC).replace(tzinfo=None), "ns")
    if first >= last:
        raise ValueError("grid start must be before end")
    return np.arange(first, last, step.astype("timedelta64[ns]"))


def load_cross_asset_features(
    root: str | Path,
    decision_times: np.ndarray,
    *,
    as_of: datetime | None = None,
    series: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    opt_in_series: Sequence[str] = (),
) -> FeatureFrame:
    """Features at ``decision_times`` using only data released by each time.

    ``decision_times`` are UTC instants (for example crypto bar close times). A
    value is usable at ``t`` only if its release time is ``<= t``. ``as_of``
    restricts the raw vintages to those captured at or before it, which makes a
    build reproducible after later refetches or revisions. Composites use only
    default (``revision_risk == "low"``) series unless ``opt_in_series`` names
    others; the frame records the opt-in set.
    """

    store = CrossAssetStore(root)
    latest = latest_records(store.read_manifest(), as_of=as_of)
    if series is not None:
        wanted = set(series)
        latest = {key: record for key, record in latest.items() if key in wanted}
    if not latest:
        raise FileNotFoundError(
            "no usable cross-asset vintages; run `python -m "
            "trade_rl.integrations.cross_asset fetch`"
        )
    obs = load_obs_series(store, latest)
    frame = compute_frame(
        obs,
        decision_times,
        input_digest=manifest_digest(latest),
        opt_in_series=opt_in_series,
    )
    return frame if feature_names is None else frame.select(feature_names)


__all__ = ["DEFAULT_ROOT", "load_cross_asset_features", "regular_grid"]
