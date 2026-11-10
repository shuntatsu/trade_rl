"""Point-in-time observation series: values stamped with when they became known."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from trade_rl.integrations.cross_asset.catalog import SeriesSpec, series_by_key
from trade_rl.integrations.cross_asset.parsers import PARSERS, ParsedSeries
from trade_rl.integrations.cross_asset.pit import (
    knowable_at_array,
    session_close_array,
)
from trade_rl.integrations.cross_asset.store import CrossAssetStore, parse_utc


@dataclass(frozen=True, slots=True)
class ObsSeries:
    """Daily values with session-close and release times in UTC nanoseconds.

    ``knowable_ns`` is non-decreasing. Rows whose session had not closed when the
    payload was fetched are never present: an in-progress bar is not an
    observation.
    """

    spec: SeriesSpec
    dates: np.ndarray  # datetime64[D]
    values: np.ndarray  # float64
    close_ns: np.ndarray  # int64, UTC
    knowable_ns: np.ndarray  # int64, UTC
    sha256: str

    @property
    def key(self) -> str:
        return self.spec.key


def build_obs_series(
    spec: SeriesSpec,
    parsed: ParsedSeries,
    *,
    sha256: str = "",
    fetched_at: datetime | None = None,
) -> ObsSeries:
    dates = parsed.dates
    values = parsed.values
    closes = session_close_array(
        dates, venue_tz=spec.venue_tz, close_local=spec.close_local
    )
    if fetched_at is not None:
        fetched = np.datetime64(fetched_at.astimezone(UTC).replace(tzinfo=None), "ns")
        closed = closes <= fetched
        dates, values, closes = dates[closed], values[closed], closes[closed]
    knowable = knowable_at_array(
        dates, venue_tz=spec.venue_tz, close_local=spec.close_local, rule=spec.release
    )
    return ObsSeries(
        spec=spec,
        dates=dates,
        values=np.asarray(values, dtype=np.float64),
        close_ns=closes.astype(np.int64),
        knowable_ns=knowable.astype(np.int64),
        sha256=sha256,
    )


def load_obs_series(
    store: CrossAssetStore, latest: Mapping[str, Mapping[str, object]]
) -> dict[str, ObsSeries]:
    """Parse the selected raw vintages. Unknown catalog keys are skipped."""

    result: dict[str, ObsSeries] = {}
    for key, record in latest.items():
        try:
            spec = series_by_key(key)
        except KeyError:
            continue
        if not store.verify_raw(record):
            raise ValueError(f"raw file for {key} is missing or fails its digest")
        _name, parser = PARSERS[spec.source]
        parsed = parser(store.read_raw(record), spec.symbol)
        result[key] = build_obs_series(
            spec,
            parsed,
            sha256=str(record["sha256"]),
            fetched_at=parse_utc(str(record["fetched_at"])),
        )
    return result


__all__ = ["ObsSeries", "build_obs_series", "load_obs_series"]
