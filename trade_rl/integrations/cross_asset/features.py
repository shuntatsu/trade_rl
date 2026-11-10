"""cross_asset_features_v1: trailing-window features and cross-asset composites.

Every per-series feature at observation ``i`` uses observations ``<= i`` only and
inherits that observation's release time. Features are then joined to the
decision grid as-of (``knowable_at <= t``) with an explicit staleness. Composites
are stateless combinations of the aligned per-series features, so they cannot
introduce any look-ahead beyond their inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.cross_asset.catalog import (
    CATALOG,
    SeriesKind,
    SeriesSpec,
)
from trade_rl.integrations.cross_asset.observations import ObsSeries
from trade_rl.integrations.cross_asset.pit import align_asof

FEATURE_SET_ID = "cross_asset_features_v1"

CHANGE_WINDOWS = (1, 5, 20, 60)
VOL_WINDOW = 20
SCALE_WINDOW = 252  # trailing window for vol scaling and level z-scores
SCALE_MIN_PERIODS = 63
MA_WINDOWS = (60, 200)
SCALED_MOMENTUM_CLIP = 5.0
_EPS = 1e-12
_NS_PER_HOUR = 3_600_000_000_000.0
_CHUNK_ROWS = 2048


@dataclass(frozen=True, slots=True)
class Composite:
    """Mean of sign-adjusted aligned per-series features."""

    name: str
    members: tuple[tuple[str, str, int], ...]  # (series key, suffix, sign)
    min_members: int

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "members": [list(member) for member in self.members],
            "min_members": self.min_members,
        }


_EQUITIES = ("SP500", "NASDAQCOM", "NIKKEI225", "y_gdaxi", "y_ftse", "y_hsi", "y_ssec")
_COMMODITIES = ("y_wti", "y_brent", "y_gold", "y_silver", "y_copper")

COMPOSITES: tuple[Composite, ...] = (
    Composite(
        "dollar_strength",
        (
            ("DTWEXBGS", "sm20", 1),
            ("DEXJPUS", "sm20", 1),
            ("DEXCHUS", "sm20", 1),
            ("DEXUSEU", "sm20", -1),
            ("DEXUSUK", "sm20", -1),
            ("DEXUSAL", "sm20", -1),
            ("y_dxy", "sm20", 1),
        ),
        3,
    ),
    Composite(
        "risk_appetite",
        (
            *((key, "sm20", 1) for key in _EQUITIES),
            ("VIXCLS", "sm20", -1),
            ("BAMLH0A0HYM2", "sm20", -1),
            ("BAMLC0A0CM", "sm20", -1),
            ("y_gold", "sm20", -1),
        ),
        4,
    ),
    Composite(
        "equity_momentum20",
        tuple((key, "sm20", 1) for key in _EQUITIES),
        3,
    ),
    Composite(
        "commodity_momentum20",
        tuple((key, "sm20", 1) for key in _COMMODITIES),
        2,
    ),
    Composite(
        "commodity_momentum60",
        tuple((key, "sm60", 1) for key in _COMMODITIES),
        2,
    ),
    Composite("curve_slope", (("T10Y2Y", "lvl", 1),), 1),
    Composite("curve_slope_chg20", (("T10Y2Y", "chg20", 1),), 1),
)
BREADTH_MIN_MEMBERS = 5
ROTATIONS: tuple[tuple[str, str, str], ...] = (
    ("rotation_equity_minus_commodity", "equity_momentum20", "commodity_momentum20"),
    ("rotation_equity_minus_dollar", "equity_momentum20", "dollar_strength"),
)
BREADTH_NAMES = ("trend_breadth_60", "trend_breadth_200")
COMPOSITE_FEATURE_NAMES: tuple[str, ...] = (
    "dollar_strength",
    "risk_appetite",
    "equity_momentum20",
    "commodity_momentum20",
    "commodity_momentum60",
    "curve_slope",
    "curve_slope_chg20",
    *BREADTH_NAMES,
    *(name for name, _a, _b in ROTATIONS),
)


def series_suffixes(kind: SeriesKind) -> tuple[str, ...]:
    change = "ret" if kind is SeriesKind.PRICE else "chg"
    names = [f"{change}{window}" for window in CHANGE_WINDOWS]
    names += [f"rvol{VOL_WINDOW}", f"z{SCALE_WINDOW}", "sm20", "sm60"]
    if kind is SeriesKind.LEVEL:
        names.append("lvl")
    else:
        names += [f"abv{window}" for window in MA_WINDOWS]
    return tuple(names)


def _breadth_members() -> tuple[str, ...]:
    return tuple(spec.key for spec in CATALOG if spec.trend_breadth)


def definitions() -> dict[str, object]:
    """The complete, canonical definition of the feature set."""

    return {
        "feature_set": FEATURE_SET_ID,
        "change_windows": list(CHANGE_WINDOWS),
        "vol_window": VOL_WINDOW,
        "scale_window": SCALE_WINDOW,
        "scale_min_periods": SCALE_MIN_PERIODS,
        "ma_windows": list(MA_WINDOWS),
        "scaled_momentum_clip": SCALED_MOMENTUM_CLIP,
        "window_unit": "observations of the series, not calendar days",
        "price_change": "log(x[i]) - log(x[i-k]); NaN if either value <= 0",
        "level_change": "x[i] - x[i-k]",
        "rvol": "sample std (ddof=1) of the 1-observation change over the window",
        "z": "(v[i] - mean) / std (ddof=1) of v over the trailing window including i;"
        " v = log price for prices and the level for levels",
        "sm": "k-observation change / (trailing 252 std of 1-obs change * sqrt(k)),"
        " clipped",
        "above_ma": "1 if x[i] > mean(x[i-W+1..i]) else 0 (prices only)",
        "series": [spec.canonical_payload() for spec in CATALOG],
        "composites": [composite.canonical_payload() for composite in COMPOSITES],
        "breadth": {
            "members": list(_breadth_members()),
            "min_members": BREADTH_MIN_MEMBERS,
        },
        "rotations": [list(rotation) for rotation in ROTATIONS],
        "staleness": "hours from the session close of the as-of observation",
        "alignment": "as-of join on knowable_at <= decision time, past values only",
    }


def definitions_digest() -> str:
    return content_digest(definitions())


# --------------------------------------------------------------------------- #
# Trailing-window primitives
# --------------------------------------------------------------------------- #


def _trailing_stats(
    values: np.ndarray, window: int, min_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing mean and sample std over ``window`` rows ending at each row.

    NaN entries are ignored; a row needs ``min_periods`` finite entries.
    """

    n = values.size
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    padded = np.concatenate([np.full(window - 1, np.nan), values])
    for start in range(0, n, _CHUNK_ROWS):
        stop = min(n, start + _CHUNK_ROWS)
        block = np.lib.stride_tricks.sliding_window_view(padded, window)[start:stop]
        finite = np.isfinite(block)
        count = finite.sum(axis=1)
        safe = np.where(count > 0, count, 1)
        centre = np.where(finite, block, 0.0).sum(axis=1) / safe
        deviation = np.where(finite, block - centre[:, None], 0.0)
        variance = (deviation**2).sum(axis=1) / np.where(count > 1, count - 1, 1)
        enough = count >= max(min_periods, 2)
        mean[start:stop] = np.where(enough, centre, np.nan)
        std[start:stop] = np.where(enough, np.sqrt(variance), np.nan)
    return mean, std


def _lagged_change(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.full(values.size, np.nan)
    if values.size > lag:
        result[lag:] = values[lag:] - values[:-lag]
    return result


def series_feature_arrays(obs: ObsSeries) -> dict[str, np.ndarray]:
    """Per-observation features keyed by suffix. Row ``i`` uses rows ``<= i``."""

    spec = obs.spec
    x = np.asarray(obs.values, dtype=np.float64)
    if spec.kind is SeriesKind.PRICE:
        with np.errstate(invalid="ignore", divide="ignore"):
            z_level = np.where(x > 0.0, np.log(np.where(x > 0.0, x, 1.0)), np.nan)
        prefix = "ret"
    else:
        z_level = x.copy()
        prefix = "chg"
    out: dict[str, np.ndarray] = {}
    changes = {window: _lagged_change(z_level, window) for window in CHANGE_WINDOWS}
    for window, change in changes.items():
        out[f"{prefix}{window}"] = change
    step = changes[1]
    _mean, vol = _trailing_stats(step, VOL_WINDOW, VOL_WINDOW)
    out[f"rvol{VOL_WINDOW}"] = vol
    _step_mean, scale = _trailing_stats(step, SCALE_WINDOW, SCALE_MIN_PERIODS)
    level_mean, level_std = _trailing_stats(z_level, SCALE_WINDOW, SCALE_MIN_PERIODS)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_score = np.where(level_std > _EPS, (z_level - level_mean) / level_std, np.nan)
        out[f"z{SCALE_WINDOW}"] = z_score
        for window, suffix in ((20, "sm20"), (60, "sm60")):
            denominator = scale * math.sqrt(window)
            scaled = np.where(denominator > _EPS, changes[window] / denominator, np.nan)
            out[suffix] = np.clip(scaled, -SCALED_MOMENTUM_CLIP, SCALED_MOMENTUM_CLIP)
    if spec.kind is SeriesKind.LEVEL:
        out["lvl"] = x.copy()
    else:
        for window in MA_WINDOWS:
            ma, _std = _trailing_stats(x, window, window)
            above = np.where(np.isfinite(ma), (x > ma).astype(np.float64), np.nan)
            out[f"abv{window}"] = above
    return out


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """Features aligned to a decision grid, with availability and staleness."""

    feature_set: str
    definitions_digest: str
    input_digest: str
    names: tuple[str, ...]
    decision_ns: np.ndarray  # int64 UTC
    values: np.ndarray  # float64 [time, feature], NaN where unavailable
    available: np.ndarray  # bool [time, feature]
    staleness_hours: np.ndarray  # float64 [time, feature], NaN where unavailable
    series_present: tuple[str, ...]

    def __post_init__(self) -> None:
        shape = (self.decision_ns.size, len(self.names))
        for label, array in (
            ("values", self.values),
            ("available", self.available),
            ("staleness_hours", self.staleness_hours),
        ):
            if array.shape != shape:
                raise ValueError(f"{label} must have shape {shape}")
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")
        if np.any(self.available & ~np.isfinite(self.values)):
            raise ValueError("available entries must be finite")

    def column(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown cross-asset feature: {name}") from error

    def select(self, names: Sequence[str]) -> FeatureFrame:
        indices = [self.column(name) for name in names]
        return FeatureFrame(
            feature_set=self.feature_set,
            definitions_digest=self.definitions_digest,
            input_digest=self.input_digest,
            names=tuple(names),
            decision_ns=self.decision_ns,
            values=self.values[:, indices],
            available=self.available[:, indices],
            staleness_hours=self.staleness_hours[:, indices],
            series_present=self.series_present,
        )


@dataclass(frozen=True, slots=True)
class _Aligned:
    value: np.ndarray
    available: np.ndarray
    staleness: np.ndarray


def _align_feature(
    obs: ObsSeries, feature: np.ndarray, grid_ns: np.ndarray
) -> _Aligned:
    index = align_asof(obs.knowable_ns, grid_ns)
    known = index >= 0
    safe = np.where(known, index, 0)
    if obs.values.size == 0:
        empty = np.full(grid_ns.size, np.nan)
        return _Aligned(empty, np.zeros(grid_ns.size, dtype=np.bool_), empty)
    value = feature[safe]
    staleness = (grid_ns - obs.close_ns[safe]) / _NS_PER_HOUR
    available = (
        known
        & np.isfinite(value)
        & (staleness <= obs.spec.max_staleness_hours)
        & (staleness >= 0.0)
    )
    return _Aligned(
        np.where(available, value, np.nan),
        available,
        np.where(available, staleness, np.nan),
    )


def _combine(
    parts: Sequence[tuple[_Aligned, int]], min_members: int, size: int
) -> _Aligned:
    if not parts:
        empty = np.full(size, np.nan)
        return _Aligned(empty, np.zeros(size, dtype=np.bool_), empty)
    values = np.stack([part.value * sign for part, sign in parts])
    ok = np.stack([part.available for part, _sign in parts])
    stale = np.stack([part.staleness for part, _sign in parts])
    count = ok.sum(axis=0)
    total = np.where(ok, values, 0.0).sum(axis=0)
    available = count >= min_members
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(available, total / np.where(count > 0, count, 1), np.nan)
    worst = np.where(ok, stale, -np.inf).max(axis=0)
    return _Aligned(mean, available, np.where(available, worst, np.nan))


def compute_frame(
    obs: Mapping[str, ObsSeries],
    decision_times: np.ndarray,
    *,
    input_digest: str = "",
) -> FeatureFrame:
    """Compute every feature at the decision times using only released data."""

    times = np.asarray(decision_times)
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("decision_times must be a one-dimensional datetime64 array")
    if np.isnat(times).any():
        raise ValueError("decision_times must not contain NaT")
    grid_ns = times.astype("datetime64[ns]").astype(np.int64)
    size = grid_ns.size

    aligned: dict[tuple[str, str], _Aligned] = {}
    for key, series in obs.items():
        arrays = series_feature_arrays(series)
        for suffix, feature in arrays.items():
            aligned[(key, suffix)] = _align_feature(series, feature, grid_ns)

    columns: dict[str, _Aligned] = {}
    for composite in COMPOSITES:
        parts = [
            (aligned[(key, suffix)], sign)
            for key, suffix, sign in composite.members
            if (key, suffix) in aligned
        ]
        columns[composite.name] = _combine(parts, composite.min_members, size)
    for name, window in zip(BREADTH_NAMES, MA_WINDOWS, strict=True):
        parts = [
            (aligned[(key, f"abv{window}")], 1)
            for key in _breadth_members()
            if (key, f"abv{window}") in aligned
        ]
        columns[name] = _combine(parts, BREADTH_MIN_MEMBERS, size)
    for name, left, right in ROTATIONS:
        a, b = columns[left], columns[right]
        ok = a.available & b.available
        columns[name] = _Aligned(
            np.where(ok, a.value - b.value, np.nan),
            ok,
            np.where(ok, np.fmax(a.staleness, b.staleness), np.nan),
        )
    names: list[str] = list(COMPOSITE_FEATURE_NAMES)
    for spec in CATALOG:
        if spec.key not in obs:
            continue
        for suffix in series_suffixes(spec.kind):
            name = f"{spec.key}__{suffix}"
            columns[name] = aligned[(spec.key, suffix)]
            names.append(name)

    values = np.stack([columns[name].value for name in names], axis=1)
    available = np.stack([columns[name].available for name in names], axis=1)
    staleness = np.stack([columns[name].staleness for name in names], axis=1)
    return FeatureFrame(
        feature_set=FEATURE_SET_ID,
        definitions_digest=definitions_digest(),
        input_digest=input_digest,
        names=tuple(names),
        decision_ns=grid_ns,
        values=values,
        available=available,
        staleness_hours=staleness,
        series_present=tuple(sorted(obs)),
    )


__all__ = [
    "BREADTH_NAMES",
    "COMPOSITES",
    "COMPOSITE_FEATURE_NAMES",
    "FEATURE_SET_ID",
    "FeatureFrame",
    "SeriesSpec",
    "compute_frame",
    "definitions",
    "definitions_digest",
    "series_feature_arrays",
    "series_suffixes",
]
