"""Normalisation statistics fitted only inside a caller-provided window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.cross_asset.features import FeatureFrame

_EPS = 1e-12


def _to_ns(value: np.datetime64 | datetime) -> int:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fit window datetimes must be timezone-aware")
        value = np.datetime64(value.astimezone(UTC).replace(tzinfo=None), "ns")
    if np.isnat(value):
        raise ValueError("fit window must not contain NaT")
    return int(np.datetime64(value, "ns").astype(np.int64))


@dataclass(frozen=True, slots=True)
class FeatureNormalizer:
    """Per-feature mean/scale from rows with ``fit_start <= t < fit_end``."""

    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray  # NaN where the fit window had too little data
    n_fit: np.ndarray
    fit_start_ns: int
    fit_end_ns: int
    clip: float
    definitions_digest: str

    @classmethod
    def fit(
        cls,
        frame: FeatureFrame,
        *,
        fit_start: np.datetime64 | datetime,
        fit_end: np.datetime64 | datetime,
        min_count: int = 30,
        clip: float = 5.0,
    ) -> FeatureNormalizer:
        start, end = _to_ns(fit_start), _to_ns(fit_end)
        if start >= end:
            raise ValueError("fit window must satisfy fit_start < fit_end")
        if min_count < 2:
            raise ValueError("min_count must be at least 2")
        if clip <= 0.0:
            raise ValueError("clip must be positive")
        rows = (frame.decision_ns >= start) & (frame.decision_ns < end)
        usable = frame.available[rows]
        values = np.where(usable, frame.values[rows], 0.0)
        count = usable.sum(axis=0)
        safe = np.where(count > 0, count, 1)
        mean = values.sum(axis=0) / safe
        deviation = np.where(usable, frame.values[rows] - mean, 0.0)
        variance = (deviation**2).sum(axis=0) / np.where(count > 1, count - 1, 1)
        scale = np.sqrt(variance)
        fitted = (count >= min_count) & (scale > _EPS)
        return cls(
            names=frame.names,
            mean=np.where(fitted, mean, np.nan),
            scale=np.where(fitted, scale, np.nan),
            n_fit=count.astype(np.int64),
            fit_start_ns=start,
            fit_end_ns=end,
            clip=float(clip),
            definitions_digest=frame.definitions_digest,
        )

    def transform(self, frame: FeatureFrame) -> FeatureFrame:
        """Standardise and clip. Features without fitted statistics become
        unavailable rather than silently unscaled."""

        if frame.names != self.names:
            raise ValueError("frame features do not match the fitted normalizer")
        if frame.definitions_digest != self.definitions_digest:
            raise ValueError("frame was built from different feature definitions")
        fitted = np.isfinite(self.scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            scaled = np.clip(
                (frame.values - self.mean) / self.scale, -self.clip, self.clip
            )
        available = frame.available & fitted[None, :]
        return FeatureFrame(
            feature_set=frame.feature_set,
            definitions_digest=frame.definitions_digest,
            input_digest=frame.input_digest,
            names=frame.names,
            decision_ns=frame.decision_ns,
            values=np.where(available, scaled, np.nan),
            available=available,
            staleness_hours=np.where(available, frame.staleness_hours, np.nan),
            series_present=frame.series_present,
        )

    def digest(self) -> str:
        return content_digest(
            {
                "names": list(self.names),
                "mean": [None if np.isnan(v) else float(v) for v in self.mean],
                "scale": [None if np.isnan(v) else float(v) for v in self.scale],
                "n_fit": [int(v) for v in self.n_fit],
                "fit_start_ns": self.fit_start_ns,
                "fit_end_ns": self.fit_end_ns,
                "clip": self.clip,
                "definitions_digest": self.definitions_digest,
            }
        )


__all__ = ["FeatureNormalizer"]
