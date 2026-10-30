"""Study-level train-window standardize-and-clip input transform (#1025 TD8).

The TD5-TD7 dev study fed raw dataset features to every arm. A non-stationary
feature (``funding_bps``) left its training range in a later block and the
linear models extrapolated. This module applies one uniform, causal input rule
before any arm sees a feature: standardize each ``(symbol, feature)`` column with
the *train-window* mean and standard deviation, then clip to ``[-clip, clip]``.

Statistics are computed from the declared train window only, so rows after the
window can never influence them. The rule was chosen after a diagnosis had
observed the funding shift; that is a disclosed researcher degree of freedom,
not a result-blind choice. :func:`covariate_shift_report` is diagnostic only and
never removes or flags a feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset

_MIN_SCALE = 1e-8


@dataclass(frozen=True, slots=True)
class FeatureTransform:
    """Train-window statistics and clip of the uniform input rule."""

    train_start_index: int
    train_stop_index: int
    clip: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    global_mean: np.ndarray
    global_scale: np.ndarray

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": "decision_value_feature_transform_v1",
                "train_start_index": self.train_start_index,
                "train_stop_index": self.train_stop_index,
                "clip": self.clip,
                "feature_mean": self.feature_mean.tolist(),
                "feature_scale": self.feature_scale.tolist(),
                "global_mean": self.global_mean.tolist(),
                "global_scale": self.global_scale.tolist(),
            }
        )


def _masked_mean_scale(
    values: np.ndarray, available: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean / population std over available cells only."""
    mask = available.astype(np.float64)
    count = mask.sum(axis=0)
    safe = np.maximum(count, 1.0)
    mean = (values * mask).sum(axis=0) / safe
    var = (((values - mean) * mask) ** 2).sum(axis=0) / safe
    std = np.sqrt(var)
    mean = np.where(count > 0, mean, 0.0)
    scale = np.where(std > _MIN_SCALE, std, 1.0)
    return mean, scale


def _validated_clip(clip: float) -> float:
    if isinstance(clip, bool) or not isinstance(clip, (int, float)):
        raise TypeError("clip must be a number")
    value = float(clip)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("clip must be positive and finite")
    return value


def fit_train_feature_transform(
    dataset: MarketDataset,
    *,
    train_start_index: int,
    train_stop_index: int,
    clip: float = 3.0,
) -> FeatureTransform:
    """Fit mean/scale on ``[train_start_index, train_stop_index]`` (inclusive)."""
    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    value = _validated_clip(clip)
    if (
        type(train_start_index) is not int
        or type(train_stop_index) is not int
        or not 0 <= train_start_index < train_stop_index < dataset.n_bars
    ):
        raise ValueError(
            "train window must satisfy 0 <= start < stop < n_bars as native integers"
        )
    window = slice(train_start_index, train_stop_index + 1)
    f_mean, f_scale = _masked_mean_scale(
        dataset.features[window].astype(np.float64),
        dataset.feature_available[window],
    )
    g_avail = dataset.resolved_array("global_feature_available")[window]
    g_mean, g_scale = _masked_mean_scale(
        dataset.global_features[window].astype(np.float64), g_avail
    )
    return FeatureTransform(
        train_start_index=train_start_index,
        train_stop_index=train_stop_index,
        clip=value,
        feature_mean=f_mean,
        feature_scale=f_scale,
        global_mean=g_mean,
        global_scale=g_scale,
    )


def apply_feature_transform(
    dataset: MarketDataset, transform: FeatureTransform
) -> MarketDataset:
    """Return ``dataset`` with every feature standardized and clipped.

    Unavailable cells stay unavailable and are set to the neutral value ``0``.
    The extra transform is bound into ``normalization_digest`` so a model fitted
    on the transformed rows cannot be run against raw or differently
    transformed rows.
    """
    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    if not isinstance(transform, FeatureTransform):
        raise TypeError("transform must be a FeatureTransform")
    if transform.feature_mean.shape != dataset.features.shape[1:]:
        raise ValueError("transform does not match the dataset feature roster")
    if transform.global_mean.shape != dataset.global_features.shape[1:]:
        raise ValueError("transform does not match the dataset global feature roster")
    z = (
        dataset.features.astype(np.float64) - transform.feature_mean
    ) / transform.feature_scale
    z = np.where(
        dataset.feature_available, np.clip(z, -transform.clip, transform.clip), 0.0
    )
    g = (
        dataset.global_features.astype(np.float64) - transform.global_mean
    ) / transform.global_scale
    g_avail = dataset.resolved_array("global_feature_available")
    g = np.where(g_avail, np.clip(g, -transform.clip, transform.clip), 0.0)
    return replace(
        dataset,
        features=z.astype(np.float32),
        global_features=g.astype(np.float32),
        normalization_digest=content_digest(
            {
                "schema_version": "decision_value_normalization_with_transform_v1",
                "base_normalization_digest": dataset.normalization_digest,
                "transform_digest": transform.digest,
            }
        ),
        identity_payload_json=None,
    )


def covariate_shift_report(
    dataset: MarketDataset,
    transform: FeatureTransform,
    *,
    start_index: int,
    stop_index: int,
) -> tuple[dict[str, object], ...]:
    """Diagnostic eval-versus-train z-scores per feature (never drops anything).

    ``dataset`` must carry the raw (untransformed) features. Rows
    ``[start_index, stop_index]`` are standardized with the *train-window*
    statistics, without clipping. Symbol features are reported once per feature
    name (mean over symbols and rows, max over symbols and rows).
    """
    if not isinstance(dataset, MarketDataset):
        raise TypeError("dataset must be a MarketDataset")
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index <= stop_index < dataset.n_bars
    ):
        raise ValueError("report window is invalid")
    window = slice(start_index, stop_index + 1)
    clip = transform.clip
    rows: list[dict[str, object]] = []
    z = (
        dataset.features[window].astype(np.float64) - transform.feature_mean
    ) / transform.feature_scale
    avail = dataset.feature_available[window]
    for j, name in enumerate(dataset.feature_names):
        cells = z[:, :, j][avail[:, :, j]]
        rows.append(_shift_row(name, "symbol", cells, clip))
    gz = (
        dataset.global_features[window].astype(np.float64) - transform.global_mean
    ) / transform.global_scale
    g_avail = dataset.resolved_array("global_feature_available")[window]
    for j, name in enumerate(dataset.global_feature_names):
        cells = gz[:, j][g_avail[:, j]]
        rows.append(_shift_row(name, "global", cells, clip))
    return tuple(rows)


def _shift_row(
    name: str, scope: str, cells: np.ndarray, clip: float
) -> dict[str, object]:
    if cells.size == 0:
        return {
            "name": name,
            "scope": scope,
            "mean_z": 0.0,
            "max_abs_z": 0.0,
            "fraction_clipped": 0.0,
        }
    return {
        "name": name,
        "scope": scope,
        "mean_z": float(cells.mean()),
        "max_abs_z": float(np.abs(cells).max()),
        "fraction_clipped": float((np.abs(cells) > clip).mean()),
    }


__all__ = [
    "FeatureTransform",
    "apply_feature_transform",
    "covariate_shift_report",
    "fit_train_feature_transform",
]
