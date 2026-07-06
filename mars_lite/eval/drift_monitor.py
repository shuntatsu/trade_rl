"""Feature and prediction distribution drift monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DriftMonitorConfig:
    psi_threshold: float = 0.2
    ks_threshold: float = 0.2
    prediction_psi_threshold: float = 0.2
    bins: int = 10


@dataclass(frozen=True)
class DriftAlert:
    metric: str
    feature_index: int | None
    value: float
    threshold: float


@dataclass(frozen=True)
class DriftReport:
    alerts: list[DriftAlert]
    feature_psi: list[float]
    feature_ks: list[float]
    prediction_psi: float | None
    should_flatten: bool


class DriftMonitor:
    """Monitor live windows against a fixed training reference distribution."""

    def __init__(
        self,
        reference_features: np.ndarray,
        reference_predictions: np.ndarray | None = None,
        config: DriftMonitorConfig | None = None,
    ) -> None:
        reference = np.asarray(reference_features, dtype=float)
        if reference.ndim != 2:
            raise ValueError("reference_features must be a 2D array")
        if reference.size == 0:
            raise ValueError("reference_features cannot be empty")
        self.reference_features = reference
        self.reference_predictions = _optional_1d(reference_predictions)
        self.config = config or DriftMonitorConfig()
        if self.config.psi_threshold <= 0:
            raise ValueError("psi_threshold must be positive")
        if self.config.ks_threshold <= 0 or self.config.ks_threshold > 1.0:
            raise ValueError("ks_threshold must be between 0 and 1")
        if self.config.prediction_psi_threshold <= 0:
            raise ValueError("prediction_psi_threshold must be positive")
        if self.config.bins <= 0:
            raise ValueError("bins must be positive")

    def evaluate(
        self,
        current_features: np.ndarray,
        predictions: np.ndarray | None = None,
    ) -> DriftReport:
        current = np.asarray(current_features, dtype=float)
        if current.ndim != 2:
            raise ValueError("current_features must be a 2D array")
        if current.size == 0:
            raise ValueError("current_features cannot be empty")
        if current.shape[1] != self.reference_features.shape[1]:
            raise ValueError("feature dimensions do not match")

        alerts: list[DriftAlert] = []
        psi_values: list[float] = []
        ks_values: list[float] = []
        for index in range(current.shape[1]):
            ref = self.reference_features[:, index]
            cur = current[:, index]
            psi_value = population_stability_index(ref, cur, bins=self.config.bins)
            ks_value = ks_statistic(ref, cur)
            psi_values.append(psi_value)
            ks_values.append(ks_value)
            if psi_value > self.config.psi_threshold:
                alerts.append(
                    DriftAlert("psi", index, psi_value, self.config.psi_threshold)
                )
            if ks_value > self.config.ks_threshold:
                alerts.append(
                    DriftAlert("ks", index, ks_value, self.config.ks_threshold)
                )

        prediction_psi = None
        current_predictions = _optional_1d(predictions)
        if self.reference_predictions is not None and current_predictions is not None:
            prediction_psi = population_stability_index(
                self.reference_predictions,
                current_predictions,
                bins=self.config.bins,
            )
            if prediction_psi > self.config.prediction_psi_threshold:
                alerts.append(
                    DriftAlert(
                        "prediction_psi",
                        None,
                        prediction_psi,
                        self.config.prediction_psi_threshold,
                    )
                )

        return DriftReport(
            alerts=alerts,
            feature_psi=psi_values,
            feature_ks=ks_values,
            prediction_psi=prediction_psi,
            should_flatten=bool(alerts),
        )

    def first_alert_step(
        self, current_features: np.ndarray, window_size: int = 100
    ) -> int | None:
        current = np.asarray(current_features, dtype=float)
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if len(current) < window_size:
            raise ValueError("current_features length must be at least window_size")
        for end in range(window_size, len(current) + 1):
            if self.evaluate(current[end - window_size : end]).should_flatten:
                return end
        return None


def population_stability_index(
    expected: Iterable[float], actual: Iterable[float], bins: int = 10
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    expected_values = np.asarray(expected, dtype=float)
    actual_values = np.asarray(actual, dtype=float)
    if expected_values.size == 0 or actual_values.size == 0:
        return 0.0

    lower = min(float(np.min(expected_values)), float(np.min(actual_values)))
    upper = max(float(np.max(expected_values)), float(np.max(actual_values)))
    if lower == upper:
        upper = lower + 1.0
    edges = np.linspace(lower, upper, bins + 1)
    expected_counts, _ = np.histogram(expected_values, bins=edges)
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_pct = np.clip(expected_counts / expected_values.size, 1e-6, None)
    actual_pct = np.clip(actual_counts / actual_values.size, 1e-6, None)
    return float(
        np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    )


def ks_statistic(reference: Iterable[float], current: Iterable[float]) -> float:
    ref = np.sort(np.asarray(reference, dtype=float))
    cur = np.sort(np.asarray(current, dtype=float))
    if ref.size == 0 or cur.size == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([ref, cur])))
    ref_cdf = np.searchsorted(ref, values, side="right") / ref.size
    cur_cdf = np.searchsorted(cur, values, side="right") / cur.size
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


def _optional_1d(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("prediction arrays must be 1D")
    return array
