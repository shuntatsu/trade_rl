"""Deterministic diagnostics for independent Cost Critic learning."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

_EPSILON = float(np.finfo(np.float64).eps)


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One fixed probability interval in an event calibration table."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_probability: float | None
    event_rate: float | None


@dataclass(frozen=True, slots=True)
class PrecisionRecallInputs:
    """Threshold-complete precision/recall inputs without threshold selection."""

    thresholds: tuple[float, ...]
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    positive_sample_count: int


@dataclass(frozen=True, slots=True)
class CostHeadDiagnostics:
    """Per-cost target, value-learning, gradient, and event diagnostics."""

    name: str
    target_mean: float
    target_std: float
    nonzero_rate: float
    positive_sample_count: int
    value_loss: float
    explained_variance: float
    adapter_gradient_norm: float
    head_gradient_norm: float
    brier_score: float | None
    calibration_bins: tuple[CalibrationBin, ...]
    precision_recall: PrecisionRecallInputs | None
    zero_only_brier_score: float | None
    beats_zero_only_baseline: bool | None
    has_positive_support: bool | None
    eligible_for_promotion: bool | None


@dataclass(frozen=True, slots=True)
class FamilyGradientDiagnostics:
    """Aggregate continuous-to-event gradient balance."""

    continuous_adapter_gradient_norm: float
    continuous_head_gradient_norm: float
    event_adapter_gradient_norm: float
    event_head_gradient_norm: float
    continuous_gradient_norm: float
    event_gradient_norm: float
    dense_to_rare_gradient_ratio: float | None


def _paired_finite_vectors(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.size == 0 or right.size == 0:
        raise ValueError(f"{first_name} and {second_name} must be non-empty")
    if left.shape != right.shape:
        raise ValueError(f"{first_name} and {second_name} must have the same shape")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{first_name} and {second_name} must contain finite values")
    return left.reshape(-1), right.reshape(-1)


def _validated_gradient_norm(value: float, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return result


def explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Return finite explained variance with explicit constant-target semantics."""

    target, prediction = _paired_finite_vectors(
        targets,
        predictions,
        first_name="targets",
        second_name="predictions",
    )
    target_variance = float(np.var(target))
    if target_variance <= _EPSILON:
        return 1.0 if np.array_equal(target, prediction) else 0.0
    residual_variance = float(np.var(target - prediction))
    return float(1.0 - residual_variance / target_variance)


def gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Calculate an aggregate L2 norm from currently populated gradients."""

    squared_norm = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError("parameter gradients must be finite")
        squared_norm += float(torch.sum(detached.double().square()).cpu())
    return math.sqrt(squared_norm)


def build_family_gradient_diagnostics(
    *,
    continuous_adapter_parameters: Iterable[torch.nn.Parameter],
    continuous_head_parameters: Iterable[torch.nn.Parameter],
    event_adapter_parameters: Iterable[torch.nn.Parameter],
    event_head_parameters: Iterable[torch.nn.Parameter],
) -> FamilyGradientDiagnostics:
    """Summarize family gradient norms without hiding a zero rare gradient."""

    continuous_adapter = gradient_l2_norm(continuous_adapter_parameters)
    continuous_heads = gradient_l2_norm(continuous_head_parameters)
    event_adapter = gradient_l2_norm(event_adapter_parameters)
    event_heads = gradient_l2_norm(event_head_parameters)
    continuous_total = math.hypot(continuous_adapter, continuous_heads)
    event_total = math.hypot(event_adapter, event_heads)
    ratio = None if event_total == 0.0 else continuous_total / event_total
    return FamilyGradientDiagnostics(
        continuous_adapter_gradient_norm=continuous_adapter,
        continuous_head_gradient_norm=continuous_heads,
        event_adapter_gradient_norm=event_adapter,
        event_head_gradient_norm=event_heads,
        continuous_gradient_norm=continuous_total,
        event_gradient_norm=event_total,
        dense_to_rare_gradient_ratio=ratio,
    )


def _calibration_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bin_count: int,
) -> tuple[CalibrationBin, ...]:
    edges = np.linspace(0.0, 1.0, bin_count + 1, dtype=np.float64)
    indices = np.minimum(
        np.floor(probabilities * bin_count).astype(np.int64),
        bin_count - 1,
    )
    bins: list[CalibrationBin] = []
    for index in range(bin_count):
        selected = indices == index
        count = int(np.count_nonzero(selected))
        bins.append(
            CalibrationBin(
                lower_bound=float(edges[index]),
                upper_bound=float(edges[index + 1]),
                count=count,
                mean_probability=(
                    None if count == 0 else float(np.mean(probabilities[selected]))
                ),
                event_rate=None if count == 0 else float(np.mean(labels[selected])),
            )
        )
    return tuple(bins)


def _precision_recall_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> PrecisionRecallInputs:
    thresholds = np.unique(probabilities)[::-1]
    positive_count = int(np.count_nonzero(labels == 1.0))
    precision: list[float] = []
    recall: list[float] = []
    for threshold in thresholds:
        predicted_positive = probabilities >= threshold
        predicted_count = int(np.count_nonzero(predicted_positive))
        true_positive = int(
            np.count_nonzero(predicted_positive & np.equal(labels, 1.0))
        )
        precision.append(
            0.0 if predicted_count == 0 else true_positive / predicted_count
        )
        recall.append(0.0 if positive_count == 0 else true_positive / positive_count)
    return PrecisionRecallInputs(
        thresholds=tuple(float(value) for value in thresholds),
        precision=tuple(precision),
        recall=tuple(recall),
        positive_sample_count=positive_count,
    )


def build_cost_head_diagnostics(
    *,
    name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    adapter_gradient_norm: float,
    head_gradient_norm: float,
    event_probabilities: np.ndarray | None = None,
    event_labels: np.ndarray | None = None,
    calibration_bin_count: int = 10,
) -> CostHeadDiagnostics:
    """Build a deterministic report for one enabled cost head."""

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    prediction, target = _paired_finite_vectors(
        predictions,
        targets,
        first_name="predictions",
        second_name="targets",
    )
    adapter_norm = _validated_gradient_norm(
        adapter_gradient_norm,
        field_name="adapter_gradient_norm",
    )
    head_norm = _validated_gradient_norm(
        head_gradient_norm,
        field_name="head_gradient_norm",
    )
    if (
        isinstance(calibration_bin_count, bool)
        or not isinstance(calibration_bin_count, int)
        or calibration_bin_count <= 0
    ):
        raise ValueError("calibration_bin_count must be a positive integer")

    positive_sample_count = int(np.count_nonzero(target > 0.0))
    brier_score: float | None = None
    calibration: tuple[CalibrationBin, ...] = ()
    precision_recall: PrecisionRecallInputs | None = None
    zero_only_brier_score: float | None = None
    beats_zero_only_baseline: bool | None = None
    has_positive_support: bool | None = None
    eligible_for_promotion: bool | None = None
    if (event_probabilities is None) != (event_labels is None):
        raise ValueError("event probabilities and labels must be provided together")
    if event_probabilities is not None and event_labels is not None:
        probabilities, labels = _paired_finite_vectors(
            event_probabilities,
            event_labels,
            first_name="event probabilities",
            second_name="event labels",
        )
        if probabilities.shape != target.shape:
            raise ValueError("event probabilities must match target shape")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError("event probabilities must be within [0, 1]")
        if np.any((labels != 0.0) & (labels != 1.0)):
            raise ValueError("event labels must be binary")
        positive_sample_count = int(np.count_nonzero(labels == 1.0))
        brier_score = float(np.mean(np.square(probabilities - labels)))
        zero_only_brier_score = float(np.mean(np.square(labels)))
        beats_zero_only_baseline = brier_score < zero_only_brier_score
        has_positive_support = positive_sample_count > 0
        eligible_for_promotion = beats_zero_only_baseline and has_positive_support
        calibration = _calibration_bins(
            probabilities,
            labels,
            bin_count=calibration_bin_count,
        )
        precision_recall = _precision_recall_inputs(probabilities, labels)

    return CostHeadDiagnostics(
        name=name,
        target_mean=float(np.mean(target)),
        target_std=float(np.std(target)),
        nonzero_rate=float(np.count_nonzero(target) / target.size),
        positive_sample_count=positive_sample_count,
        value_loss=float(np.mean(np.square(prediction - target))),
        explained_variance=explained_variance(target, prediction),
        adapter_gradient_norm=adapter_norm,
        head_gradient_norm=head_norm,
        brier_score=brier_score,
        calibration_bins=calibration,
        precision_recall=precision_recall,
        zero_only_brier_score=zero_only_brier_score,
        beats_zero_only_baseline=beats_zero_only_baseline,
        has_positive_support=has_positive_support,
        eligible_for_promotion=eligible_for_promotion,
    )


__all__ = [
    "CalibrationBin",
    "CostHeadDiagnostics",
    "FamilyGradientDiagnostics",
    "PrecisionRecallInputs",
    "build_cost_head_diagnostics",
    "build_family_gradient_diagnostics",
    "explained_variance",
    "gradient_l2_norm",
]
