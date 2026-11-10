"""Leakage detection harness for feature builders (Issue #1146, card SD1).

A feature at decision row ``t`` is *causal* only if it is a function of inputs at rows
``<= t``. :func:`future_perturbation_probe` tests exactly that, structurally: it
rebuilds the feature after randomly perturbing every input row **after** a cutoff and
requires every feature row at or before the cutoff to be unchanged. A feature that
reads a future price, a centred window, or a full-sample statistic fails, whatever its
IC looks like.

:func:`affine_label_copy` is a second, narrow screen for the crudest leak, a feature
that is an affine or monotone copy of the matured label. It is not a substitute for the
probe: a subtler leak (a centred average) is not a copy and only the probe sees it.

Neither function reads data; both take the builder and arrays from the caller.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from trade_rl.evaluation.signal_measurement.ic import (
    SignalMeasurementError,
    linear_ic,
    rank_ic,
)

_MAX_RELATIVE_SCALE = (
    0.3  # keeps the multiplicative factor 1 + scale * clip(z, +-3) > 0
)


@dataclass(frozen=True, slots=True)
class LeakageProbeResult:
    """Outcome per cutoff: how many feature rows at or before it moved."""

    cutoffs: tuple[int, ...]
    changed_rows: tuple[tuple[int, int], ...]

    @property
    def leaking_cutoffs(self) -> tuple[int, ...]:
        return tuple(cutoff for cutoff, changed in self.changed_rows if changed > 0)

    @property
    def leaks(self) -> bool:
        return bool(self.leaking_cutoffs)


def _cutoff(value: object, rows: int) -> int:
    if type(value) is not int or not 0 <= value <= rows - 2:
        raise SignalMeasurementError(
            "each cutoff must be an integer row index with at least one later row"
        )
    return value


def future_perturbation_probe(
    builder: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    *,
    cutoffs: Sequence[int],
    seed: int,
    relative_scale: float,
) -> LeakageProbeResult:
    """Flag a feature builder whose rows at or before a cutoff depend on later inputs.

    ``inputs`` is a ``(rows, columns)`` array of non-zero values (prices, volumes);
    ``builder`` maps it to a feature array with the same number of rows. For each
    cutoff ``c`` the rows after ``c`` are multiplied by ``1 + relative_scale * z`` with
    ``z`` a seeded standard normal clipped to ``[-3, 3]``, the feature is rebuilt, and
    the rows ``0..c`` of both feature arrays must be identical (``NaN`` equals
    ``NaN``). The input array is never modified.
    """

    values = np.asarray(inputs)
    if values.ndim != 2 or values.dtype.kind not in "fiu":
        raise SignalMeasurementError("inputs must be a 2-D numeric array")
    if (
        isinstance(relative_scale, bool)
        or not isinstance(relative_scale, (int, float))
        or not math.isfinite(relative_scale)
        or not 0.0 < relative_scale < _MAX_RELATIVE_SCALE
    ):
        raise SignalMeasurementError(
            f"relative_scale must lie strictly between 0 and {_MAX_RELATIVE_SCALE}"
        )
    if type(seed) is not int or seed < 0:
        raise SignalMeasurementError("seed must be a non-negative integer")
    rows = values.shape[0]
    chosen = tuple(_cutoff(cutoff, rows) for cutoff in cutoffs)
    if not chosen:
        raise SignalMeasurementError("at least one cutoff is required")

    base_inputs = values.astype(np.float64, copy=True)
    base = np.asarray(builder(base_inputs.copy()))
    if base.shape[0] != rows:
        raise SignalMeasurementError(
            "the builder must return one feature row per input row"
        )

    rng = np.random.default_rng(seed)
    changed: list[tuple[int, int]] = []
    for cutoff in chosen:
        noise = np.clip(rng.standard_normal(base_inputs[cutoff + 1 :].shape), -3.0, 3.0)
        perturbed = base_inputs.copy()
        perturbed[cutoff + 1 :] = perturbed[cutoff + 1 :] * (
            1.0 + relative_scale * noise
        )
        if np.array_equal(perturbed, base_inputs):
            raise SignalMeasurementError(
                "the perturbation changed nothing; the inputs after the cutoff must be non-zero"
            )
        rebuilt = np.asarray(builder(perturbed))
        if rebuilt.shape != base.shape:
            raise SignalMeasurementError("the builder must keep its output shape")
        past_base = base[: cutoff + 1].reshape(cutoff + 1, -1)
        past_new = rebuilt[: cutoff + 1].reshape(cutoff + 1, -1)
        same = (past_base == past_new) | (np.isnan(past_base) & np.isnan(past_new))
        changed.append((cutoff, int((~same.all(axis=1)).sum())))
    return LeakageProbeResult(cutoffs=chosen, changed_rows=tuple(changed))


def affine_label_copy(
    feature: object, label: object, *, tolerance: float = 1e-9
) -> bool:
    """True if the feature is an affine or monotone copy of the matured label.

    Both arrays are flattened and compared on the rows where both are finite. The
    feature is flagged when its linear or rank IC with the label has absolute value at
    least ``1 - tolerance``. Fewer than three eligible rows, or a constant series, can
    not be judged and return ``False``.
    """

    left = np.asarray(feature).ravel()
    right = np.asarray(label).ravel()
    if left.shape != right.shape:
        raise SignalMeasurementError("feature and label must have the same size")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 0.0 < tolerance < 1.0
    ):
        raise SignalMeasurementError("tolerance must lie strictly between 0 and 1")
    keep = np.isfinite(left.astype(np.float64)) & np.isfinite(right.astype(np.float64))
    if int(keep.sum()) < 3:
        return False
    for statistic in (
        linear_ic(left[keep], right[keep]),
        rank_ic(left[keep], right[keep]),
    ):
        if statistic is not None and abs(statistic) >= 1.0 - tolerance:
            return True
    return False


__all__ = ["LeakageProbeResult", "affine_label_copy", "future_perturbation_probe"]
