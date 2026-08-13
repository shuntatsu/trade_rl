from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)


def test_signal_diagnostics_report_exact_alignment_and_fixed_bins() -> None:
    diagnostics = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
        np.asarray([-1.0, -2.0, 0.0, 2.0, 1.0]),
    )

    assert diagnostics.sample_count == 5
    assert diagnostics.prediction_mean == pytest.approx(0.0)
    assert diagnostics.realized_mean == pytest.approx(0.0)
    assert diagnostics.prediction_quantiles == pytest.approx(
        (-2.0, -1.6, -1.0, 0.0, 1.0, 1.6, 2.0)
    )
    assert diagnostics.pearson_correlation == pytest.approx(0.8)
    assert diagnostics.rank_correlation == pytest.approx(0.8)
    assert diagnostics.direction_accuracy == pytest.approx(1.0)
    assert diagnostics.prediction_negative_rate == pytest.approx(0.4)
    assert diagnostics.prediction_flat_rate == pytest.approx(0.2)
    assert diagnostics.prediction_positive_rate == pytest.approx(0.4)
    assert sum(item.count for item in diagnostics.bins) == 5
    assert diagnostics.undefined_correlation_reason is None
    assert diagnostics.digest


def test_signal_diagnostics_use_average_ranks_for_ties() -> None:
    diagnostics = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([1.0, 1.0, 2.0, 3.0]),
        np.asarray([1.0, 2.0, 2.0, 4.0]),
    )

    assert diagnostics.rank_correlation == pytest.approx(5.0 / 6.0)


def test_signal_diagnostics_preserve_undefined_constant_prediction() -> None:
    diagnostics = evaluate_causal_alpha_signal_diagnostics(
        np.ones(5, dtype=np.float64),
        np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
    )

    assert diagnostics.pearson_correlation is None
    assert diagnostics.rank_correlation is None
    assert diagnostics.undefined_correlation_reason == "constant_prediction"
    assert sum(item.count for item in diagnostics.bins) == 5


@pytest.mark.parametrize(
    ("predicted", "realized"),
    (
        ([1.0], [1.0]),
        ([1.0, 2.0], [1.0]),
        ([1.0, float("nan")], [1.0, 2.0]),
    ),
)
def test_signal_diagnostics_reject_invalid_samples(
    predicted: list[float], realized: list[float]
) -> None:
    with pytest.raises(ValueError):
        evaluate_causal_alpha_signal_diagnostics(predicted, realized)
