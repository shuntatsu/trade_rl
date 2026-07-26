from __future__ import annotations

import numpy as np
import pytest
import torch


def test_continuous_cost_head_report_uses_explicit_target_statistics() -> None:
    from trade_rl.rl.cost_diagnostics import build_cost_head_diagnostics

    predictions = np.array([0.1, 0.2, 0.4, 0.5])
    targets = np.array([0.0, 0.2, 0.3, 0.7])
    report = build_cost_head_diagnostics(
        name="daily_turnover",
        predictions=predictions,
        targets=targets,
        adapter_gradient_norm=2.0,
        head_gradient_norm=0.5,
    )

    assert report.name == "daily_turnover"
    assert report.target_mean == pytest.approx(0.3)
    assert report.target_std == pytest.approx(np.std(targets))
    assert report.nonzero_rate == pytest.approx(0.75)
    assert report.positive_sample_count == 3
    assert report.value_loss == pytest.approx(0.015)
    expected_variance = 1.0 - np.var(targets - predictions) / np.var(targets)
    assert report.explained_variance == pytest.approx(expected_variance)
    assert report.adapter_gradient_norm == 2.0
    assert report.head_gradient_norm == 0.5
    assert report.brier_score is None
    assert report.calibration_bins == ()
    assert report.precision_recall is None


def test_event_cost_head_report_has_deterministic_calibration_and_pr_inputs() -> None:
    from trade_rl.rl.cost_diagnostics import build_cost_head_diagnostics

    probabilities = np.array([0.05, 0.20, 0.55, 0.80, 1.0])
    labels = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    report = build_cost_head_diagnostics(
        name="forced_liquidation_event",
        predictions=np.array([0.1, 0.3, 0.2, 0.7, 0.8]),
        targets=np.array([0.0, 0.8, 0.2, 1.0, 1.0]),
        adapter_gradient_norm=0.75,
        head_gradient_norm=0.25,
        event_probabilities=probabilities,
        event_labels=labels,
        calibration_bin_count=4,
    )

    assert report.brier_score == pytest.approx(np.mean((probabilities - labels) ** 2))
    assert tuple(bin_.count for bin_ in report.calibration_bins) == (2, 0, 1, 2)
    assert report.calibration_bins[0].mean_probability == pytest.approx(0.125)
    assert report.calibration_bins[0].event_rate == pytest.approx(0.5)
    assert report.calibration_bins[1].mean_probability is None
    assert report.calibration_bins[1].event_rate is None
    assert report.calibration_bins[-1].mean_probability == pytest.approx(0.9)
    assert report.calibration_bins[-1].event_rate == pytest.approx(1.0)

    curve = report.precision_recall
    assert curve is not None
    assert curve.thresholds == (1.0, 0.8, 0.55, 0.2, 0.05)
    assert curve.precision == pytest.approx((1.0, 1.0, 2 / 3, 0.75, 0.6))
    assert curve.recall == pytest.approx((1 / 3, 2 / 3, 2 / 3, 1.0, 1.0))
    assert curve.positive_sample_count == 3


def test_constant_target_explained_variance_is_explicit() -> None:
    from trade_rl.rl.cost_diagnostics import explained_variance

    assert explained_variance(np.ones(3), np.ones(3)) == 1.0
    assert explained_variance(np.zeros(3), np.ones(3)) == 0.0


def test_gradient_diagnostics_report_dense_to_rare_ratio() -> None:
    from trade_rl.rl.cost_diagnostics import (
        build_family_gradient_diagnostics,
        gradient_l2_norm,
    )

    dense = torch.nn.Linear(2, 1, bias=False)
    rare = torch.nn.Linear(2, 1, bias=False)
    dense.weight.grad = torch.tensor([[3.0, 4.0]])
    rare.weight.grad = torch.tensor([[0.0, 2.0]])

    assert gradient_l2_norm(dense.parameters()) == pytest.approx(5.0)
    diagnostics = build_family_gradient_diagnostics(
        continuous_adapter_parameters=dense.parameters(),
        continuous_head_parameters=(),
        event_adapter_parameters=rare.parameters(),
        event_head_parameters=(),
    )

    assert diagnostics.continuous_gradient_norm == pytest.approx(5.0)
    assert diagnostics.event_gradient_norm == pytest.approx(2.0)
    assert diagnostics.dense_to_rare_gradient_ratio == pytest.approx(2.5)


def test_gradient_ratio_is_undefined_when_rare_gradient_is_zero() -> None:
    from trade_rl.rl.cost_diagnostics import build_family_gradient_diagnostics

    dense = torch.nn.Linear(1, 1, bias=False)
    rare = torch.nn.Linear(1, 1, bias=False)
    dense.weight.grad = torch.ones_like(dense.weight)
    rare.weight.grad = torch.zeros_like(rare.weight)

    diagnostics = build_family_gradient_diagnostics(
        continuous_adapter_parameters=dense.parameters(),
        continuous_head_parameters=(),
        event_adapter_parameters=rare.parameters(),
        event_head_parameters=(),
    )

    assert diagnostics.event_gradient_norm == 0.0
    assert diagnostics.dense_to_rare_gradient_ratio is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"predictions": np.array([]), "targets": np.array([])}, "non-empty"),
        (
            {"predictions": np.array([0.0]), "targets": np.array([0.0, 1.0])},
            "shape",
        ),
        (
            {"predictions": np.array([np.nan]), "targets": np.array([0.0])},
            "finite",
        ),
        (
            {
                "predictions": np.array([0.0]),
                "targets": np.array([0.0]),
                "event_probabilities": np.array([1.1]),
                "event_labels": np.array([1.0]),
            },
            "probabilities",
        ),
        (
            {
                "predictions": np.array([0.0]),
                "targets": np.array([0.0]),
                "event_probabilities": np.array([0.5]),
                "event_labels": np.array([0.5]),
            },
            "binary",
        ),
    ],
)
def test_cost_diagnostics_fail_closed(
    kwargs: dict[str, np.ndarray],
    message: str,
) -> None:
    from trade_rl.rl.cost_diagnostics import build_cost_head_diagnostics

    with pytest.raises(ValueError, match=message):
        build_cost_head_diagnostics(
            name="cost",
            adapter_gradient_norm=0.0,
            head_gradient_norm=0.0,
            **kwargs,
        )
