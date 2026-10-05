"""Independent unit, population-variance, and legacy Ridge numeric oracles."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.strategies.forecasts.ridge import _fit_ridge_training_set
from trade_rl.strategies.forecasts.simple_stream import (
    FrozenSimpleReturnStream,
    SimpleReturnPacket,
    SimpleReturnVintage,
)
from trade_rl.strategies.forecasts.simple_stream import (
    _model_payload as simple_model_payload,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock, _model_payload
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet
from trade_rl.strategies.forecasts.training_trace import ForecastTrainingTrace


def api() -> ModuleType:
    try:
        return importlib.import_module("trade_rl.strategies.forecasts.simple_return")
    except ModuleNotFoundError as error:
        if error.name != "trade_rl.strategies.forecasts.simple_return":
            raise
        pytest.fail("Missing direct-simple numerical foundation", pytrace=False)


def traced_training(
    *,
    starts: tuple[float, ...] = (100.0, 200.0),
    ends: tuple[float, ...] = (200.0, 100.0),
    weights: tuple[float, ...] = (1.0, 1.0),
) -> CausalForecastTrainingSet:
    count = len(starts)
    times = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        count
    ) * np.timedelta64(1, "h")
    trace = ForecastTrainingTrace(
        row_symbols=tuple("S0" for _ in starts),
        start_times=times,
        end_times=times + np.timedelta64(1, "h"),
        start_available_at=times,
        end_available_at=times + np.timedelta64(1, "h"),
        start_close=np.array(starts),
        end_close=np.array(ends),
    )
    return CausalForecastTrainingSet(
        feature_indices=(0,),
        features=np.full((count, 1), 7.0),
        labels=np.array([math.log(end / start) for start, end in zip(starts, ends)]),
        label_end_times=trace.end_times,
        sample_weights=np.array(weights),
        fit_cutoff=times[-1] + np.timedelta64(2, "h"),
        horizon_hours=1,
        trace=trace,
        feature_names=("constant",),
    )


def test_direct_simple_mean_is_not_transformed_mean_log() -> None:
    module = api()
    original = traced_training()
    training = module.SimpleReturnTrainingSet(original)
    model = module._fit_simple_return_training(training, alpha=1.0)
    np.testing.assert_array_equal(training.labels, [1.0, -0.5])
    assert math.expm1(_fit_ridge_training_set(original, alpha=1.0).intercept) == 0.0
    assert model.intercept == 0.25
    assert model.predict_selected(np.array([7.0])) == 0.25
    assert model.predict(np.array([7.0, 123.0])) == 0.25
    assert model.fit_prefix_marginal_variance == 0.5625
    assert training.fit_prefix_marginal_variance == 0.5625
    assert model.return_unit == "expected_simple_return"
    assert model.valuation_basis == "same_close_price_return"
    assert model.variance_kind == "fit_prefix_marginal_variance"
    assert type(model) is module.SimpleReturnRidgeModel


def test_variance_uses_same_weighted_population_and_no_annualization() -> None:
    module = api()
    original = traced_training(
        starts=(100.0, 100.0, 100.0),
        ends=(50.0, 100.0, 200.0),
        weights=(0.5, 0.5, 2.0),
    )
    training = module.SimpleReturnTrainingSet(original)
    model = module._fit_simple_return_training(training, alpha=2.0)
    assert model.intercept == pytest.approx(7.0 / 12.0)
    assert model.fit_prefix_marginal_variance == pytest.approx(53.0 / 144.0)
    np.testing.assert_array_equal(model.coefficients, [0.0])
    assert training.features is original.features
    assert training.sample_weights is original.sample_weights
    assert training.trace is original.trace
    assert training.label_end_times is original.label_end_times
    assert training.feature_indices == original.feature_indices
    assert training.selected_feature_names == original.feature_names
    assert training.n_samples == original.n_samples
    assert training.horizon_hours == original.horizon_hours
    assert training.fit_cutoff == original.fit_cutoff


def test_nonconstant_simple_fit_preserves_absolute_regularization_weights() -> None:
    module = api()
    original = replace(
        traced_training(
            starts=(100.0, 100.0, 100.0),
            ends=(100.0, 150.0, 200.0),
            weights=(1.0, 1.0, 1.0),
        ),
        features=np.arange(3.0).reshape(-1, 1),
    )
    model = module._fit_simple_return_training(
        module.SimpleReturnTrainingSet(original), alpha=1.0
    )
    reduced = module._fit_simple_return_training(
        module.SimpleReturnTrainingSet(
            replace(original, sample_weights=original.sample_weights / 2.0)
        ),
        alpha=1.0,
    )
    # x=0,1,2 and y=0,.5,1: Gram=3 before alpha=1, so slope=.375.
    assert model.intercept == 0.5
    assert model.predict_selected(np.array([3.0])) == pytest.approx(1.25)
    assert reduced.predict_selected(np.array([3.0])) == pytest.approx(1.1)
    assert model.fit_prefix_marginal_variance == pytest.approx(1.0 / 6.0)
    assert reduced.fit_prefix_marginal_variance == model.fit_prefix_marginal_variance


def test_simple_scope_binds_source_rows_direct_labels_units_and_proxy() -> None:
    module = api()
    original = traced_training()
    training = module.SimpleReturnTrainingSet(original)
    payload = training.scope_payload()
    assert payload["schema"] == "simple_return_training_scope_v1"
    assert payload["log_training"] == original.scope_payload()
    assert payload["labels"] == [1.0, -0.5]
    assert payload["return_unit"] == "expected_simple_return"
    assert payload["valuation_basis"] == "same_close_price_return"
    assert payload["variance_kind"] == "fit_prefix_marginal_variance"
    assert payload["fit_prefix_marginal_variance"] == 0.5625
    assert training.scope_digest == content_digest(payload)
    payload["labels"][0] = 999.0
    assert training.scope_payload()["labels"] == [1.0, -0.5]
    assert original.labels[0] == math.log(2.0)
    model = module._fit_simple_return_training(training, alpha=1.0)
    for array in (
        training.labels,
        training.features,
        training.sample_weights,
        model.feature_mean,
        model.feature_scale,
        model.coefficients,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("bad", [None, False, 1, "training"])
def test_simple_training_requires_actual_traced_scope(bad: object) -> None:
    with pytest.raises(ValueError):
        api().SimpleReturnTrainingSet(bad)


def test_simple_training_rejects_untraced_or_invalid_feature_scope() -> None:
    module = api()
    original = traced_training()
    with pytest.raises(ValueError):
        module.SimpleReturnTrainingSet(replace(original, trace=None))
    for indices in ((True,), (-1,)):
        with pytest.raises(ValueError):
            module.SimpleReturnTrainingSet(replace(original, feature_indices=indices))


@pytest.mark.parametrize("alpha", [True, False, 0.0, -1.0, np.nan, np.inf])
def test_simple_fit_rejects_invalid_alpha(alpha: float) -> None:
    module = api()
    with pytest.raises(ValueError):
        module._fit_simple_return_training(
            module.SimpleReturnTrainingSet(traced_training()), alpha=alpha
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_indices", (True,)),
        ("feature_indices", (-1,)),
        ("feature_mean", np.array([True])),
        ("feature_mean", np.array([1.0 + 2.0j])),
        ("feature_mean", np.array(["1.0"])),
        ("feature_scale", np.array([0.0])),
        ("coefficients", np.array([np.inf])),
        ("intercept", True),
        ("intercept", np.nan),
        ("alpha", True),
        ("alpha", 0.0),
        ("n_samples", True),
        ("horizon_hours", True),
        ("fit_cutoff", np.datetime64("NaT")),
        ("fit_prefix_marginal_variance", -1.0),
        ("fit_prefix_marginal_variance", True),
        ("fit_prefix_marginal_variance", np.nan),
        ("return_unit", "log_return"),
        ("valuation_basis", "held_quantity_total_return"),
        ("variance_kind", "conditional_variance"),
    ],
)
def test_simple_model_rejects_invalid_metadata(field: str, value: object) -> None:
    module = api()
    model = module._fit_simple_return_training(
        module.SimpleReturnTrainingSet(traced_training()), alpha=1.0
    )
    with pytest.raises(ValueError):
        replace(model, **{field: value})


def test_simple_prediction_validates_layout_and_finite_inputs() -> None:
    module = api()
    model = module._fit_simple_return_training(
        module.SimpleReturnTrainingSet(traced_training()), alpha=1.0
    )
    for features in (np.array([]), np.array([np.nan]), np.array([np.inf])):
        with pytest.raises(ValueError):
            model.predict_selected(features)
    with pytest.raises(ValueError):
        model.predict(np.array([]))


@pytest.mark.parametrize(
    "features", [np.array([True]), np.array([1.0 + 2.0j]), np.array(["7.0"])]
)
def test_simple_prediction_rejects_non_real_feature_types(features: np.ndarray) -> None:
    module = api()
    model = module._fit_simple_return_training(
        module.SimpleReturnTrainingSet(traced_training()), alpha=1.0
    )
    with pytest.raises(ValueError):
        model.predict_selected(features)
    with pytest.raises(ValueError):
        model.predict(features)


def test_simple_prediction_rejects_overflow_without_fallback() -> None:
    module = api()
    model = replace(
        module._fit_simple_return_training(
            module.SimpleReturnTrainingSet(traced_training()), alpha=1.0
        ),
        coefficients=np.array([1e308]),
    )
    with pytest.raises(ValueError):
        model.predict_selected(np.array([1e308]))


@pytest.mark.parametrize("prediction", [-2.0, -1.0, 0.5])
def test_simple_prediction_has_no_invalid_return_clipping(prediction: float) -> None:
    module = api()
    model = replace(
        module._fit_simple_return_training(
            module.SimpleReturnTrainingSet(traced_training()), alpha=1.0
        ),
        intercept=prediction,
    )
    if prediction < -1.0:
        with pytest.raises(ValueError):
            model.predict_selected(np.array([7.0]))
    else:
        assert model.predict_selected(np.array([7.0])) == prediction


def test_simple_training_rejects_nonfinite_weight_total() -> None:
    original = traced_training(weights=(1e308, 1e308))
    with pytest.raises(ValueError):
        api().SimpleReturnTrainingSet(original)


def test_simple_training_rejects_overflowed_variance() -> None:
    original = traced_training(starts=(1.0, 1.0), ends=(1.0, 1e200))
    with pytest.raises(ValueError):
        api().SimpleReturnTrainingSet(original)


@pytest.mark.parametrize(
    ("field", "scalar"),
    [
        ("alpha", np.float32(1.25)),
        ("alpha", np.float64(1.25)),
        ("alpha", np.int64(2)),
        ("intercept", np.float32(0.25)),
        ("intercept", np.float64(0.25)),
        ("intercept", np.int64(1)),
        ("fit_prefix_marginal_variance", np.float32(0.5625)),
        ("fit_prefix_marginal_variance", np.float64(0.5625)),
        ("fit_prefix_marginal_variance", np.int64(1)),
    ],
)
def test_model_admitted_numpy_scalar_has_builtin_float_payload(
    field: str, scalar: object
) -> None:
    module = api()
    original = module._fit_simple_return_training(
        module.SimpleReturnTrainingSet(traced_training()), alpha=1.0
    )
    model = replace(original, **{field: scalar})
    payload = simple_model_payload(model)
    encoded = canonical_json_bytes(payload)
    expected = replace(original, **{field: float(scalar)})
    assert encoded == canonical_json_bytes(simple_model_payload(expected))
    assert type(getattr(model, field)) is float
    assert getattr(model, field).hex() == float(scalar).hex()


@pytest.mark.parametrize("alpha", [np.float32(1.25), np.int64(2)])
def test_numpy_alpha_fit_vintage_stream_roundtrip(alpha: float) -> None:
    module = api()
    training = module.SimpleReturnTrainingSet(traced_training())
    model = replace(
        module._fit_simple_return_training(training, alpha=alpha),
        intercept=np.float32(0.25),
        fit_prefix_marginal_variance=np.float32(0.5625),
    )
    block = ForecastBlock(
        fit_cutoff=training.fit_cutoff,
        model_fit_time=training.fit_cutoff + np.timedelta64(1, "h"),
        prediction_start=training.fit_cutoff + np.timedelta64(2, "h"),
        prediction_stop=training.fit_cutoff + np.timedelta64(3, "h"),
    )
    vintage = SimpleReturnVintage(block, training, model, ("S0",))
    packet = SimpleReturnPacket(
        symbol="S0",
        as_of=block.prediction_start,
        source_available_at=block.prediction_start,
        forecast_available_at=block.prediction_start,
        horizon_end=block.prediction_start + np.timedelta64(1, "h"),
        horizon_seconds=3600,
        expected_simple_return=model.predict_selected(np.array([7.0])),
        fit_prefix_marginal_variance=model.fit_prefix_marginal_variance,
        vintage_digest=vintage.digest,
        feature_values=(7.0,),
        decision_close=100.0,
    )
    stream = FrozenSimpleReturnStream("1" * 64, (vintage,), (packet,))
    serialized = json.loads(canonical_json_bytes(stream.payload()))
    restored = FrozenSimpleReturnStream.from_payload(
        serialized, expected_digest=stream.digest
    )
    assert restored.payload() == stream.payload()
    assert restored.digest == stream.digest
    assert restored.vintages[0].model.predict_selected(np.array([7.0])) == 0.25
    for field in ("alpha", "intercept", "fit_prefix_marginal_variance"):
        assert type(getattr(model, field)) is float
        assert getattr(restored.vintages[0].model, field) == getattr(model, field)


def test_old_weighted_solver_keeps_frozen_legacy_operation_order() -> None:
    x = np.array(
        [
            [0.125, 3.0, 7.0],
            [-2.0, 0.25, 7.0],
            [4.0, -1.5, 7.0],
            [1.75, 2.0, 7.0],
            [-0.5, -3.0, 7.0],
        ]
    )
    y = np.array([0.03, -0.017, 0.125, -0.25, 0.001])
    weights = np.array([0.75, 1.25, 0.5, 2.0, 1.5])
    training = CausalForecastTrainingSet(
        feature_indices=(2, 0, 4),
        features=x,
        labels=y,
        label_end_times=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(1, 6) * np.timedelta64(1, "h"),
        sample_weights=weights,
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )
    # Frozen pre-extraction arithmetic, independent of the shared solver.
    total = float(weights.sum())
    mean = np.sum(x * weights[:, None], axis=0) / total
    centered_x = x - mean
    variance = np.sum(centered_x**2 * weights[:, None], axis=0) / total
    raw_scale = np.sqrt(variance)
    scale = np.where(raw_scale > 1e-12, raw_scale, 1.0)
    standardized = centered_x / scale
    intercept = float(np.dot(weights, y) / total)
    weighted_x = standardized * np.sqrt(weights)[:, None]
    weighted_y = (y - intercept) * np.sqrt(weights)
    coefficients = np.linalg.solve(
        weighted_x.T @ weighted_x + 1.75 * np.eye(3, dtype=np.float64),
        weighted_x.T @ weighted_y,
    )
    model = _fit_ridge_training_set(training, alpha=1.75)
    assert model.feature_mean.tobytes() == mean.tobytes()
    assert model.feature_scale.tobytes() == scale.tobytes()
    assert model.coefficients.tobytes() == coefficients.tobytes()
    assert model.intercept.hex() == intercept.hex()
    selected = np.array([1.25, -0.75, 7.0])
    assert (
        model.predict_selected(selected).hex()
        == float(intercept + ((selected - mean) / scale) @ coefficients).hex()
    )


def test_old_log_solver_keeps_fixed_preextraction_bits_and_payload() -> None:
    # Captured before extraction on e473a75e2. Orthogonal columns keep the
    # normalized Gram matrix exactly diagonal across supported BLAS runtimes.
    training = CausalForecastTrainingSet(
        feature_indices=(2, 0, 4),
        features=np.array(
            [[-1.0, -1.0, 7.0], [-1.0, 1.0, 7.0], [1.0, -1.0, 7.0], [1.0, 1.0, 7.0]]
        ),
        labels=np.array([0.125, -0.25, 0.5, -0.125]),
        label_end_times=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(1, 5) * np.timedelta64(1, "h"),
        sample_weights=np.ones(4),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )
    model = _fit_ridge_training_set(training, alpha=2.0)
    assert [float(value).hex() for value in model.feature_mean] == [
        "0x0.0p+0",
        "0x0.0p+0",
        "0x1.c000000000000p+2",
    ]
    assert [float(value).hex() for value in model.feature_scale] == [
        "0x1.0000000000000p+0"
    ] * 3
    assert [float(value).hex() for value in model.coefficients] == [
        "0x1.5555555555555p-4",
        "-0x1.5555555555555p-3",
        "0x0.0p+0",
    ]
    assert model.intercept.hex() == "0x1.0000000000000p-4"
    assert (
        model.predict_selected(np.array([0.5, -0.25, 7.0])).hex()
        == "0x1.2aaaaaaaaaaaap-3"
    )
    assert (
        content_digest(_model_payload(model))
        == "742cc535f85b186993217a8c2c7a8e1eaac62854cd8ea1db4f664bd0fc673e01"
    )
