from __future__ import annotations

import copy
import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

import tools.market_b4_oracle.frozen_forecast as frozen_forecast_module
from tools.market_b4_oracle.allocation_choice import AllocationAssumptions
from tools.market_b4_oracle.frozen_forecast import (
    OwnedDecisionSnapshot,
    _validate_packet,
    bind_frozen_forecast_assumptions,
)
from trade_rl.strategies.forecasts.simple_return import (
    SimpleReturnRidgeModel,
    SimpleReturnTrainingSet,
)
from trade_rl.strategies.forecasts.simple_stream import (
    FrozenSimpleReturnStream,
    SimpleReturnPacket,
    SimpleReturnVintage,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.forecasts.supervised import (
    CausalForecastTrainingSet,
)
from trade_rl.strategies.forecasts.training_trace import (
    ForecastTrainingTrace,
)


def ns(hour: int) -> int:
    return int(
        (np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(hour, "h")).astype(
            np.int64
        )
    )


def fixture():
    trace = ForecastTrainingTrace(
        row_symbols=("ALPHA", "ALPHA"),
        start_times=np.array([ns(0), ns(1)], dtype="datetime64[ns]"),
        end_times=np.array([ns(1), ns(2)], dtype="datetime64[ns]"),
        start_available_at=np.array([ns(0), ns(1)], dtype="datetime64[ns]"),
        end_available_at=np.array([ns(1), ns(2)], dtype="datetime64[ns]"),
        start_close=np.array([100.0, 100.0]),
        end_close=np.array([110.0, 90.0]),
    )
    log_training = CausalForecastTrainingSet(
        feature_indices=(2, 4),
        feature_names=("feature_2", "feature_4"),
        features=np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float64),
        # Use the exact endpoint ratio and scalar log used by the causal
        # training-trace contract. np.log can differ by 1 ULP across libm builds.
        labels=np.array([math.log(110.0 / 100.0), math.log(90.0 / 100.0)]),
        sample_weights=np.ones(2, dtype=np.float64),
        label_end_times=trace.end_times,
        fit_cutoff=np.datetime64(ns(3), "ns"),
        horizon_hours=1,
        trace=trace,
    )
    training = SimpleReturnTrainingSet(log_training)
    model = SimpleReturnRidgeModel(
        feature_indices=(2, 4),
        feature_mean=np.array([0.0, 0.0]),
        feature_scale=np.array([1.0, 1.0]),
        coefficients=np.array([0.25, 0.0]),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=2,
        fit_cutoff=np.datetime64(ns(3), "ns"),
        fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
    )
    owner = SimpleReturnVintage(
        ForecastBlock(
            np.datetime64(ns(3), "ns"),
            np.datetime64(ns(3), "ns"),
            np.datetime64(ns(4), "ns"),
            np.datetime64(ns(8), "ns"),
            inference_delay_seconds=0,
        ),
        training,
        model,
        ("ALPHA", "BETA"),
    )
    stream = FrozenSimpleReturnStream(
        dataset_id="a" * 64,
        vintages=(owner,),
        packets=(
            SimpleReturnPacket(
                symbol="ALPHA",
                as_of=np.datetime64(ns(4), "ns"),
                source_available_at=np.datetime64(ns(4), "ns"),
                forecast_available_at=np.datetime64(ns(4), "ns"),
                horizon_end=np.datetime64(ns(5), "ns"),
                horizon_seconds=3600,
                expected_simple_return=0.125,
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
                vintage_digest=owner.digest,
                feature_values=(0.5, 0.7),
                decision_close=100.0,
            ),
            SimpleReturnPacket(
                symbol="ALPHA",
                as_of=np.datetime64(ns(5), "ns"),
                source_available_at=np.datetime64(ns(5), "ns"),
                forecast_available_at=np.datetime64(ns(5), "ns"),
                horizon_end=np.datetime64(ns(6), "ns"),
                horizon_seconds=3600,
                expected_simple_return=0.15,
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
                vintage_digest=owner.digest,
                feature_values=(0.6, 0.8),
                decision_close=101.0,
            ),
            SimpleReturnPacket(
                symbol="BETA",
                as_of=np.datetime64(ns(4), "ns"),
                source_available_at=np.datetime64(ns(4), "ns"),
                forecast_available_at=np.datetime64(ns(4), "ns"),
                horizon_end=np.datetime64(ns(5), "ns"),
                horizon_seconds=3600,
                expected_simple_return=0.2,
                fit_prefix_marginal_variance=training.fit_prefix_marginal_variance,
                vintage_digest=owner.digest,
                feature_values=(0.8, 0.9),
                decision_close=102.0,
            ),
        ),
    )
    source = OwnedDecisionSnapshot(
        dataset_id="a" * 64,
        symbol="ALPHA",
        decision_time_ns=ns(4),
        source_available_at_ns=ns(4),
        information_available=True,
        feature_names=(
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
            "feature_5",
        ),
        feature_values=(0.1, 0.2, 0.5, 0.4, 0.7, 0.6),
        feature_available=(True, True, True, True, True, True),
        dataset_close=100.0,
        book_mark=100.0,
        expected_horizon_seconds=3600,
    )
    costs = AllocationAssumptions(
        expected_simple_return=9.0,
        return_variance=9.0,
        buy_cost=0.001,
        sell_cost=0.002,
        exit_cost=0.003,
        funding_return=0.004,
        borrow_return=0.005,
        cash_return=0.006,
    )
    return stream.payload(), source, costs


def two_feature_fixture():
    payload, source, costs = fixture()
    payload = copy.deepcopy(payload)
    vintage = payload["vintages"][0]
    vintage["model"]["feature_indices"] = [0, 1]
    vintage["training"]["log_training"]["feature_indices"] = [0, 1]
    vintage["training"]["log_training"]["feature_names"] = [
        "feature_0",
        "feature_1",
    ]
    payload["packets"][0]["feature_values"] = [0.1, 0.2]
    source = replace(
        source,
        feature_names=("feature_0", "feature_1"),
        feature_values=(0.1, 0.2),
        feature_available=(True, True),
        observation_feature_indices=(0, 1),
    )
    return payload, source, costs


def test_maps_exact_source_packet_and_vintage_to_cost_assumptions():
    payload, source, costs = fixture()
    result = bind_frozen_forecast_assumptions(
        payload, source=source, cost_assumptions=costs
    )
    assert type(result) is AllocationAssumptions
    assert result.expected_simple_return == pytest.approx(0.125)
    assert result.return_variance == pytest.approx(0.01)
    assert (
        result.buy_cost,
        result.sell_cost,
        result.exit_cost,
        result.funding_return,
        result.borrow_return,
        result.cash_return,
    ) == (0.001, 0.002, 0.003, 0.004, 0.005, 0.006)


def test_uses_full_dataset_feature_indices_and_allows_unselected_unavailable_fields():
    payload, source, costs = fixture()
    source = replace(
        source,
        feature_available=(True, False, True, False, True, False),
        observation_feature_indices=(0, 2, 4),
    )
    result = bind_frozen_forecast_assumptions(
        payload, source=source, cost_assumptions=costs
    )
    assert result.expected_simple_return == pytest.approx(0.125)
    assert result.return_variance == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("availability", "observation_indices"),
    [
        ((False, False, True, False, True, False), (0, 2, 4)),
        ((True, True, False, False, True, False), (0,)),
    ],
)
def test_rejects_unavailable_observation_or_forecast_feature(
    availability, observation_indices
):
    payload, source, costs = fixture()
    source = replace(
        source,
        feature_available=availability,
        observation_feature_indices=observation_indices,
    )
    with pytest.raises(ValueError):
        bind_frozen_forecast_assumptions(payload, source=source, cost_assumptions=costs)


@pytest.mark.parametrize("indices", [(6,), (0, 0)])
def test_rejects_invalid_observation_feature_indices(indices):
    _, source, _ = fixture()
    with pytest.raises(ValueError):
        replace(source, observation_feature_indices=indices)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("information_available", False),
        ("source_available_at_ns", ns(5)),
        ("feature_available", (False,)),
        (
            "feature_names",
            ("feature_0", "feature_1", "wrong", "feature_3", "feature_4", "feature_5"),
        ),
        ("feature_values", (0.1, 0.2, 0.5000000000000001, 0.4, 0.7, 0.6)),
        ("dataset_close", 100.00000000000001),
        ("book_mark", 100.00000000000001),
        ("feature_available", (True, True, False, True, True, True)),
        ("decision_time_ns", ns(5)),
        ("expected_horizon_seconds", 7200),
        ("dataset_id", "b" * 64),
    ],
)
def test_rejects_owned_source_clock_dataset_feature_or_mark_mismatch(field, value):
    payload, source, costs = fixture()
    with pytest.raises(ValueError):
        bind_frozen_forecast_assumptions(
            payload,
            source=replace(source, **{field: value}),
            cost_assumptions=costs,
        )


@pytest.mark.parametrize(
    "field",
    ["prediction_start", "prediction_stop", "model_fit_time"],
)
def test_rejects_vintage_outside_its_declared_prediction_window(field):
    payload, source, costs = two_feature_fixture()
    block = payload["vintages"][0]["block"]
    if field == "prediction_start":
        block[field] = ns(5)
    elif field == "prediction_stop":
        block["prediction_start"] = ns(3)
        block[field] = ns(4)
    else:
        block[field] = ns(5)
        block["prediction_start"] = ns(5)
    with pytest.raises(ValueError):
        bind_frozen_forecast_assumptions(payload, source=source, cost_assumptions=costs)


@pytest.mark.parametrize("field", ["dataset_close", "book_mark"])
def test_rejects_nonpositive_owned_close_or_mark(field):
    _, source, _ = fixture()
    with pytest.raises(ValueError):
        replace(source, **{field: 0.0})


def test_rejects_nonpositive_packet_close():
    payload, source, costs = two_feature_fixture()
    payload = copy.deepcopy(payload)
    payload["packets"][0]["decision_close"] = 0.0
    with pytest.raises(ValueError):
        _validate_packet(payload["packets"][0])
    with pytest.raises(ValueError):
        bind_frozen_forecast_assumptions(payload, source=source, cost_assumptions=costs)


@pytest.mark.parametrize(
    ("mutation",),
    [
        ("duplicate_packet",),
        ("duplicate_vintage",),
        ("packet_extra_key",),
        ("packet_missing_key",),
        ("vintage_extra_key",),
        ("model_variance_mismatch",),
        ("packet_feature_order_mismatch",),
        ("forecast_index_order_mismatch",),
        ("forecast_index_out_of_range",),
        ("packet_variance_mismatch",),
        ("nonzero_inference_delay",),
        ("late_forecast_availability",),
        ("late_source_availability",),
        ("packet_horizon_mismatch",),
        ("packet_horizon_end_mismatch",),
        ("training_feature_mismatch",),
        ("bad_stream_schema",),
        ("stream_extra_key",),
        ("bad_availability_semantics",),
        ("dataset_identity_mismatch",),
    ],
)
def test_rejects_duplicate_or_closed_forecast_record_mismatch(mutation):
    payload, source, costs = fixture()
    payload = copy.deepcopy(payload)
    if mutation == "duplicate_packet":
        payload["packets"].append(copy.deepcopy(payload["packets"][0]))
    elif mutation == "duplicate_vintage":
        payload["vintages"].append(copy.deepcopy(payload["vintages"][0]))
    elif mutation == "packet_extra_key":
        payload["packets"][0]["unused"] = 1
    elif mutation == "packet_missing_key":
        payload["packets"][0].pop("snapshot_digest")
    elif mutation == "vintage_extra_key":
        payload["vintages"][0]["unused"] = 1
    elif mutation == "model_variance_mismatch":
        payload["vintages"][0]["model"]["fit_prefix_marginal_variance"] += 1.0
    elif mutation == "packet_feature_order_mismatch":
        payload["packets"][0]["feature_values"] = [0.7, 0.5]
    elif mutation == "forecast_index_order_mismatch":
        payload["vintages"][0]["model"]["feature_indices"] = [4, 2]
        payload["vintages"][0]["training"]["log_training"]["feature_indices"] = [
            4,
            2,
        ]
    elif mutation == "forecast_index_out_of_range":
        payload["vintages"][0]["model"]["feature_indices"] = [2, 6]
        payload["vintages"][0]["training"]["log_training"]["feature_indices"] = [
            2,
            6,
        ]
    elif mutation == "packet_variance_mismatch":
        payload["packets"][0]["fit_prefix_marginal_variance"] += 1.0
    elif mutation == "nonzero_inference_delay":
        payload["vintages"][0]["block"]["inference_delay_seconds"] = 1
        payload["packets"][0]["forecast_available_at"] += 1_000_000_000
    elif mutation == "late_forecast_availability":
        payload["packets"][0]["forecast_available_at"] += 1
    elif mutation == "late_source_availability":
        payload["packets"][0]["source_available_at"] += 1
    elif mutation == "packet_horizon_mismatch":
        payload["packets"][0]["horizon_seconds"] += 1
    elif mutation == "packet_horizon_end_mismatch":
        payload["packets"][0]["horizon_end"] += 1
    elif mutation == "training_feature_mismatch":
        payload["vintages"][0]["training"]["log_training"]["feature_names"] = ["other"]
    elif mutation == "bad_stream_schema":
        payload["schema"] = "other"
    elif mutation == "stream_extra_key":
        payload["unused"] = True
    elif mutation == "bad_availability_semantics":
        payload["availability_semantics"] = "unknown"
    elif mutation == "dataset_identity_mismatch":
        payload["dataset_id"] = "b" * 64
    with pytest.raises(ValueError):
        bind_frozen_forecast_assumptions(payload, source=source, cost_assumptions=costs)


def test_rejects_cost_assumptions_of_another_type():
    payload, source, _ = fixture()
    with pytest.raises(ValueError):
        bind_frozen_forecast_assumptions(
            payload, source=source, cost_assumptions=object()
        )


def test_frozen_forecast_has_no_trade_rl_production_imports():
    source = inspect.getsource(frozen_forecast_module)
    assert "trade_rl" not in source
    assert "fit_prequential_simple_ridge" not in source
    assert "predict_selected" not in source
