"""Literal frozen preprocessing declaration, without a learner consumer."""

import hashlib
import importlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.strategies.rl.ppo_normalization import PPOFeatureNormalizer

MODULE = "trade_rl.strategies.rl.allocation_preprocessing"
EXPECTED = {
    "schema": "allocation_feature_preprocessing_v1",
    "statistics": {
        "schema": "ppo_feature_standardization_v1",
        "source_dataset_id": "6" * 64,
        "feature_indices": [0],
        "feature_names": ["signal"],
        "fit_symbol_indices": [0],
        "start_index": 0,
        "stop_index": 3,
        "mean": [1.0],
        "scale": [1.0],
        "usable_counts": [[2]],
    },
    "feature_config_digest": "0" * 64,
    "source_normalization_digest": "0" * 64,
    "fit_last_event_time_ns": 2,
    "fit_as_of_ns": 3,
    "policy_start_index": 3,
    "policy_start_time_ns": 3,
    "admitted_row_indices": [[0, 2]],
    "excluded_row_counts": [1],
    "fit_consumption_digest": "a" * 64,
    "row_admission": "all_selected_finite_available_at_row_clock_v1",
    "weighting": "equal_symbol_population_variance_v1",
    "scale_floor": 1e-12,
    "scale_fallback": 1.0,
    "transform": "frozen_float64_standardization_guarded_float32_v1",
}


def owner():
    try:
        return importlib.import_module(MODULE).AllocationFeaturePreprocessing
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        pytest.fail("allocation causal preprocessing declaration is missing")


def declaration(**changes):
    values = {k: v for k, v in EXPECTED["statistics"].items() if k != "schema"}
    kwargs = {
        k: EXPECTED[k]
        for k in (
            "feature_config_digest",
            "source_normalization_digest",
            "fit_last_event_time_ns",
            "fit_as_of_ns",
            "policy_start_index",
            "policy_start_time_ns",
            "fit_consumption_digest",
        )
    }
    kwargs.update(
        normalizer=PPOFeatureNormalizer(**values), admitted_row_indices=((0, 2),)
    )
    return owner()(**(kwargs | changes))


def transform(value, values=(4.0,), **changes):
    kwargs = dict(
        feature_names=("signal",),
        feature_config_digest="0" * 64,
        source_normalization_digest="0" * 64,
        decision_time_ns=3,
    )
    return value.transform(values, **(kwargs | changes))


def test_literal_payload_digest_roundtrip_and_feature_only_numeric_oracle():
    value = declaration()
    assert value.payload() == EXPECTED
    raw = json.dumps(
        EXPECTED, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    assert value.digest == hashlib.sha256(raw).hexdigest()
    assert owner().from_payload(json.loads(raw)) == value
    result = transform(value)
    assert (
        result.dtype == np.float32 and result.shape == (1,) and result.tolist() == [3.0]
    )


def test_payloads_and_statistics_are_detached_and_fields_are_frozen():
    value = declaration()
    payload = value.payload()
    payload["statistics"]["mean"][0] = 100
    payload["admitted_row_indices"][0].append(1)
    assert value.payload() == EXPECTED
    with pytest.raises(FrozenInstanceError):
        value.fit_as_of_ns = 4
    with pytest.raises(TypeError):
        value.admitted_row_indices[0][0] = 1
    result = transform(value)
    result[0] = 100
    assert transform(value).tolist() == [3.0]


@pytest.mark.parametrize(
    "field,bad",
    [
        ("fit_as_of_ns", True),
        ("fit_as_of_ns", 2),
        ("fit_as_of_ns", 4),
        ("fit_last_event_time_ns", -(2**63)),
        ("policy_start_time_ns", 2**63),
        ("policy_start_index", 2),
        ("policy_start_index", True),
        ("feature_config_digest", "bad"),
        ("source_normalization_digest", "bad"),
        ("fit_consumption_digest", "bad"),
        ("admitted_row_indices", ((0, 1, 2),)),
        ("admitted_row_indices", ((2, 0),)),
        ("admitted_row_indices", ((0, True),)),
        ("admitted_row_indices", ((0, 3),)),
        ("admitted_row_indices", [[0, 2]]),
    ],
)
def test_constructor_rejects_invalid_clock_identity_and_joint_row_counts(field, bad):
    with pytest.raises(ValueError):
        declaration(**{field: bad})


@pytest.mark.parametrize(
    "field,changed",
    [
        ("feature_config_digest", "1" * 64),
        ("source_normalization_digest", "1" * 64),
        ("fit_consumption_digest", "b" * 64),
        ("policy_start_index", 4),
        ("policy_start_time_ns", 4),
        ("fit_as_of_ns", 2),
    ],
)
def test_every_variable_clock_or_source_identity_changes_digest(field, changed):
    value = declaration(fit_last_event_time_ns=0)
    assert replace(value, **{field: changed}).digest != value.digest


@pytest.mark.parametrize(
    "kind", ("extra", "missing", "fixed", "statistics_extra", "count", "path", "cycle")
)
def test_closed_reader_rejects_tampering_and_non_json(kind):
    payload = deepcopy(EXPECTED)
    if kind == "extra":
        payload["extra"] = 0
    elif kind == "missing":
        del payload["fit_as_of_ns"]
    elif kind == "fixed":
        payload["scale_fallback"] = True
    elif kind == "statistics_extra":
        payload["statistics"]["extra"] = 0
    elif kind == "count":
        payload["excluded_row_counts"] = [0]
    elif kind == "path":
        payload["statistics"]["feature_names"] = [Path("signal")]
    else:
        payload["cycle"] = payload
    with pytest.raises(ValueError):
        owner().from_payload(payload)


@pytest.mark.parametrize(
    "values", ((True,), (np.nan,), (np.inf,), (1e100,), (1.0, 2.0))
)
def test_feature_transform_rejects_invalid_values_or_float32_overflow(values):
    with pytest.raises(ValueError):
        transform(declaration(), values)


@pytest.mark.parametrize(
    "changes",
    [
        dict(feature_names=("other",)),
        dict(feature_config_digest="1" * 64),
        dict(source_normalization_digest="1" * 64),
        dict(decision_time_ns=2),
        dict(decision_time_ns=True),
    ],
)
def test_feature_transform_requires_frozen_feature_build_and_application_clock(changes):
    with pytest.raises(ValueError):
        transform(declaration(), **changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", [np.inf]),
        ("scale", [0.0]),
        ("scale", [1e-12]),
        ("usable_counts", [[True]]),
    ],
)
def test_reader_rejects_invalid_frozen_numerical_statistics(field, value):
    payload = deepcopy(EXPECTED)
    payload["statistics"][field] = value
    with pytest.raises(ValueError):
        owner().from_payload(payload)


def test_constructor_rejects_scale_that_should_have_used_fixed_fallback():
    statistics = replace(declaration().normalizer, scale=(1e-13,))
    with pytest.raises(ValueError):
        declaration(normalizer=statistics)
