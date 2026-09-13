from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

MODULE = "research.issue529_portable_exp003"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0003 preregistration helper is not implemented"
    return import_module(MODULE)


def test_candidate_feature_names_append_only_registered_feature() -> None:
    module = _module()
    baseline = module.BASELINE_FEATURE_NAMES
    candidate = module.build_candidate_feature_names(baseline)
    assert candidate == baseline + (module.CANDIDATE_FEATURE_NAME,)
    assert len(candidate) == len(baseline) + 1
    assert len(set(candidate)) == len(candidate)
    with pytest.raises(RuntimeError, match="already present"):
        module.build_candidate_feature_names(candidate)


def test_formal_decision_rule_boundaries() -> None:
    module = _module()
    kind = module.ExperimentDecisionKind
    assert (
        module.formal_decision(
            positive_effects=4,
            median_excess=1e-9,
            candidate_positive_returns=5,
        )
        is kind.ACCEPT_CANDIDATE
    )
    for kwargs in (
        dict(positive_effects=2, median_excess=0.1, candidate_positive_returns=5),
        dict(positive_effects=5, median_excess=0.0, candidate_positive_returns=5),
        dict(positive_effects=5, median_excess=0.1, candidate_positive_returns=3),
    ):
        assert module.formal_decision(**kwargs) is kind.KEEP_BASELINE
    assert (
        module.formal_decision(
            positive_effects=3,
            median_excess=0.1,
            candidate_positive_returns=4,
        )
        is kind.INCONCLUSIVE
    )


def test_feature_diagnostics_only_require_identity_availability_and_finite_values() -> None:
    module = _module()
    names = tuple(f"feature_{index}" for index in range(57)) + (
        module.CANDIDATE_FEATURE_NAME,
    )
    timestamps = np.asarray(
        [
            "2022-12-31T22:00:00",
            "2022-12-31T23:00:00",
            "2023-01-01T00:00:00",
            "2023-01-01T01:00:00",
        ],
        dtype="datetime64[ns]",
    )
    features = np.zeros((4, 2, 58), dtype=np.float32)
    available = np.ones_like(features, dtype=np.bool_)
    available[0, 0, 57] = False
    dataset = SimpleNamespace(
        dataset_id=module.EXPECTED_DATASET_ID,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        feature_names=names,
        features=features,
        feature_available=available,
    )
    diagnostics = module.feature_diagnostics(
        dataset,
        fit_cutoff=np.datetime64("2023-01-01T00:00:00", "ns"),
        require_frozen_roster=False,
    )
    assert diagnostics["feature_index"] == 57
    assert diagnostics["fit_coverage"]["BTCUSDT"]["available_rows"] == 1
    assert diagnostics["fit_coverage"]["ETHUSDT"]["available_rows"] == 2
    assert diagnostics["evaluation_coverage"]["BTCUSDT"]["available_rows"] == 2
    assert diagnostics["evaluation_coverage"]["ETHUSDT"]["available_rows"] == 2
    assert diagnostics["no_return_target_or_pnl_relation_inspected"] is True


def test_feature_diagnostics_reject_nonfinite_available_value() -> None:
    module = _module()
    names = tuple(f"feature_{index}" for index in range(57)) + (
        module.CANDIDATE_FEATURE_NAME,
    )
    timestamps = np.asarray(
        ["2022-12-31T23:00:00", "2023-01-01T00:00:00"],
        dtype="datetime64[ns]",
    )
    features = np.zeros((2, 1, 58), dtype=np.float32)
    available = np.ones_like(features, dtype=np.bool_)
    features[0, 0, 57] = np.nan
    dataset = SimpleNamespace(
        dataset_id=module.EXPECTED_DATASET_ID,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        feature_names=names,
        features=features,
        feature_available=available,
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        module.feature_diagnostics(
            dataset,
            fit_cutoff=np.datetime64("2023-01-01T00:00:00", "ns"),
            require_frozen_roster=False,
        )
