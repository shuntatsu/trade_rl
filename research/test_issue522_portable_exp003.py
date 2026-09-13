from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytest

MODULE = "research.issue522_portable_exp003"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0003 structural helper is not implemented"
    return import_module(MODULE)


def test_candidate_feature_identity_requires_exact_name_and_index() -> None:
    module = _module()
    names = tuple(f"f{index}" for index in range(115)) + (
        "1d__log_return_4bar",
        "tail",
    )
    assert module.require_feature_index(
        names,
        name="1d__log_return_4bar",
        expected_index=115,
    ) == 115


def test_candidate_feature_identity_rejects_index_drift() -> None:
    module = _module()
    names = ("1d__log_return_4bar", "other")
    with pytest.raises(RuntimeError, match="feature index drift"):
        module.require_feature_index(
            names,
            name="1d__log_return_4bar",
            expected_index=115,
        )


def test_trend_transition_count_uses_frozen_hysteresis_semantics() -> None:
    module = _module()
    signal = np.asarray([0.0, 0.02, 0.015, 0.0, -0.02, -0.015, 0.0])
    available = np.ones(signal.shape, dtype=np.bool_)
    # FLAT -> LONG -> FLAT -> SHORT -> FLAT = four state changes.
    assert module.trend_transition_count(
        signal,
        available,
        entry_threshold=0.01,
        exit_threshold=0.0025,
    ) == 4


def test_trend_transition_count_flattens_on_unavailable_signal() -> None:
    module = _module()
    signal = np.asarray([0.02, 0.02, 0.02])
    available = np.asarray([True, False, True], dtype=np.bool_)
    # FLAT -> LONG -> FLAT -> LONG.
    assert module.trend_transition_count(
        signal,
        available,
        entry_threshold=0.01,
        exit_threshold=0.0025,
    ) == 3
