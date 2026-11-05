"""Preserve the established ISO input contract without NumPy timezone warnings."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from trade_rl.evaluation.runs.config import _normalized_timestamp


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-01-01T05:00:00+01", "2026-01-01T04:00:00"),
        ("2026-01-01T05:00:00+0100", "2026-01-01T04:00:00"),
        ("2026-01-01T05:00:00+01:00", "2026-01-01T04:00:00"),
        ("2026-01-01T04:00:00Z", "2026-01-01T04:00:00"),
        ("2026-01-01 05:00:00+01:00", "2026-01-01T04:00:00"),
        ("2026-01-01T05:00:00.123456789+01:00", "2026-01-01T04:00:00.123456789"),
        ("2026-01-01T05:00:00.1234567890+01", "2026-01-01T04:00:00.123456789"),
    ],
)
def test_timezone_iso_normalization_emits_no_numpy_deprecation(
    value: str, expected: str
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        actual = _normalized_timestamp(value, field="fit_cutoff")
    assert actual == np.datetime64(expected, "ns")


@pytest.mark.parametrize(
    "value",
    [
        "2026-01-01T05:00:00+01:00:00",
        "2026-01-01T05:00:00+01:00:00.0",
        "2026-01-01T05:00:00z",
        "2026-01-01t05:00:00+01:00",
        "20260101T050000+01:00",
        "2026-01-01T05:00:00,123456789+01:00",
        "2026-01-01T05:00.0000000001+01:00",
        "2026-01-01T05:00:00+01:00.0000000001",
        "2026-01-01T05:00:00+1:00",
        "2026-01-01T05:00:00+010000",
        "2026-01-01T05:00:00,123456789",
        "2026-01-01T05:00.0000000001",
        "2026-01-01T05.1",
    ],
)
def test_timezone_lexical_rejections_remain_strict(value: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(ValueError, match="fit_cutoff must be an ISO datetime"):
            _normalized_timestamp(value, field="fit_cutoff")
