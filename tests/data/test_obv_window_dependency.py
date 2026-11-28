"""OBV features depend only on rows inside their declared source window.

Regression for an audit witness: OBV slope/change/acceleration were
differences of a segment-cumulative OBV, so the float value at ``t`` depended
(at ULP level) on rows before ``source_start[t]``. Once late-published base
rows join feature windows, that would let a still-unpublished earlier row
reach a published event, so the value must be exact in the window alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.features import calculate_feature_events


def _inputs(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    close = 100.0 * np.exp(np.cumsum(rng.standard_t(4, n) * 0.01))
    open_price = close * np.exp(rng.normal(0.0, 0.003, n))
    return {
        "open_price": open_price,
        "high": np.maximum(open_price, close) * 1.004,
        "low": np.minimum(open_price, close) * 0.996,
        "close": close,
        "volume": rng.lognormal(7.0, 0.5, n),
        "funding_rate": np.zeros(n),
        "funding_available": np.zeros(n, dtype=np.bool_),
    }


@pytest.mark.parametrize(
    "kind",
    (FeatureKind.OBV_SLOPE, FeatureKind.OBV_CHANGE, FeatureKind.OBV_ACCELERATION),
)
def test_obv_value_is_independent_of_rows_before_source_start(
    kind: FeatureKind,
) -> None:
    n, offset = 400, 100
    inputs = _inputs(n)
    spec = FeatureSpec("obv", kind, lookback=9)

    def run(start: int, volume_scale: float = 1.0):
        window = {key: value[start:].copy() for key, value in inputs.items()}
        window["volume"][:offset] *= volume_scale
        size = n - start
        return calculate_feature_events(
            spec,
            **window,
            row_present=np.ones(size, dtype=np.bool_),
            active=np.ones(size, dtype=np.bool_),
        )

    full_values, full_valid, full_start = run(0)
    late_values, late_valid, late_start = run(offset)
    # Rows before ``offset`` in the full history with a very different volume.
    noisy_values, noisy_valid, _ = run(0, volume_scale=1e7)

    inside = full_valid & (full_start >= offset)
    tail = inside[offset:]
    assert tail.sum() > 250
    np.testing.assert_array_equal(late_valid[tail], True)
    np.testing.assert_array_equal(late_start[tail] + offset, full_start[offset:][tail])
    np.testing.assert_array_equal(full_values[offset:][tail], late_values[tail])
    np.testing.assert_array_equal(noisy_valid, full_valid)
    np.testing.assert_array_equal(noisy_values[inside], full_values[inside])
