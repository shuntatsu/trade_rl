"""Phase 3 weekly expected-return scaling must reject non-finite output."""

from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.dev_realdata_study.phase3_sources import ScaledSource


class FakeInner:
    calls = 0
    null_packets = 0

    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = values

    def expected_simple_returns(self, snapshot: object) -> tuple[float, ...]:
        self.calls += 1
        return self.values


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(1e308, id="finite_input_overflow"),
        pytest.param(math.nan, id="nan_input"),
        pytest.param(math.inf, id="infinite_input"),
        pytest.param(-math.inf, id="negative_infinite_input"),
    ],
)
def test_weekly_scaled_source_rejects_nonfinite_output_atomically(
    bad_value: float,
) -> None:
    inner = FakeInner((0.01, bad_value, 0.0))
    adapter = ScaledSource(inner, 7.0)

    with pytest.raises(ValueError, match="scaled expected returns must be finite"):
        adapter.expected_simple_returns(object())
    assert adapter.history == []
    assert inner.calls == 1

    inner.values = (0.01, -0.02, 0.0)
    assert adapter.expected_simple_returns(object()) == pytest.approx(
        (0.07, -0.14, 0.0)
    )
    assert adapter.history == [(0.07, -0.14, 0.0)]
    assert inner.calls == 2


def test_weekly_scaled_source_preserves_existing_valid_conversion() -> None:
    inner = FakeInner((0.001, -0.002, 0.0))
    adapter = ScaledSource(inner, 7.0)
    result = adapter.expected_simple_returns(object())

    assert result == pytest.approx((0.007, -0.014, 0.0))
    assert adapter.history == [result]
