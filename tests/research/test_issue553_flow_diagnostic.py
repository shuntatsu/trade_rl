from __future__ import annotations

import math

import numpy as np
import pytest

from tools.issue553_flow_diagnostic import (
    HourlyFlowPoint,
    aggregate_completed_hour_flows,
    align_next_executable_hour_returns,
    centered_ols_stats,
)

_HOUR_NS = 3_600_000_000_000
_CUTOFF_NS = int(np.datetime64("2023-01-01T00:00:00", "ns").astype(np.int64))


def _ns(text: str) -> int:
    return int(np.datetime64(text, "ns").astype(np.int64))


def test_aggregate_completed_hour_flow_uses_buyer_maker_side_exactly() -> None:
    timestamps = np.asarray(
        [
            np.datetime64("2022-01-01T00:10:00", "ns"),
            np.datetime64("2022-01-01T00:20:00", "ns"),
            np.datetime64("2022-01-01T00:30:00", "ns"),
        ]
    )
    prices = np.asarray([2.0, 2.0, 2.0])
    quantities = np.asarray([5.0, 5.0, 5.0])
    buyer_is_maker = np.asarray([False, False, True])

    points = aggregate_completed_hour_flows(
        timestamps=timestamps,
        prices=prices,
        quantities=quantities,
        buyer_is_maker=buyer_is_maker,
    )

    assert len(points) == 1
    assert points[0].hour_end_ns == _ns("2022-01-01T01:00:00")
    assert points[0].buy_taker_notional == pytest.approx(20.0)
    assert points[0].sell_taker_notional == pytest.approx(10.0)
    assert points[0].flow_imbalance == pytest.approx(1.0 / 3.0)


def test_alignment_uses_next_executable_hour_not_same_hour() -> None:
    timestamps_ns = np.asarray(
        [_ns(f"2022-01-01T0{hour}:00:00") for hour in range(1, 6)],
        dtype=np.int64,
    )
    # For decision row t=1 (02:00 close), the label must use open[2] -> open[3].
    opens = np.asarray([100.0, 1_000.0, 100.0, 110.0, 999.0])
    active = np.ones(5, dtype=np.bool_)
    tradable = np.ones(5, dtype=np.bool_)
    flow = (HourlyFlowPoint(_ns("2022-01-01T02:00:00"), 1.0, 0.0, 1.0),)

    samples = align_next_executable_hour_returns(
        flow_points=flow,
        timestamps_ns=timestamps_ns,
        open_prices=opens,
        active=active,
        tradable=tradable,
        cutoff_ns=_CUTOFF_NS,
    )

    assert len(samples) == 1
    assert samples[0].decision_timestamp_ns == _ns("2022-01-01T02:00:00")
    assert samples[0].execution_open == pytest.approx(100.0)
    assert samples[0].label_end_open == pytest.approx(110.0)
    assert samples[0].log_return == pytest.approx(math.log(1.1))


def test_alignment_requires_active_tradable_and_exact_hour_continuity() -> None:
    base = _ns("2022-01-01T01:00:00")
    timestamps_ns = np.asarray([base + i * _HOUR_NS for i in range(5)], dtype=np.int64)
    opens = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
    flow = (HourlyFlowPoint(base + _HOUR_NS, 2.0, 1.0, 1.0 / 3.0),)

    active = np.ones(5, dtype=np.bool_)
    tradable = np.ones(5, dtype=np.bool_)
    tradable[2] = False
    assert not align_next_executable_hour_returns(
        flow_points=flow,
        timestamps_ns=timestamps_ns,
        open_prices=opens,
        active=active,
        tradable=tradable,
        cutoff_ns=_CUTOFF_NS,
    )

    tradable[2] = True
    gapped = timestamps_ns.copy()
    gapped[2:] += _HOUR_NS
    assert not align_next_executable_hour_returns(
        flow_points=flow,
        timestamps_ns=gapped,
        open_prices=opens,
        active=active,
        tradable=tradable,
        cutoff_ns=_CUTOFF_NS,
    )


def test_alignment_rejects_2023_event_or_label_endpoint() -> None:
    timestamps_ns = np.asarray(
        [
            _ns("2022-12-31T21:00:00"),
            _ns("2022-12-31T22:00:00"),
            _ns("2022-12-31T23:00:00"),
            _ns("2023-01-01T00:00:00"),
            _ns("2023-01-01T01:00:00"),
        ],
        dtype=np.int64,
    )
    opens = np.asarray([100.0, 101.0, 102.0, 103.0, 104.0])
    active = np.ones(5, dtype=np.bool_)
    tradable = np.ones(5, dtype=np.bool_)

    # 22:00 decision is valid: endpoints 22:00 and 23:00 are both pre-cutoff.
    valid = align_next_executable_hour_returns(
        flow_points=(HourlyFlowPoint(_ns("2022-12-31T22:00:00"), 2.0, 1.0, 1.0 / 3.0),),
        timestamps_ns=timestamps_ns,
        open_prices=opens,
        active=active,
        tradable=tradable,
        cutoff_ns=_CUTOFF_NS,
    )
    assert len(valid) == 1

    # 23:00 decision would end at the 2023-01-01 00:00 open and is forbidden.
    invalid = align_next_executable_hour_returns(
        flow_points=(HourlyFlowPoint(_ns("2022-12-31T23:00:00"), 2.0, 1.0, 1.0 / 3.0),),
        timestamps_ns=timestamps_ns,
        open_prices=opens,
        active=active,
        tradable=tradable,
        cutoff_ns=_CUTOFF_NS,
    )
    assert not invalid


def test_centered_ols_is_with_intercept_and_rejects_zero_variance() -> None:
    stats = centered_ols_stats(
        x=(1.0, 2.0, 4.0, 8.0),
        y=(5.0, 8.0, 14.0, 26.0),  # y = 2 + 3*x
    )
    assert stats.count == 4
    assert stats.beta == pytest.approx(3.0)
    assert stats.correlation == pytest.approx(1.0)
    assert stats.centered_denominator > 0.0

    with pytest.raises(ValueError, match="variance"):
        centered_ols_stats(x=(1.0, 1.0, 1.0), y=(1.0, 2.0, 3.0))


def test_aggregate_rejects_nonfinite_or_nonpositive_trade_inputs() -> None:
    timestamps = np.asarray([np.datetime64("2022-01-01T00:10:00", "ns")])
    with pytest.raises(ValueError, match="prices"):
        aggregate_completed_hour_flows(
            timestamps=timestamps,
            prices=np.asarray([math.nan]),
            quantities=np.asarray([1.0]),
            buyer_is_maker=np.asarray([False]),
        )
    with pytest.raises(ValueError, match="quantities"):
        aggregate_completed_hour_flows(
            timestamps=timestamps,
            prices=np.asarray([1.0]),
            quantities=np.asarray([0.0]),
            buyer_is_maker=np.asarray([False]),
        )
