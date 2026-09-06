from __future__ import annotations

import math
from statistics import fmean, median

import numpy as np
import pytest


def _module():
    from trade_rl.workflows import universal_trade_rl_u2_selection

    return universal_trade_rl_u2_selection


def _panel_points():
    module = _module()
    rows: list[object] = []
    values = {
        ("development_future_1", 100): {
            0: {"DEV_A": 1.0, "DEV_B": 3.0},
            1: {"DEV_A": 5.0, "DEV_B": 7.0},
            2: {"DEV_A": 9.0, "DEV_B": 11.0},
        },
        ("development_future_1", 200): {
            0: {"DEV_A": 3.0, "DEV_B": 5.0},
            1: {"DEV_A": 7.0, "DEV_B": 9.0},
            2: {"DEV_A": 11.0, "DEV_B": 13.0},
        },
        ("development_future_2", 300): {
            0: {"DEV_A": -2.0, "DEV_B": 0.0},
            1: {"DEV_A": 1.0, "DEV_B": 3.0},
            2: {"DEV_A": 4.0, "DEV_B": 6.0},
        },
        ("development_future_2", 400): {
            0: {"DEV_A": -1.0, "DEV_B": 1.0},
            1: {"DEV_A": 2.0, "DEV_B": 4.0},
            2: {"DEV_A": 5.0, "DEV_B": 7.0},
        },
    }
    for (source_window, timestamp_ns), by_seed in reversed(tuple(values.items())):
        for training_seed in reversed((0, 1, 2)):
            for concrete_symbol in reversed(("DEV_A", "DEV_B")):
                rows.append(
                    module.UniversalTradeRLU2PairedExcessPoint(
                        training_seed=training_seed,
                        source_window=source_window,
                        concrete_symbol=concrete_symbol,
                        decision_timestamp_ns=timestamp_ns,
                        candidate_minus_cash_net_log_excess=by_seed[training_seed][
                            concrete_symbol
                        ],
                    )
                )
    return tuple(rows)


def test_u2_development_panel_reduces_symbols_then_training_seeds() -> None:
    module = _module()

    segments = module.reduce_universal_trade_rl_u2_development_panel(
        points=_panel_points(),
        expected_symbols=("DEV_A", "DEV_B"),
    )

    assert tuple(segment.source_window for segment in segments) == (
        "development_future_1",
        "development_future_2",
    )
    assert segments[0].decision_timestamps_ns == (100, 200)
    assert segments[0].net_log_excess == pytest.approx((6.0, 8.0))
    assert segments[1].decision_timestamps_ns == (300, 400)
    assert segments[1].net_log_excess == pytest.approx((2.0, 3.0))


def test_u2_development_panel_fails_closed_on_incomplete_timestamp_closure() -> None:
    module = _module()
    points = _panel_points()

    with pytest.raises(ValueError, match="closure|timestamp|seed|symbol|complete"):
        module.reduce_universal_trade_rl_u2_development_panel(
            points=points[:-1],
            expected_symbols=("DEV_A", "DEV_B"),
        )


@pytest.mark.parametrize("training_seed", (False, True))
def test_u2_paired_excess_point_rejects_boolean_training_seed(
    training_seed: bool,
) -> None:
    module = _module()

    with pytest.raises(ValueError, match="seed|closure"):
        module.UniversalTradeRLU2PairedExcessPoint(
            training_seed=training_seed,
            source_window="development_future_1",
            concrete_symbol="DEV_A",
            decision_timestamp_ns=100,
            candidate_minus_cash_net_log_excess=0.0,
        )


def _reference_segmented_bootstrap(
    segments: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = np.empty(2_000, dtype=np.float64)
    for draw in range(2_000):
        sampled_values: list[float] = []
        for values in segments:
            segment_length = len(values)
            block_length = min(
                segment_length,
                math.ceil(math.sqrt(segment_length)),
            )
            sampled_indices: list[int] = []
            while len(sampled_indices) < segment_length:
                start = int(rng.integers(0, segment_length))
                sampled_indices.extend(
                    (start + offset) % segment_length for offset in range(block_length)
                )
            sampled_values.extend(
                values[index] for index in sampled_indices[:segment_length]
            )
        means[draw] = fmean(sampled_values)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def _incorrect_flattened_bootstrap(
    segments: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    values = tuple(value for segment in segments for value in segment)
    rng = np.random.default_rng(0)
    block_length = math.ceil(math.sqrt(len(values)))
    means = np.empty(2_000, dtype=np.float64)
    for draw in range(2_000):
        sampled_indices: list[int] = []
        while len(sampled_indices) < len(values):
            start = int(rng.integers(0, len(values)))
            sampled_indices.extend(
                (start + offset) % len(values) for offset in range(block_length)
            )
        means[draw] = fmean(values[index] for index in sampled_indices[: len(values)])
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return float(lower), float(upper)


def test_u2_segmented_bootstrap_never_flattens_d1_d2_boundary() -> None:
    module = _module()
    raw_segments = (
        (10.0, 10.0, 10.0, -20.0, -20.0),
        (-5.0, -5.0, 8.0, 8.0, 8.0, 8.0, 8.0),
    )
    segments = tuple(
        module.UniversalTradeRLU2ReducedBootstrapSegment(
            source_window=source_window,
            decision_timestamps_ns=tuple(range(len(values))),
            net_log_excess=values,
        )
        for source_window, values in zip(
            ("development_future_1", "development_future_2"),
            raw_segments,
            strict=True,
        )
    )

    result = module.bootstrap_universal_trade_rl_u2_development_panel(segments=segments)
    expected_lower, expected_upper = _reference_segmented_bootstrap(raw_segments)
    flattened_lower, flattened_upper = _incorrect_flattened_bootstrap(raw_segments)

    assert result.resamples == 2_000
    assert result.bootstrap_seed == 0
    assert result.confidence_level == 0.95
    assert result.quantile_method == "linear"
    assert result.block_lengths == (3, 3)
    assert result.observed_mean == pytest.approx(
        fmean(value for segment in raw_segments for value in segment)
    )
    assert result.lower_ci == pytest.approx(expected_lower, abs=1e-12)
    assert result.upper_ci == pytest.approx(expected_upper, abs=1e-12)
    assert (result.lower_ci, result.upper_ci) != pytest.approx(
        (flattened_lower, flattened_upper),
        abs=1e-12,
    )
    assert result.passed is (result.lower_ci > 0.0)


def test_u2_panel_reduction_matches_explicit_equal_weight_then_median_oracle() -> None:
    points = _panel_points()
    grouped: dict[tuple[str, int], dict[int, list[float]]] = {}
    for point in points:
        key = (point.source_window, point.decision_timestamp_ns)
        grouped.setdefault(key, {}).setdefault(point.training_seed, []).append(
            point.candidate_minus_cash_net_log_excess
        )

    expected = {
        key: median(fmean(values) for values in by_seed.values())
        for key, by_seed in grouped.items()
    }

    module = _module()
    segments = module.reduce_universal_trade_rl_u2_development_panel(
        points=points,
        expected_symbols=("DEV_A", "DEV_B"),
    )
    observed = {
        (segment.source_window, timestamp_ns): value
        for segment in segments
        for timestamp_ns, value in zip(
            segment.decision_timestamps_ns,
            segment.net_log_excess,
            strict=True,
        )
    }
    assert observed == pytest.approx(expected)
