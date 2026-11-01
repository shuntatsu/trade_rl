from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    FoldOOSResult,
    StitchMode,
    stitch_oos,
)
from trade_rl.evaluation.series import ReturnKind, ReturnSeries


def oos(
    fold_index: int,
    start: int,
    values: tuple[float, ...],
    *,
    kind: ReturnKind = ReturnKind.BASE_BAR,
    periods_per_year: int = 8_760,
    elapsed_years: float | None = None,
    opening_state_digest: str | None = None,
    closing_state_digest: str | None = None,
) -> FoldOOSResult:
    return FoldOOSResult(
        fold_index=fold_index,
        start=start,
        stop=start + len(values),
        returns=ReturnSeries(
            values=values,
            kind=kind,
            periods_per_year=periods_per_year,
            elapsed_years=elapsed_years,
        ),
        opening_state_digest=opening_state_digest,
        closing_state_digest=closing_state_digest,
    )


_CLOCK_TEST_RETURNS = (0.01, -0.01, 0.02, -0.005)


def paired_oos(
    mode: StitchMode,
    *,
    first_elapsed_years: float | None,
    second_elapsed_years: float | None,
) -> tuple[FoldOOSResult, FoldOOSResult]:
    continuous = mode is StitchMode.CONTINUOUS_ACCOUNT
    return (
        oos(
            0,
            10,
            _CLOCK_TEST_RETURNS[:2],
            periods_per_year=252,
            elapsed_years=first_elapsed_years,
            opening_state_digest="start" if continuous else None,
            closing_state_digest="middle" if continuous else None,
        ),
        oos(
            1,
            12 if continuous else 15,
            _CLOCK_TEST_RETURNS[2:],
            periods_per_year=252,
            elapsed_years=second_elapsed_years,
            opening_state_digest="middle" if continuous else None,
            closing_state_digest="end" if continuous else None,
        ),
    )


def literal_sharpe(
    values: tuple[float, ...],
    *,
    elapsed_years: float | None = None,
    periods_per_year: int = 252,
) -> float:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    frequency = (
        len(values) / elapsed_years
        if elapsed_years is not None
        else float(periods_per_year)
    )
    return mean / math.sqrt(variance) * math.sqrt(frequency)


def test_stitch_oos_sorts_folds_chronologically() -> None:
    result = stitch_oos(
        (
            oos(1, 12, (0.03, 0.04)),
            oos(0, 10, (0.01, 0.02)),
            oos(2, 14, (-0.01, 0.00)),
        )
    )

    assert result.returns.values == (0.01, 0.02, 0.03, 0.04, -0.01, 0.00)
    assert result.fold_indices == (0, 1, 2)
    assert result.boundaries == ((10, 12), (12, 14), (14, 16))
    assert result.mode is StitchMode.INDEPENDENT_FOLDS
    assert result.gaps == ()


def test_independent_stitch_records_gaps_without_claiming_continuity() -> None:
    result = stitch_oos(
        (
            oos(0, 10, (0.01, 0.02)),
            oos(1, 15, (0.03, 0.04)),
        )
    )

    assert result.mode is StitchMode.INDEPENDENT_FOLDS
    assert result.gaps == ((12, 15),)


def test_continuous_stitch_requires_contiguous_ranges() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        stitch_oos(
            (
                oos(
                    0,
                    10,
                    (0.01, 0.02),
                    opening_state_digest="start",
                    closing_state_digest="middle",
                ),
                oos(
                    1,
                    15,
                    (0.03, 0.04),
                    opening_state_digest="middle",
                    closing_state_digest="end",
                ),
            ),
            mode=StitchMode.CONTINUOUS_ACCOUNT,
        )


def test_continuous_stitch_requires_state_handoff_identity() -> None:
    with pytest.raises(ValueError, match="state handoff"):
        stitch_oos(
            (
                oos(
                    0,
                    10,
                    (0.01, 0.02),
                    opening_state_digest="start",
                    closing_state_digest="middle-a",
                ),
                oos(
                    1,
                    12,
                    (0.03, 0.04),
                    opening_state_digest="middle-b",
                    closing_state_digest="end",
                ),
            ),
            mode=StitchMode.CONTINUOUS_ACCOUNT,
        )


def test_continuous_stitch_accepts_verified_state_chain() -> None:
    result = stitch_oos(
        (
            oos(
                0,
                10,
                (0.01, 0.02),
                opening_state_digest="start",
                closing_state_digest="middle",
            ),
            oos(
                1,
                12,
                (0.03, 0.04),
                opening_state_digest="middle",
                closing_state_digest="end",
            ),
        ),
        mode=StitchMode.CONTINUOUS_ACCOUNT,
    )

    assert result.mode is StitchMode.CONTINUOUS_ACCOUNT
    assert result.gaps == ()


def test_stitch_oos_rejects_overlapping_ranges() -> None:
    with pytest.raises(ValueError, match="overlap"):
        stitch_oos((oos(0, 10, (0.01, 0.02)), oos(1, 11, (0.03, 0.04))))


def test_stitch_oos_rejects_incompatible_return_identity() -> None:
    with pytest.raises(ValueError, match="kind"):
        stitch_oos(
            (
                oos(0, 10, (0.01,), kind=ReturnKind.BASE_BAR),
                oos(1, 11, (0.02,), kind=ReturnKind.DECISION_STEP),
            )
        )


@pytest.mark.parametrize("mode", tuple(StitchMode))
@pytest.mark.parametrize(
    ("second_elapsed_years", "expected_years"),
    (
        pytest.param(2 / 365.0, 4 / 365.0, id="all-explicit"),
        pytest.param(None, 2 / 365.0 + 2 / 252.0, id="mixed-default"),
    ),
)
def test_stitch_sums_represented_duration_for_sharpe(
    mode: StitchMode,
    second_elapsed_years: float | None,
    expected_years: float,
) -> None:
    result = stitch_oos(
        paired_oos(
            mode,
            first_elapsed_years=2 / 365.0,
            second_elapsed_years=second_elapsed_years,
        ),
        mode=mode,
    )
    expected_frequency = len(_CLOCK_TEST_RETURNS) / expected_years
    metrics = evaluate_performance(result.returns)

    assert result.returns.values == _CLOCK_TEST_RETURNS
    assert result.returns.elapsed_years == pytest.approx(expected_years)
    assert result.returns.annualization_periods_per_year == pytest.approx(
        expected_frequency
    )
    assert metrics.sharpe == pytest.approx(
        literal_sharpe(_CLOCK_TEST_RETURNS, elapsed_years=expected_years)
    )
    assert metrics.periods_per_year == round(expected_frequency)
    assert result.gaps == (((12, 15),) if mode is StitchMode.INDEPENDENT_FOLDS else ())


@pytest.mark.parametrize("mode", tuple(StitchMode))
def test_stitch_keeps_legacy_clock_when_all_fold_durations_are_defaulted(
    mode: StitchMode,
) -> None:
    result = stitch_oos(
        paired_oos(
            mode,
            first_elapsed_years=None,
            second_elapsed_years=None,
        ),
        mode=mode,
    )
    metrics = evaluate_performance(result.returns)

    assert result.returns.values == _CLOCK_TEST_RETURNS
    assert result.returns.elapsed_years is None
    assert result.returns.annualization_periods_per_year == 252.0
    assert metrics.sharpe == pytest.approx(
        literal_sharpe(_CLOCK_TEST_RETURNS, periods_per_year=252)
    )
    assert metrics.periods_per_year == 252
    assert result.gaps == (((12, 15),) if mode is StitchMode.INDEPENDENT_FOLDS else ())


@pytest.mark.parametrize("mode", tuple(StitchMode))
def test_single_fold_stitch_preserves_explicit_duration(mode: StitchMode) -> None:
    continuous = mode is StitchMode.CONTINUOUS_ACCOUNT
    result = stitch_oos(
        (
            oos(
                0,
                10,
                _CLOCK_TEST_RETURNS[:2],
                periods_per_year=252,
                elapsed_years=2 / 365.0,
                opening_state_digest="start" if continuous else None,
                closing_state_digest="end" if continuous else None,
            ),
        ),
        mode=mode,
    )

    assert result.returns.values == _CLOCK_TEST_RETURNS[:2]
    assert result.returns.elapsed_years == pytest.approx(2 / 365.0)
    assert result.returns.annualization_periods_per_year == pytest.approx(365.0)
    assert result.gaps == ()


def test_fold_result_rejects_range_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        FoldOOSResult(
            fold_index=0,
            start=10,
            stop=13,
            returns=ReturnSeries(
                values=(0.01, 0.02),
                kind=ReturnKind.BASE_BAR,
                periods_per_year=8_760,
            ),
        )
