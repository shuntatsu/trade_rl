from __future__ import annotations

import pytest

from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.stitching import (
    FoldOOSResult,
    StitchMode,
    stitch_oos,
)


def oos(
    fold_index: int,
    start: int,
    values: tuple[float, ...],
    *,
    kind: ReturnKind = ReturnKind.BASE_BAR,
    periods_per_year: int = 8_760,
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
        ),
        opening_state_digest=opening_state_digest,
        closing_state_digest=closing_state_digest,
    )


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
