from __future__ import annotations

import pytest

from trade_rl.evaluation.walk_forward.folds import (
    IndexRange,
    WalkForwardFold,
    build_folds,
)


def test_build_folds_uses_purged_expanding_windows() -> None:
    folds = build_folds(
        n_bars=400,
        train_bars=100,
        checkpoint_bars=20,
        selection_bars=20,
        test_bars=30,
        purge_bars=5,
        step_bars=30,
        max_folds=3,
    )

    assert len(folds) == 3
    assert tuple(fold.train.start for fold in folds) == (0, 0, 0)
    assert tuple(fold.train.stop for fold in folds) == (100, 130, 160)
    assert tuple(fold.test for fold in folds) == (
        IndexRange(155, 185),
        IndexRange(185, 215),
        IndexRange(215, 245),
    )
    for fold in folds:
        assert fold.checkpoint_validation.start - fold.train.stop == 5
        assert (
            fold.configuration_selection.start - fold.checkpoint_validation.stop == 5
        )
        assert fold.test.start - fold.configuration_selection.stop == 5


def test_rolling_train_windows_shift_with_each_fold() -> None:
    folds = build_folds(
        n_bars=300,
        train_bars=80,
        checkpoint_bars=10,
        selection_bars=10,
        test_bars=20,
        purge_bars=2,
        step_bars=20,
        max_folds=2,
        expanding_train=False,
    )

    assert tuple(fold.train for fold in folds) == (
        IndexRange(0, 80),
        IndexRange(20, 100),
    )


def test_fold_rejects_leaking_window_boundaries() -> None:
    with pytest.raises(ValueError, match="purge"):
        WalkForwardFold(
            fold_index=0,
            train=IndexRange(0, 100),
            checkpoint_validation=IndexRange(103, 120),
            configuration_selection=IndexRange(125, 140),
            test=IndexRange(145, 160),
            purge_bars=5,
        )


def test_builder_rejects_overlapping_outer_test_windows() -> None:
    with pytest.raises(ValueError, match="step_bars"):
        build_folds(
            n_bars=300,
            train_bars=100,
            checkpoint_bars=10,
            selection_bars=10,
            test_bars=30,
            purge_bars=2,
            step_bars=10,
        )


def test_builder_rejects_insufficient_executable_data() -> None:
    with pytest.raises(ValueError, match="executable fold"):
        build_folds(
            n_bars=100,
            train_bars=80,
            checkpoint_bars=10,
            selection_bars=10,
            test_bars=10,
            purge_bars=5,
        )
