from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from trade_rl.evaluation.walk_forward.folds import build_folds


@settings(max_examples=100, deadline=None)
@given(
    train_bars=st.integers(min_value=10, max_value=120),
    checkpoint_bars=st.integers(min_value=2, max_value=30),
    selection_bars=st.integers(min_value=2, max_value=30),
    test_bars=st.integers(min_value=2, max_value=40),
    purge_bars=st.integers(min_value=0, max_value=10),
    fold_count=st.integers(min_value=1, max_value=6),
    expanding=st.booleans(),
)
def test_generated_folds_are_purged_and_outer_oos_never_overlaps(
    train_bars: int,
    checkpoint_bars: int,
    selection_bars: int,
    test_bars: int,
    purge_bars: int,
    fold_count: int,
    expanding: bool,
) -> None:
    step_bars = test_bars
    required = (
        train_bars
        + checkpoint_bars
        + selection_bars
        + test_bars
        + 3 * purge_bars
        + (fold_count - 1) * step_bars
    )
    folds = build_folds(
        n_bars=required,
        train_bars=train_bars,
        checkpoint_bars=checkpoint_bars,
        selection_bars=selection_bars,
        test_bars=test_bars,
        purge_bars=purge_bars,
        step_bars=step_bars,
        max_folds=fold_count,
        expanding_train=expanding,
    )

    assert len(folds) == fold_count
    for index, fold in enumerate(folds):
        assert fold.checkpoint_validation.start - fold.train.stop >= purge_bars
        assert (
            fold.configuration_selection.start - fold.checkpoint_validation.stop
            >= purge_bars
        )
        assert fold.test.start - fold.configuration_selection.stop >= purge_bars
        if index:
            assert folds[index - 1].test.stop <= fold.test.start
        if expanding:
            assert fold.train.start == 0
        else:
            assert fold.train.size == train_bars
