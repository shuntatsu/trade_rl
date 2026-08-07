"""Pure nested walk-forward fold boundaries and leakage validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class IndexRange:
    """Half-open integer index range."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("range start must be non-negative")
        if self.stop <= self.start:
            raise ValueError("range stop must be greater than start")

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One leak-separated nested selection and outer-OOS fold."""

    fold_index: int
    train: IndexRange
    checkpoint_validation: IndexRange
    configuration_selection: IndexRange
    test: IndexRange
    purge_bars: int

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative")
        if self.purge_bars < 0:
            raise ValueError("purge_bars must be non-negative")
        self._require_purge(
            self.train,
            self.checkpoint_validation,
            boundary="train/checkpoint",
        )
        self._require_purge(
            self.checkpoint_validation,
            self.configuration_selection,
            boundary="checkpoint/selection",
        )
        self._require_purge(
            self.configuration_selection,
            self.test,
            boundary="selection/test",
        )

    def _require_purge(
        self,
        left: IndexRange,
        right: IndexRange,
        *,
        boundary: str,
    ) -> None:
        actual = right.start - left.stop
        if actual < self.purge_bars:
            raise ValueError(
                f"{boundary} purge is {actual}, expected at least {self.purge_bars}"
            )


def _require_positive(value: int, *, field: str) -> int:
    if value <= 0:
        raise ValueError(f"{field} must be positive")
    return value


def build_folds(
    *,
    n_bars: int,
    train_bars: int,
    checkpoint_bars: int,
    selection_bars: int,
    test_bars: int,
    purge_bars: int,
    step_bars: int | None = None,
    max_folds: int | None = None,
    expanding_train: bool = True,
) -> tuple[WalkForwardFold, ...]:
    """Build deterministic, chronological, non-overlapping outer-OOS folds."""

    _require_positive(n_bars, field="n_bars")
    _require_positive(train_bars, field="train_bars")
    _require_positive(checkpoint_bars, field="checkpoint_bars")
    _require_positive(selection_bars, field="selection_bars")
    _require_positive(test_bars, field="test_bars")
    if purge_bars < 0:
        raise ValueError("purge_bars must be non-negative")
    if max_folds is not None:
        _require_positive(max_folds, field="max_folds")

    resolved_step = test_bars if step_bars is None else step_bars
    _require_positive(resolved_step, field="step_bars")
    if resolved_step < test_bars:
        raise ValueError("step_bars must be at least test_bars to avoid OOS overlap")

    folds: list[WalkForwardFold] = []
    fold_index = 0
    while max_folds is None or fold_index < max_folds:
        shift = fold_index * resolved_step
        train_start = 0 if expanding_train else shift
        train_stop = shift + train_bars
        checkpoint_start = train_stop + purge_bars
        checkpoint_stop = checkpoint_start + checkpoint_bars
        selection_start = checkpoint_stop + purge_bars
        selection_stop = selection_start + selection_bars
        test_start = selection_stop + purge_bars
        test_stop = test_start + test_bars
        if test_stop > n_bars:
            break

        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train=IndexRange(train_start, train_stop),
                checkpoint_validation=IndexRange(
                    checkpoint_start,
                    checkpoint_stop,
                ),
                configuration_selection=IndexRange(
                    selection_start,
                    selection_stop,
                ),
                test=IndexRange(test_start, test_stop),
                purge_bars=purge_bars,
            )
        )
        fold_index += 1

    if not folds:
        raise ValueError("no executable fold could be constructed from the data")
    return tuple(folds)
