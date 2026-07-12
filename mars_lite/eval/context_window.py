from __future__ import annotations

from dataclasses import dataclass

from mars_lite.features.feature_pipeline import FeatureSet


@dataclass(frozen=True)
class ContextualEvaluationWindow:
    feature_set: FeatureSet
    start_idx: int
    scored_bars: int
    absolute_start: int
    absolute_end: int


def with_history_context(
    fs: FeatureSet,
    *,
    start: int,
    end: int,
    history_bars: int,
) -> ContextualEvaluationWindow:
    """Include causal pre-window history while scoring only [start, end)."""

    if not 0 <= start < end <= fs.n_bars:
        raise ValueError("evaluation bounds must satisfy 0 <= start < end <= n_bars")
    if history_bars < 0:
        raise ValueError("history_bars must be non-negative")
    context_start = max(0, start - history_bars)
    contextual = fs.slice(context_start, end)
    start_idx = start - context_start
    return ContextualEvaluationWindow(
        feature_set=contextual,
        start_idx=start_idx,
        scored_bars=end - start,
        absolute_start=start,
        absolute_end=end,
    )
