"""Dataset-bound selection validation shared across strategy families."""

from __future__ import annotations

from trade_rl.data.market import MarketDataset


def validated_feature_indices(
    dataset: MarketDataset,
    feature_indices: tuple[int, ...],
) -> tuple[int, ...]:
    """Validate an ordered feature selection against one dataset."""

    indices = tuple(feature_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("feature_indices must be non-empty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("feature_indices must contain non-negative integers")
    if max(indices) >= dataset.n_features:
        raise ValueError("feature index is outside dataset features")
    return indices


def validated_symbol_indices(
    dataset: MarketDataset,
    symbol_indices: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve an optional symbol scope while rejecting ambiguous indices."""

    if symbol_indices is None:
        return tuple(range(dataset.n_symbols))
    indices = tuple(symbol_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("symbol_indices must be non-empty and unique")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("symbol_indices must contain non-negative integers")
    if max(indices) >= dataset.n_symbols:
        raise ValueError("symbol index is outside dataset symbols")
    return indices


__all__ = ["validated_feature_indices", "validated_symbol_indices"]
