"""Dataset-bound selection validation shared across strategy families."""

from __future__ import annotations

from collections.abc import Mapping

from trade_rl.data.contracts import FeatureKind
from trade_rl.data.features.cross_asset import CROSS_ASSET_FEATURE_KINDS
from trade_rl.data.identity import parse_identity_json
from trade_rl.data.market import MarketDataset

_REFERENCE_CROSS_ASSET_KINDS = frozenset(
    {
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
    }
)
_UNIVERSE_CROSS_ASSET_KINDS = CROSS_ASSET_FEATURE_KINDS - _REFERENCE_CROSS_ASSET_KINDS


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


def _source_market_build_config(
    dataset: MarketDataset,
) -> Mapping[str, object] | None:
    if dataset.identity_payload_json is None:
        return None
    current: Mapping[str, object] = parse_identity_json(dataset.identity_payload_json)
    for _ in range(8):
        config = current.get("config")
        if isinstance(config, Mapping) and isinstance(config.get("features"), list):
            return config
        source = current.get("source_dataset")
        if not isinstance(source, Mapping):
            return None
        nested = source.get("identity_payload")
        if not isinstance(nested, Mapping):
            return None
        current = nested
    raise ValueError("Dataset source identity nesting is too deep")


def _validate_subset_feature_dependencies(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    symbol_indices: tuple[int, ...],
) -> None:
    if len(symbol_indices) == dataset.n_symbols:
        return
    config = _source_market_build_config(dataset)
    if dataset.identity_payload_json is None:
        # Synthetic/legacy datasets do not claim source-bound feature dependency
        # provenance. Preserve their current behavior; formal generalization claims
        # require a content-verified Dataset with build provenance.
        return
    if config is None:
        raise ValueError(
            "verified Dataset identity lacks feature dependency provenance "
            "required for a strict fit symbol scope"
        )

    raw_features = config.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("Dataset feature dependency provenance is malformed")
    specs: dict[str, Mapping[str, object]] = {}
    for raw_spec in raw_features:
        if not isinstance(raw_spec, Mapping):
            raise ValueError("Dataset feature dependency provenance is malformed")
        name = raw_spec.get("name")
        if not isinstance(name, str) or not name or name in specs:
            raise ValueError("Dataset feature dependency provenance has invalid names")
        specs[name] = raw_spec

    fit_symbols = {dataset.symbols[index] for index in symbol_indices}
    for feature_index in feature_indices:
        feature_name = dataset.feature_names[feature_index]
        spec = specs.get(feature_name)
        if spec is None:
            raise ValueError(
                f"selected feature is absent from Dataset dependency provenance: "
                f"{feature_name}"
            )
        raw_kind = spec.get("kind")
        if not isinstance(raw_kind, str):
            raise ValueError(
                f"selected feature has invalid dependency kind: {feature_name}"
            )
        try:
            kind = FeatureKind(raw_kind)
        except ValueError as error:
            raise ValueError(
                f"selected feature has unsupported dependency kind: {feature_name}"
            ) from error

        if kind in _UNIVERSE_CROSS_ASSET_KINDS:
            raise ValueError(
                f"selected feature depends on symbols outside fit symbol scope: "
                f"{feature_name}"
            )
        if kind in _REFERENCE_CROSS_ASSET_KINDS:
            reference = config.get("cross_asset_reference_symbol")
            if not isinstance(reference, str) or not reference:
                raise ValueError(
                    f"selected reference feature lacks a bound reference symbol: "
                    f"{feature_name}"
                )
            if reference not in fit_symbols:
                raise ValueError(
                    f"reference symbol must belong to fit symbol scope for "
                    f"{feature_name}: {reference}"
                )


def validated_training_scope(
    dataset: MarketDataset,
    *,
    feature_indices: tuple[int, ...],
    fit_symbol_indices: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate training rows and selected-feature information dependencies."""

    indices = validated_feature_indices(dataset, feature_indices)
    symbols = validated_symbol_indices(dataset, fit_symbol_indices)
    _validate_subset_feature_dependencies(
        dataset,
        feature_indices=indices,
        symbol_indices=symbols,
    )
    return indices, symbols


__all__ = [
    "validated_feature_indices",
    "validated_symbol_indices",
    "validated_training_scope",
]
