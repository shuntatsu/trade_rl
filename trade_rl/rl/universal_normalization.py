from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from trade_rl.artifacts.hashing import content_digest

_NORMALIZER_VERSION = "symbol_balanced_standard_normalizer_v1"
_EPSILON = 1e-12


class _UniversalTradeNormalizationAccess(Protocol):
    @property
    def phase(self) -> object: ...

    def require_normalization_scope(self, symbols: tuple[str, ...]) -> None: ...


@dataclass(frozen=True, slots=True)
class UniversalTradePublishedSource:
    symbol: str
    artifact_root: Path


def fit_universal_trade_sequence_normalizer(
    *,
    manifest: object,
    access: _UniversalTradeNormalizationAccess,
    sources: Sequence[UniversalTradePublishedSource],
    contract: object,
    knowledge_cutoff_ns: int,
) -> None:
    del manifest, contract, knowledge_cutoff_ns
    phase_value = getattr(access.phase, "value", access.phase)
    if phase_value != "train":
        raise PermissionError("Universal Trade RL normalization fitting is Train-only")
    access.require_normalization_scope(tuple(source.symbol for source in sources))
    raise NotImplementedError("U1 sequence normalizer fitting is not implemented yet")


def _evenly_spaced_indices(count: int, sample_count: int) -> np.ndarray:
    if sample_count <= 0 or count <= 0:
        raise ValueError("sample counts must be positive")
    if sample_count > count:
        raise ValueError("sample_count cannot exceed count")
    if sample_count == count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, sample_count, dtype=np.int64)


@dataclass(frozen=True)
class SymbolBalancedStandardNormalizer:
    mean: np.ndarray
    std: np.ndarray
    constant_mask: np.ndarray
    train_symbols: tuple[str, ...]
    feature_schema_digest: str
    catalog_digest: str
    split_manifest_digest: str
    fold_train_range: tuple[int, int]
    sample_count_per_feature: tuple[int, ...]
    statistics_digest: str
    version: str = _NORMALIZER_VERSION
    clip_value: float = 10.0

    @property
    def sample_count_per_symbol(self) -> int:
        return min(self.sample_count_per_feature)

    @classmethod
    def fit(
        cls,
        symbol_features: Mapping[str, np.ndarray],
        *,
        train_symbols: Sequence[str],
        feature_schema_digest: str,
        catalog_digest: str,
        split_manifest_digest: str,
        fold_train_range: tuple[int, int],
        max_samples_per_symbol: int = 100_000,
        symbol_available: Mapping[str, np.ndarray] | None = None,
    ) -> SymbolBalancedStandardNormalizer:
        ordered_symbols = tuple(train_symbols)
        if not ordered_symbols or len(set(ordered_symbols)) != len(ordered_symbols):
            raise ValueError("train_symbols must be non-empty and unique")
        if set(symbol_features) != set(ordered_symbols):
            raise ValueError("symbol_features must exactly match train_symbols")
        if symbol_available is not None and set(symbol_available) != set(
            ordered_symbols
        ):
            raise ValueError("symbol_available must exactly match train_symbols")
        if max_samples_per_symbol <= 0:
            raise ValueError("max_samples_per_symbol must be positive")
        start, stop = fold_train_range
        if start < 0 or stop <= start:
            raise ValueError(
                "fold_train_range must be an increasing non-negative range"
            )

        scoped: dict[str, np.ndarray] = {}
        scoped_available: dict[str, np.ndarray] = {}
        feature_count: int | None = None
        for symbol in ordered_symbols:
            values = np.asarray(symbol_features[symbol], dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("every symbol feature array must be rank-2")
            if stop > values.shape[0]:
                raise ValueError(
                    f"fold_train_range exceeds available rows for symbol {symbol}"
                )
            if feature_count is None:
                feature_count = values.shape[1]
            elif values.shape[1] != feature_count:
                raise ValueError("all symbols must share the same feature width")
            scoped[symbol] = values[start:stop]
            if symbol_available is not None:
                availability = np.asarray(symbol_available[symbol], dtype=np.bool_)
                if availability.shape != values.shape:
                    raise ValueError(
                        f"symbol_available shape mismatch for symbol {symbol}"
                    )
                scoped_available[symbol] = availability[start:stop]

        assert feature_count is not None
        means = np.empty(feature_count, dtype=np.float64)
        stds = np.empty(feature_count, dtype=np.float64)
        sample_counts: list[int] = []
        for feature_index in range(feature_count):
            valid_by_symbol: dict[str, np.ndarray] = {}
            for symbol in ordered_symbols:
                column = scoped[symbol][:, feature_index]
                valid = np.isfinite(column)
                if symbol_available is not None:
                    valid &= scoped_available[symbol][:, feature_index]
                valid_values = column[valid]
                if valid_values.size == 0:
                    raise ValueError(
                        f"symbol {symbol} has no valid observations for feature {feature_index}"
                    )
                valid_by_symbol[symbol] = valid_values
            sample_count = min(
                max_samples_per_symbol,
                min(values.size for values in valid_by_symbol.values()),
            )
            samples = []
            for symbol in ordered_symbols:
                values = valid_by_symbol[symbol]
                indices = _evenly_spaced_indices(values.size, sample_count)
                samples.append(values[indices])
            combined = np.concatenate(samples)
            means[feature_index] = combined.mean()
            stds[feature_index] = combined.std()
            sample_counts.append(sample_count)

        constant_mask = stds <= _EPSILON
        safe_std = stds.copy()
        safe_std[constant_mask] = 1.0
        sample_count_per_feature = tuple(sample_counts)

        digest_payload: dict[str, object] = {
            "version": _NORMALIZER_VERSION,
            "catalog_digest": catalog_digest,
            "split_manifest_digest": split_manifest_digest,
            "train_symbols": ordered_symbols,
            "fold_train_range": fold_train_range,
            "feature_schema_digest": feature_schema_digest,
            "sample_count_per_feature": sample_count_per_feature,
            "mean": means.tolist(),
            "std": safe_std.tolist(),
            "constant_mask": constant_mask.tolist(),
        }
        if symbol_available is not None:
            digest_payload["availability_aware"] = True
        statistics_digest = content_digest(digest_payload)
        return cls(
            mean=means,
            std=safe_std,
            constant_mask=constant_mask,
            train_symbols=ordered_symbols,
            feature_schema_digest=feature_schema_digest,
            catalog_digest=catalog_digest,
            split_manifest_digest=split_manifest_digest,
            fold_train_range=fold_train_range,
            sample_count_per_feature=sample_count_per_feature,
            statistics_digest=statistics_digest,
        )

    def transform(
        self,
        values: np.ndarray,
        *,
        available: np.ndarray | None = None,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError("feature width does not match fitted normalizer")
        availability: np.ndarray | None = None
        if available is not None:
            availability = np.asarray(available, dtype=np.bool_)
            if availability.shape != array.shape:
                raise ValueError("available shape must match values")
        normalized = (array - self.mean) / self.std
        normalized[..., self.constant_mask] = 0.0
        normalized = np.clip(normalized, -self.clip_value, self.clip_value)
        if availability is not None:
            normalized = np.where(availability, normalized, 0.0)
        return normalized
