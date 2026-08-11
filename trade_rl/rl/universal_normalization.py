from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


_NORMALIZER_VERSION = "symbol_balanced_standard_normalizer_v1"
_EPSILON = 1e-12


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
    sample_count_per_symbol: int
    statistics_digest: str
    version: str = _NORMALIZER_VERSION
    clip_value: float = 10.0

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
    ) -> "SymbolBalancedStandardNormalizer":
        ordered_symbols = tuple(train_symbols)
        if not ordered_symbols or len(set(ordered_symbols)) != len(ordered_symbols):
            raise ValueError("train_symbols must be non-empty and unique")
        if set(symbol_features) != set(ordered_symbols):
            raise ValueError("symbol_features must exactly match train_symbols")
        if max_samples_per_symbol <= 0:
            raise ValueError("max_samples_per_symbol must be positive")
        start, stop = fold_train_range
        if start < 0 or stop <= start:
            raise ValueError("fold_train_range must be an increasing non-negative range")

        valid_rows: dict[str, np.ndarray] = {}
        feature_count: int | None = None
        for symbol in ordered_symbols:
            values = np.asarray(symbol_features[symbol], dtype=np.float64)
            if values.ndim != 2:
                raise ValueError("every symbol feature array must be rank-2")
            if feature_count is None:
                feature_count = values.shape[1]
            elif values.shape[1] != feature_count:
                raise ValueError("all symbols must share the same feature width")
            finite = np.isfinite(values).all(axis=1)
            rows = values[finite]
            if rows.shape[0] == 0:
                raise ValueError(f"symbol {symbol} has no fully valid feature rows")
            valid_rows[symbol] = rows

        sample_count = min(
            max_samples_per_symbol,
            min(rows.shape[0] for rows in valid_rows.values()),
        )
        sampled = []
        for symbol in ordered_symbols:
            rows = valid_rows[symbol]
            sampled.append(rows[_evenly_spaced_indices(rows.shape[0], sample_count)])
        matrix = np.concatenate(sampled, axis=0)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        constant_mask = std <= _EPSILON
        safe_std = std.copy()
        safe_std[constant_mask] = 1.0

        statistics_digest = _canonical_digest(
            {
                "version": _NORMALIZER_VERSION,
                "catalog_digest": catalog_digest,
                "split_manifest_digest": split_manifest_digest,
                "train_symbols": ordered_symbols,
                "fold_train_range": fold_train_range,
                "feature_schema_digest": feature_schema_digest,
                "sample_count_per_symbol": sample_count,
                "mean": mean.tolist(),
                "std": safe_std.tolist(),
                "constant_mask": constant_mask.tolist(),
            }
        )
        return cls(
            mean=mean,
            std=safe_std,
            constant_mask=constant_mask,
            train_symbols=ordered_symbols,
            feature_schema_digest=feature_schema_digest,
            catalog_digest=catalog_digest,
            split_manifest_digest=split_manifest_digest,
            fold_train_range=fold_train_range,
            sample_count_per_symbol=sample_count,
            statistics_digest=statistics_digest,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.mean.shape[0]:
            raise ValueError("feature width does not match fitted normalizer")
        normalized = (array - self.mean) / self.std
        normalized[..., self.constant_mask] = 0.0
        return np.clip(normalized, -self.clip_value, self.clip_value)
