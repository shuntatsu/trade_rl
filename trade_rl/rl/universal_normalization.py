from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)

_NORMALIZER_VERSION = "symbol_balanced_standard_normalizer_v1"
_UNIVERSAL_SEQUENCE_NORMALIZER_VERSION = "universal_trade_sequence_normalizer_v1"
_UNIVERSAL_STATISTICS_SEMANTICS = "equal_symbol_source_event_moments_v1"
_EPSILON = 1e-12
_NS_PER_HOUR = 3_600_000_000_000


def _readonly_float_vector(value: np.ndarray, *, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite rank-1 vector")
    result = array.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class UniversalTradePublishedSource:
    symbol: str
    artifact_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("Universal Trade published source symbol is invalid")
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))


@dataclass(frozen=True, slots=True)
class UniversalTradeChannelStatistics:
    timeframe: str
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    per_symbol_sample_counts: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, str) or not self.timeframe:
            raise ValueError("Universal Trade normalization timeframe is invalid")
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("Universal Trade normalization feature names are invalid")
        mean = _readonly_float_vector(self.mean, field="normalization mean")
        scale = _readonly_float_vector(self.scale, field="normalization scale")
        if mean.shape != scale.shape or mean.size != len(self.feature_names):
            raise ValueError("Universal Trade normalization statistics width mismatch")
        if np.any(scale <= 0.0):
            raise ValueError("Universal Trade normalization scale must be positive")
        symbols = tuple(symbol for symbol, _counts in self.per_symbol_sample_counts)
        if symbols != tuple(sorted(symbols)) or len(set(symbols)) != len(symbols):
            raise ValueError(
                "normalization sample-count symbols must be sorted and unique"
            )
        for symbol, counts in self.per_symbol_sample_counts:
            if not symbol or len(counts) != len(self.feature_names):
                raise ValueError(
                    "normalization sample counts do not match feature width"
                )
            if any(isinstance(count, bool) or count <= 0 for count in counts):
                raise ValueError("normalization sample counts must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def digest_payload(self) -> dict[str, object]:
        return {
            "timeframe": self.timeframe,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "per_symbol_sample_counts": self.per_symbol_sample_counts,
        }


def _universal_statistics_payload(
    *,
    channels: tuple[UniversalTradeChannelStatistics, ...],
    contract_digest: str,
    source_dataset_digests: tuple[tuple[str, str], ...],
    knowledge_cutoff_ns: int,
    clip_value: float,
) -> dict[str, object]:
    return {
        "version": _UNIVERSAL_SEQUENCE_NORMALIZER_VERSION,
        "statistics_semantics": _UNIVERSAL_STATISTICS_SEMANTICS,
        "contract_digest": contract_digest,
        "source_dataset_digests": source_dataset_digests,
        "knowledge_cutoff_ns": knowledge_cutoff_ns,
        "clip_value": clip_value,
        "channels": tuple(channel.digest_payload() for channel in channels),
    }


def _universal_artifact_payload(
    *,
    statistics_digest: str,
    universe_manifest_digest: str,
    provenance_digest: str,
    contract_digest: str,
) -> dict[str, object]:
    return {
        "version": _UNIVERSAL_SEQUENCE_NORMALIZER_VERSION,
        "statistics_digest": statistics_digest,
        "universe_manifest_digest": universe_manifest_digest,
        "provenance_digest": provenance_digest,
        "contract_digest": contract_digest,
    }


@dataclass(frozen=True, slots=True)
class UniversalTradeSequenceNormalizer:
    channels: tuple[UniversalTradeChannelStatistics, ...]
    contract_digest: str
    train_symbols: tuple[str, ...]
    source_dataset_digests: tuple[tuple[str, str], ...]
    knowledge_cutoff_ns: int
    universe_manifest_digest: str
    provenance_digest: str
    statistics_digest: str
    digest: str
    version: str = _UNIVERSAL_SEQUENCE_NORMALIZER_VERSION
    clip_value: float = 10.0

    def __post_init__(self) -> None:
        if self.version != _UNIVERSAL_SEQUENCE_NORMALIZER_VERSION:
            raise ValueError("unsupported Universal Trade sequence normalizer version")
        timeframes = tuple(channel.timeframe for channel in self.channels)
        if not self.channels or len(set(timeframes)) != len(timeframes):
            raise ValueError("Universal Trade normalization channels are invalid")
        if (
            self.train_symbols != tuple(sorted(self.train_symbols))
            or not self.train_symbols
            or len(set(self.train_symbols)) != len(self.train_symbols)
        ):
            raise ValueError("Universal Trade normalization Train symbols are invalid")
        if (
            tuple(symbol for symbol, _digest in self.source_dataset_digests)
            != self.train_symbols
        ):
            raise ValueError("Universal Trade normalization source identities mismatch")
        require_sha256(
            self.contract_digest,
            field="Universal Trade normalization contract digest",
        )
        for symbol, dataset_digest in self.source_dataset_digests:
            require_sha256(
                dataset_digest,
                field=f"Universal Trade normalization source {symbol} digest",
            )
        require_sha256(
            self.universe_manifest_digest,
            field="Universal Trade normalization universe manifest digest",
        )
        require_sha256(
            self.provenance_digest,
            field="Universal Trade normalization provenance digest",
        )
        require_sha256(
            self.statistics_digest,
            field="Universal Trade normalization statistics digest",
        )
        require_sha256(
            self.digest,
            field="Universal Trade normalization artifact digest",
        )
        if (
            isinstance(self.knowledge_cutoff_ns, bool)
            or not isinstance(self.knowledge_cutoff_ns, int)
            or self.knowledge_cutoff_ns <= 0
        ):
            raise ValueError("Universal Trade normalization cutoff must be positive")
        if not np.isfinite(self.clip_value) or self.clip_value <= 0.0:
            raise ValueError("Universal Trade normalization clip must be positive")

        expected_statistics_digest = content_digest(
            _universal_statistics_payload(
                channels=self.channels,
                contract_digest=self.contract_digest,
                source_dataset_digests=self.source_dataset_digests,
                knowledge_cutoff_ns=self.knowledge_cutoff_ns,
                clip_value=self.clip_value,
            )
        )
        if self.statistics_digest != expected_statistics_digest:
            raise ValueError("Universal Trade normalization statistics digest mismatch")
        expected_digest = content_digest(
            _universal_artifact_payload(
                statistics_digest=self.statistics_digest,
                universe_manifest_digest=self.universe_manifest_digest,
                provenance_digest=self.provenance_digest,
                contract_digest=self.contract_digest,
            )
        )
        if self.digest != expected_digest:
            raise ValueError("Universal Trade normalization artifact digest mismatch")

    def statistics_for(self, timeframe: str) -> UniversalTradeChannelStatistics:
        for channel in self.channels:
            if channel.timeframe == timeframe:
                return channel
        raise KeyError(timeframe)

    def transform(
        self,
        timeframe: str,
        values: np.ndarray,
        available: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        statistics = self.statistics_for(timeframe)
        if feature_names != statistics.feature_names:
            raise ValueError("Universal Trade normalization feature order mismatch")
        array = np.asarray(values, dtype=np.float64)
        availability = np.asarray(available, dtype=np.bool_)
        if array.shape != availability.shape or array.ndim < 1:
            raise ValueError("Universal Trade normalization values/mask shape mismatch")
        if array.shape[-1] != len(statistics.feature_names):
            raise ValueError("Universal Trade normalization feature width mismatch")
        safe_values = np.where(availability, array, statistics.mean)
        if not np.all(np.isfinite(safe_values)):
            raise ValueError(
                "available Universal Trade normalization values must be finite"
            )
        normalized = (safe_values - statistics.mean) / statistics.scale
        normalized = np.clip(normalized, -self.clip_value, self.clip_value)
        normalized = np.where(availability, normalized, 0.0)
        return np.asarray(normalized, dtype=np.float32)


def _source_event_samples(
    dataset: MarketDataset,
    *,
    feature_index: int,
    knowledge_cutoff_ns: int,
) -> np.ndarray:
    timestamps_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    values = np.asarray(dataset.features[:, 0, feature_index], dtype=np.float64)
    available = np.asarray(
        dataset.resolved_array("feature_available")[:, 0, feature_index],
        dtype=np.bool_,
    )
    staleness = np.asarray(
        dataset.resolved_array("feature_staleness_hours")[:, 0, feature_index],
        dtype=np.float64,
    )
    finite_staleness = np.isfinite(staleness)
    safe_staleness = np.where(finite_staleness, staleness, 0.0)
    event_ns = timestamps_ns - np.rint(safe_staleness * _NS_PER_HOUR).astype(np.int64)
    valid = available & finite_staleness & np.isfinite(values)
    valid &= timestamps_ns <= knowledge_cutoff_ns
    valid &= event_ns <= knowledge_cutoff_ns
    valid_values = values[valid]
    valid_events = event_ns[valid]
    if valid_values.size == 0:
        raise ValueError(
            f"no causal normalization samples for feature index {feature_index}"
        )
    _unique_events, unique_indices = np.unique(valid_events, return_index=True)
    samples = np.asarray(valid_values[unique_indices], dtype=np.float64)
    if samples.size == 0:
        raise ValueError(
            f"no unique causal normalization samples for feature index {feature_index}"
        )
    return samples


def build_universal_trade_sequence_normalizer(
    *,
    symbol_datasets: Mapping[str, MarketDataset],
    contract: UniversalTradePolicyContract,
    source_dataset_digests: tuple[tuple[str, str], ...],
    knowledge_cutoff_ns: int,
    universe_manifest_digest: str,
    provenance_digest: str,
    clip_value: float = 10.0,
) -> UniversalTradeSequenceNormalizer:
    """Build equal-symbol source-event statistics from already verified Train data."""

    train_symbols = tuple(symbol for symbol, _digest in source_dataset_digests)
    if not train_symbols or train_symbols != tuple(sorted(train_symbols)):
        raise ValueError(
            "Universal Trade normalization source identities must be sorted"
        )
    if len(set(train_symbols)) != len(train_symbols):
        raise ValueError(
            "Universal Trade normalization source identities must be unique"
        )
    if set(symbol_datasets) != set(train_symbols):
        raise ValueError(
            "Universal Trade normalization datasets must exactly match Train"
        )
    if isinstance(knowledge_cutoff_ns, bool) or knowledge_cutoff_ns <= 0:
        raise ValueError("Universal Trade normalization cutoff must be positive")
    expected_feature_names = tuple(spec.name for spec in contract.feature_specs)
    for symbol in train_symbols:
        dataset = symbol_datasets[symbol]
        if dataset.n_symbols != 1 or dataset.symbols != (symbol,):
            raise ValueError(
                "Universal Trade normalization requires one symbol per dataset"
            )
        if dataset.feature_names != expected_feature_names:
            raise ValueError("Universal Trade normalization feature order mismatch")

    channels: list[UniversalTradeChannelStatistics] = []
    for timeframe, _window in UNIVERSAL_TRADE_SEQUENCE_WINDOWS:
        feature_indices = tuple(
            index
            for index, spec in enumerate(contract.feature_specs)
            if spec.resolved_timeframe("15m") == timeframe
        )
        if not feature_indices:
            continue
        feature_names = tuple(
            expected_feature_names[index] for index in feature_indices
        )
        per_symbol_means: list[np.ndarray] = []
        per_symbol_seconds: list[np.ndarray] = []
        sample_counts: list[tuple[str, tuple[int, ...]]] = []
        for symbol in train_symbols:
            dataset = symbol_datasets[symbol]
            means: list[float] = []
            seconds: list[float] = []
            counts: list[int] = []
            for feature_index in feature_indices:
                samples = _source_event_samples(
                    dataset,
                    feature_index=feature_index,
                    knowledge_cutoff_ns=knowledge_cutoff_ns,
                )
                means.append(float(samples.mean()))
                seconds.append(float(np.mean(samples * samples)))
                counts.append(int(samples.size))
            per_symbol_means.append(np.asarray(means, dtype=np.float64))
            per_symbol_seconds.append(np.asarray(seconds, dtype=np.float64))
            sample_counts.append((symbol, tuple(counts)))
        mean = np.mean(np.stack(per_symbol_means), axis=0)
        second = np.mean(np.stack(per_symbol_seconds), axis=0)
        variance = np.maximum(second - mean * mean, 0.0)
        scale = np.sqrt(variance)
        scale = np.where(scale <= _EPSILON, 1.0, scale)
        channels.append(
            UniversalTradeChannelStatistics(
                timeframe=timeframe,
                feature_names=feature_names,
                mean=mean,
                scale=scale,
                per_symbol_sample_counts=tuple(sample_counts),
            )
        )

    resolved_channels = tuple(channels)
    statistics_digest = content_digest(
        _universal_statistics_payload(
            channels=resolved_channels,
            contract_digest=contract.digest,
            source_dataset_digests=source_dataset_digests,
            knowledge_cutoff_ns=knowledge_cutoff_ns,
            clip_value=clip_value,
        )
    )
    digest = content_digest(
        _universal_artifact_payload(
            statistics_digest=statistics_digest,
            universe_manifest_digest=universe_manifest_digest,
            provenance_digest=provenance_digest,
            contract_digest=contract.digest,
        )
    )
    return UniversalTradeSequenceNormalizer(
        channels=resolved_channels,
        contract_digest=contract.digest,
        train_symbols=train_symbols,
        source_dataset_digests=source_dataset_digests,
        knowledge_cutoff_ns=knowledge_cutoff_ns,
        universe_manifest_digest=universe_manifest_digest,
        provenance_digest=provenance_digest,
        statistics_digest=statistics_digest,
        digest=digest,
        clip_value=clip_value,
    )


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
