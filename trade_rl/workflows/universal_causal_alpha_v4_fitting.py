"""Hierarchical train-only ridge fitting for Causal Alpha V4."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
    fit_causal_alpha_ridge,
)
from trade_rl.learning.causal_alpha_v3 import causal_alpha_overlap_uniqueness_weights
from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4FitConfig,
    CausalAlphaV4Forecast,
    CausalAlphaV4SymbolSamples,
    build_causal_alpha_v4_forecast,
    build_causal_alpha_v4_residual_labels,
)
from trade_rl.workflows.universal_causal_alpha_v4_runtime import (
    validate_causal_alpha_v4_train_sample_scope,
)

_V4_HORIZONS = ("4h", "24h", "72h")
_V4_FIT_SCHEMA = "universal_causal_alpha_v4_fit_v1"
_V4_WEIGHT_SCHEMA = "universal_causal_alpha_v4_weight_v1"
_T = TypeVar("_T")


def _horizon_labels(
    sample: CausalAlphaV4SymbolSamples,
    horizon: str,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon not in _V4_HORIZONS:
        raise ValueError("unsupported V4 horizon")
    return (
        np.asarray(getattr(sample, f"labels_{horizon}"), dtype=np.float64),
        np.asarray(getattr(sample, f"label_end_indices_{horizon}"), dtype=np.int64),
    )


def _shared_feature_surface(
    sample: CausalAlphaV4SymbolSamples,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    names = (
        *sample.target_local_feature_names,
        *sample.local_context.feature_names,
        *sample.global_context.feature_names,
        *sample.instrument_descriptor_names,
        "causal_beta",
    )
    if len(set(names)) != len(names):
        raise ValueError("V4 shared fit feature names must be unique")
    features = np.column_stack(
        (
            sample.target_local_features,
            sample.local_context.values,
            sample.global_context.values,
            sample.instrument_descriptors,
            sample.beta[:, None],
        )
    ).astype(np.float64, copy=False)
    available = np.column_stack(
        (
            sample.target_local_available,
            sample.local_context.available,
            sample.global_context.available,
            sample.instrument_descriptor_available,
            sample.beta_available[:, None],
        )
    ).astype(np.bool_, copy=False)
    if features.shape != available.shape or features.shape[1] != len(names):
        raise RuntimeError("V4 shared fit feature surface is misaligned")
    if not np.isfinite(features).all():
        raise ValueError("V4 shared fit features must use finite inert storage")
    return names, features, available


def _normalized_uniqueness_weights(
    *,
    decisions: np.ndarray,
    label_ends: np.ndarray,
    knowledge_cutoff: int,
    additional_eligible: np.ndarray,
    label: str,
) -> np.ndarray:
    weights = causal_alpha_overlap_uniqueness_weights(
        decisions,
        label_ends,
        knowledge_cutoff=knowledge_cutoff,
    )
    eligible = np.asarray(additional_eligible, dtype=np.bool_).reshape(-1)
    if eligible.shape != weights.shape:
        raise ValueError(f"V4 {label} eligibility must align with weights")
    weights = np.where(eligible, weights, 0.0)
    total = float(weights.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"V4 {label} has no eligible weighted support")
    normalized = np.asarray(weights / total, dtype=np.float64)
    normalized.setflags(write=False)
    return normalized


def _weight_digest(
    *,
    horizon: str,
    kind: str,
    knowledge_cutoff: int,
    symbols: tuple[str, ...],
    weights: Mapping[str, np.ndarray],
) -> str:
    return content_and_arrays_digest(
        {
            "horizon": horizon,
            "kind": kind,
            "knowledge_cutoff": knowledge_cutoff,
            "schema_version": _V4_WEIGHT_SCHEMA,
            "symbols": symbols,
        },
        tuple((f"weights:{symbol}", weights[symbol]) for symbol in symbols),
    )


def _weighted_rmse(
    model: CausalAlphaRidgeModel,
    *,
    features: np.ndarray,
    feature_available: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    indices = model.eligible_indices
    selected = np.asarray(weights[indices], dtype=np.float64)
    total = float(selected.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("V4 fitted model has no weighted residual support")
    prediction = model.predict(
        features[indices],
        feature_available=feature_available[indices],
    )
    residual = labels[indices] - prediction
    value = math.sqrt(
        float(np.sum(selected * np.square(residual), dtype=np.float64) / total)
    )
    if not math.isfinite(value):
        raise ValueError("V4 weighted RMSE became non-finite")
    return value


def _mapping_exact(value: Mapping[str, _T], *, field: str) -> dict[str, _T]:
    resolved = dict(value)
    if tuple(resolved) != _V4_HORIZONS:
        raise ValueError(f"{field} must use the canonical V4 horizon order")
    return resolved


@dataclass(frozen=True, slots=True)
class CausalAlphaV4Fit:
    train_symbols: tuple[str, ...]
    knowledge_cutoff: int
    config: CausalAlphaV4FitConfig
    market_models: Mapping[str, CausalAlphaRidgeModel]
    residual_models: Mapping[str, CausalAlphaRidgeModel]
    direction_models: Mapping[str, CausalAlphaRidgeModel]
    market_weight_digests: Mapping[str, str]
    residual_weight_digests: Mapping[str, str]
    direction_weight_digests: Mapping[str, str]
    market_rmse: Mapping[str, float]
    residual_rmse: Mapping[str, float]
    direction_rmse: Mapping[str, float]
    direction_zero_label_counts: Mapping[str, int]
    market_feature_names: tuple[str, ...]
    shared_feature_names: tuple[str, ...]
    sample_scope_digest: str
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("V4 fit train_symbols must be non-empty and unique")
        if "BTCUSDT" not in symbols:
            raise ValueError("V4 fit train_symbols must contain BTCUSDT market proxy")
        if (
            isinstance(self.knowledge_cutoff, bool)
            or not isinstance(self.knowledge_cutoff, int)
            or self.knowledge_cutoff <= 0
        ):
            raise ValueError("V4 fit knowledge_cutoff must be positive")
        if not isinstance(self.config, CausalAlphaV4FitConfig):
            raise TypeError("V4 fit config is invalid")

        market_models = _mapping_exact(self.market_models, field="market_models")
        residual_models = _mapping_exact(self.residual_models, field="residual_models")
        direction_models = _mapping_exact(
            self.direction_models, field="direction_models"
        )
        for model_mapping, field_name in (
            (market_models, "market_models"),
            (residual_models, "residual_models"),
            (direction_models, "direction_models"),
        ):
            if any(
                not isinstance(model, CausalAlphaRidgeModel)
                for model in model_mapping.values()
            ):
                raise TypeError(f"V4 {field_name} must contain ridge models")

        market_names = tuple(self.market_feature_names)
        shared_names = tuple(self.shared_feature_names)
        if not market_names or not shared_names:
            raise ValueError("V4 fit feature names must be non-empty")
        if len(set(market_names)) != len(market_names) or len(set(shared_names)) != len(
            shared_names
        ):
            raise ValueError("V4 fit feature names must be unique")
        for model in market_models.values():
            assert isinstance(model, CausalAlphaRidgeModel)
            if model.feature_names != market_names:
                raise ValueError("V4 market model feature schema drifted")
        for mapping in (residual_models, direction_models):
            for model in mapping.values():
                assert isinstance(model, CausalAlphaRidgeModel)
                if model.feature_names != shared_names:
                    raise ValueError("V4 shared model feature schema drifted")

        digest_maps: dict[str, dict[str, str]] = {}
        for field_name in (
            "market_weight_digests",
            "residual_weight_digests",
            "direction_weight_digests",
        ):
            resolved = _mapping_exact(getattr(self, field_name), field=field_name)
            typed: dict[str, str] = {}
            for horizon, raw in resolved.items():
                digest_value = str(raw)
                if len(digest_value) != 64:
                    raise ValueError(f"V4 {field_name}[{horizon}] is invalid")
                typed[horizon] = digest_value
            digest_maps[field_name] = typed

        numeric_maps: dict[str, dict[str, float]] = {}
        for field_name in ("market_rmse", "residual_rmse", "direction_rmse"):
            resolved = _mapping_exact(getattr(self, field_name), field=field_name)
            typed_float: dict[str, float] = {}
            for horizon, raw in resolved.items():
                numeric_value = float(raw)
                if not math.isfinite(numeric_value) or numeric_value < 0.0:
                    raise ValueError(f"V4 {field_name}[{horizon}] must be non-negative")
                typed_float[horizon] = numeric_value
            numeric_maps[field_name] = typed_float

        zero_raw = _mapping_exact(
            self.direction_zero_label_counts,
            field="direction_zero_label_counts",
        )
        zero_counts: dict[str, int] = {}
        for horizon, raw in zero_raw.items():
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ValueError("V4 direction zero-label counts must be non-negative")
            zero_counts[horizon] = raw
        if (
            not isinstance(self.sample_scope_digest, str)
            or len(self.sample_scope_digest) != 64
        ):
            raise ValueError("V4 fit sample_scope_digest is invalid")

        payload = {
            "config_digest": self.config.digest,
            "direction_model_digests": tuple(
                (horizon, direction_models[horizon].digest) for horizon in _V4_HORIZONS
            ),
            "direction_rmse": tuple(numeric_maps["direction_rmse"].items()),
            "direction_weight_digests": tuple(
                digest_maps["direction_weight_digests"].items()
            ),
            "direction_zero_label_counts": tuple(zero_counts.items()),
            "knowledge_cutoff": self.knowledge_cutoff,
            "market_feature_names": market_names,
            "market_model_digests": tuple(
                (horizon, market_models[horizon].digest) for horizon in _V4_HORIZONS
            ),
            "market_rmse": tuple(numeric_maps["market_rmse"].items()),
            "market_weight_digests": tuple(
                digest_maps["market_weight_digests"].items()
            ),
            "residual_model_digests": tuple(
                (horizon, residual_models[horizon].digest) for horizon in _V4_HORIZONS
            ),
            "residual_rmse": tuple(numeric_maps["residual_rmse"].items()),
            "residual_weight_digests": tuple(
                digest_maps["residual_weight_digests"].items()
            ),
            "sample_scope_digest": self.sample_scope_digest,
            "schema_version": _V4_FIT_SCHEMA,
            "shared_feature_names": shared_names,
            "train_symbols": symbols,
        }
        expected = content_digest(payload)
        if self.digest and self.digest != expected:
            raise ValueError("V4 fit digest mismatch")

        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "market_feature_names", market_names)
        object.__setattr__(self, "shared_feature_names", shared_names)
        object.__setattr__(self, "market_models", MappingProxyType(market_models))
        object.__setattr__(self, "residual_models", MappingProxyType(residual_models))
        object.__setattr__(self, "direction_models", MappingProxyType(direction_models))
        for field_name, digest_mapping in digest_maps.items():
            object.__setattr__(self, field_name, MappingProxyType(digest_mapping))
        for field_name, numeric_mapping in numeric_maps.items():
            object.__setattr__(self, field_name, MappingProxyType(numeric_mapping))
        object.__setattr__(
            self,
            "direction_zero_label_counts",
            MappingProxyType(zero_counts),
        )
        object.__setattr__(self, "digest", expected)

    def predict(self, sample: CausalAlphaV4SymbolSamples) -> CausalAlphaV4Forecast:
        if not isinstance(sample, CausalAlphaV4SymbolSamples):
            raise TypeError("V4 fit prediction requires V4 symbol samples")
        if sample.global_context.feature_names != self.market_feature_names:
            raise ValueError("V4 prediction global feature schema drifted")
        shared_names, shared_features, shared_available = _shared_feature_surface(
            sample
        )
        if shared_names != self.shared_feature_names:
            raise ValueError("V4 prediction shared feature schema drifted")
        market_predictions = {
            horizon: self.market_models[horizon].predict(
                sample.global_context.values,
                feature_available=sample.global_context.available,
            )
            for horizon in _V4_HORIZONS
        }
        residual_predictions = {
            horizon: self.residual_models[horizon].predict(
                shared_features,
                feature_available=shared_available,
            )
            for horizon in _V4_HORIZONS
        }
        direction_scores = {
            horizon: self.direction_models[horizon].predict(
                shared_features,
                feature_available=shared_available,
            )
            for horizon in _V4_HORIZONS
        }
        return build_causal_alpha_v4_forecast(
            symbol=sample.symbol,
            decision_indices=sample.decision_indices,
            beta=sample.beta,
            beta_available=sample.beta_available,
            market_predictions=market_predictions,
            residual_predictions=residual_predictions,
            direction_scores=direction_scores,
            market_model_digests={
                horizon: self.market_models[horizon].digest for horizon in _V4_HORIZONS
            },
            residual_model_digests={
                horizon: self.residual_models[horizon].digest
                for horizon in _V4_HORIZONS
            },
            direction_model_digests={
                horizon: self.direction_models[horizon].digest
                for horizon in _V4_HORIZONS
            },
            fit_digest=self.digest,
        )


def fit_causal_alpha_v4(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaV4SymbolSamples],
    knowledge_cutoff: int,
    config: CausalAlphaV4FitConfig,
) -> CausalAlphaV4Fit:
    """Fit fixed market-proxy, shared residual, and shared direction ridge heads."""

    ordered = validate_causal_alpha_v4_train_sample_scope(
        train_symbols=train_symbols,
        samples=samples,
    )
    symbols = tuple(train_symbols)
    if "BTCUSDT" not in ordered:
        raise ValueError("V4 train sample scope requires BTCUSDT market proxy")
    if not isinstance(config, CausalAlphaV4FitConfig):
        raise TypeError("V4 fit requires CausalAlphaV4FitConfig")
    btc = ordered["BTCUSDT"]
    market_feature_names = btc.global_context.feature_names
    shared_feature_names, _, _ = _shared_feature_surface(ordered[symbols[0]])
    for symbol in symbols[1:]:
        names, _, _ = _shared_feature_surface(ordered[symbol])
        if names != shared_feature_names:
            raise ValueError("V4 shared feature schema drifted across train symbols")

    decomposed = {
        symbol: build_causal_alpha_v4_residual_labels(
            symbol_samples=ordered[symbol],
            btc_market_proxy_samples=btc,
        )
        for symbol in symbols
    }
    shared_features = np.concatenate(
        tuple(_shared_feature_surface(ordered[symbol])[1] for symbol in symbols),
        axis=0,
    )
    shared_available = np.concatenate(
        tuple(_shared_feature_surface(ordered[symbol])[2] for symbol in symbols),
        axis=0,
    )

    market_models: dict[str, CausalAlphaRidgeModel] = {}
    residual_models: dict[str, CausalAlphaRidgeModel] = {}
    direction_models: dict[str, CausalAlphaRidgeModel] = {}
    market_weight_digests: dict[str, str] = {}
    residual_weight_digests: dict[str, str] = {}
    direction_weight_digests: dict[str, str] = {}
    market_rmse: dict[str, float] = {}
    residual_rmse: dict[str, float] = {}
    direction_rmse: dict[str, float] = {}
    zero_counts: dict[str, int] = {}

    for horizon in _V4_HORIZONS:
        btc_labels, btc_ends = _horizon_labels(btc, horizon)
        market_weights = _normalized_uniqueness_weights(
            decisions=btc.decision_indices,
            label_ends=btc_ends,
            knowledge_cutoff=knowledge_cutoff,
            additional_eligible=np.isfinite(btc_labels),
            label=f"market {horizon}",
        )
        market_model = fit_causal_alpha_ridge(
            features=btc.global_context.values,
            labels=btc_labels,
            feature_available=btc.global_context.available,
            label_end_indices=btc_ends,
            knowledge_cutoff=knowledge_cutoff,
            feature_names=market_feature_names,
            config=CausalAlphaRidgeConfig(ridge_strength=config.market_ridge_strength),
            sample_weights=market_weights,
            normalize_objective=True,
            working_memory_rows=4096,
        )
        market_models[horizon] = market_model
        market_weight_digests[horizon] = _weight_digest(
            horizon=horizon,
            kind="market",
            knowledge_cutoff=knowledge_cutoff,
            symbols=("BTCUSDT",),
            weights={"BTCUSDT": market_weights},
        )
        market_rmse[horizon] = _weighted_rmse(
            market_model,
            features=btc.global_context.values,
            feature_available=btc.global_context.available,
            labels=btc_labels,
            weights=market_weights,
        )

        residual_labels_by_symbol: dict[str, np.ndarray] = {}
        residual_weights_by_symbol: dict[str, np.ndarray] = {}
        direction_labels_by_symbol: dict[str, np.ndarray] = {}
        direction_weights_by_symbol: dict[str, np.ndarray] = {}
        zero_count = 0
        for symbol in symbols:
            sample = ordered[symbol]
            original_labels, label_ends = _horizon_labels(sample, horizon)
            residual_labels = np.asarray(
                getattr(decomposed[symbol], f"residual_labels_{horizon}"),
                dtype=np.float64,
            )
            residual_available = np.asarray(
                getattr(decomposed[symbol], f"available_{horizon}"),
                dtype=np.bool_,
            )
            residual_labels_by_symbol[symbol] = residual_labels
            residual_weights_by_symbol[symbol] = _normalized_uniqueness_weights(
                decisions=sample.decision_indices,
                label_ends=label_ends,
                knowledge_cutoff=knowledge_cutoff,
                additional_eligible=residual_available & np.isfinite(residual_labels),
                label=f"residual {symbol} {horizon}",
            )

            realized_before_cutoff = (
                np.isfinite(original_labels)
                & (label_ends >= 0)
                & (label_ends < knowledge_cutoff)
            )
            zero_mask = realized_before_cutoff & (original_labels == 0.0)
            zero_count += int(np.count_nonzero(zero_mask))
            direction_labels = np.full(original_labels.shape, np.nan, dtype=np.float64)
            direction_labels[realized_before_cutoff & (original_labels > 0.0)] = 1.0
            direction_labels[realized_before_cutoff & (original_labels < 0.0)] = -1.0
            direction_labels_by_symbol[symbol] = direction_labels
            direction_weights_by_symbol[symbol] = _normalized_uniqueness_weights(
                decisions=sample.decision_indices,
                label_ends=label_ends,
                knowledge_cutoff=knowledge_cutoff,
                additional_eligible=np.isfinite(direction_labels),
                label=f"direction {symbol} {horizon}",
            )
        zero_counts[horizon] = zero_count

        pooled_residual_labels = np.concatenate(
            tuple(residual_labels_by_symbol[symbol] for symbol in symbols)
        )
        pooled_residual_ends = np.concatenate(
            tuple(_horizon_labels(ordered[symbol], horizon)[1] for symbol in symbols)
        )
        pooled_residual_weights = np.concatenate(
            tuple(residual_weights_by_symbol[symbol] for symbol in symbols)
        )
        residual_model = fit_causal_alpha_ridge(
            features=shared_features,
            labels=pooled_residual_labels,
            feature_available=shared_available,
            label_end_indices=pooled_residual_ends,
            knowledge_cutoff=knowledge_cutoff,
            feature_names=shared_feature_names,
            config=CausalAlphaRidgeConfig(
                ridge_strength=config.residual_ridge_strength
            ),
            sample_weights=pooled_residual_weights,
            normalize_objective=True,
            working_memory_rows=4096,
        )
        residual_models[horizon] = residual_model
        residual_weight_digests[horizon] = _weight_digest(
            horizon=horizon,
            kind="residual",
            knowledge_cutoff=knowledge_cutoff,
            symbols=symbols,
            weights=residual_weights_by_symbol,
        )
        residual_rmse[horizon] = _weighted_rmse(
            residual_model,
            features=shared_features,
            feature_available=shared_available,
            labels=pooled_residual_labels,
            weights=pooled_residual_weights,
        )

        pooled_direction_labels = np.concatenate(
            tuple(direction_labels_by_symbol[symbol] for symbol in symbols)
        )
        pooled_direction_ends = np.concatenate(
            tuple(_horizon_labels(ordered[symbol], horizon)[1] for symbol in symbols)
        )
        pooled_direction_weights = np.concatenate(
            tuple(direction_weights_by_symbol[symbol] for symbol in symbols)
        )
        direction_model = fit_causal_alpha_ridge(
            features=shared_features,
            labels=pooled_direction_labels,
            feature_available=shared_available,
            label_end_indices=pooled_direction_ends,
            knowledge_cutoff=knowledge_cutoff,
            feature_names=shared_feature_names,
            config=CausalAlphaRidgeConfig(
                ridge_strength=config.direction_ridge_strength
            ),
            sample_weights=pooled_direction_weights,
            normalize_objective=True,
            working_memory_rows=4096,
        )
        direction_models[horizon] = direction_model
        direction_weight_digests[horizon] = _weight_digest(
            horizon=horizon,
            kind="direction",
            knowledge_cutoff=knowledge_cutoff,
            symbols=symbols,
            weights=direction_weights_by_symbol,
        )
        direction_rmse[horizon] = _weighted_rmse(
            direction_model,
            features=shared_features,
            feature_available=shared_available,
            labels=pooled_direction_labels,
            weights=pooled_direction_weights,
        )

    sample_scope_digest = content_digest(
        {
            "knowledge_cutoff": knowledge_cutoff,
            "market_feature_names": market_feature_names,
            "schema_version": "universal_causal_alpha_v4_fit_scope_v1",
            "shared_feature_names": shared_feature_names,
            "train_symbols": symbols,
        }
    )
    return CausalAlphaV4Fit(
        train_symbols=symbols,
        knowledge_cutoff=knowledge_cutoff,
        config=config,
        market_models=market_models,
        residual_models=residual_models,
        direction_models=direction_models,
        market_weight_digests=market_weight_digests,
        residual_weight_digests=residual_weight_digests,
        direction_weight_digests=direction_weight_digests,
        market_rmse=market_rmse,
        residual_rmse=residual_rmse,
        direction_rmse=direction_rmse,
        direction_zero_label_counts=zero_counts,
        market_feature_names=market_feature_names,
        shared_feature_names=shared_feature_names,
        sample_scope_digest=sample_scope_digest,
    )


__all__ = [
    "CausalAlphaV4Fit",
    "fit_causal_alpha_v4",
]
