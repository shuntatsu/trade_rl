"""Research-only Universal assembly for overlap-aware causal alpha V3 fits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaRidgeConfig,
    CausalAlphaRidgeModel,
    fit_causal_alpha_ridge,
)
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3Forecast,
    causal_alpha_overlap_uniqueness_weights,
    causal_alpha_v3_forecast,
)
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaSymbolSamples,
)

_V3_FIT_EVIDENCE_SCHEMA = "universal_causal_alpha_v3_fit_v1"


def _validated_scope(
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
) -> tuple[tuple[str, ...], tuple[CausalAlphaSymbolSamples, ...], str]:
    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols) or any(not item for item in symbols):
        raise ValueError("V3 train_symbols must be non-empty and unique")
    if set(samples) != set(symbols):
        raise ValueError("V3 samples must exactly match train_symbols")
    blocks = tuple(samples[symbol] for symbol in symbols)
    for symbol, block in zip(symbols, blocks, strict=True):
        if not isinstance(block, CausalAlphaSymbolSamples) or block.symbol != symbol:
            raise ValueError("V3 sample symbol identity drifted")
    if len({block.feature_names for block in blocks}) != 1:
        raise ValueError("V3 feature names drifted across train symbols")
    if len({block.feature_schema_digest for block in blocks}) != 1:
        raise ValueError("V3 feature schema drifted across train symbols")
    scope_digest = content_digest(
        {
            "sample_digests": tuple(block.digest for block in blocks),
            "schema_version": "universal_causal_alpha_v3_scope_v1",
            "train_symbols": symbols,
        }
    )
    return symbols, blocks, scope_digest


def build_causal_alpha_v3_symbol_balanced_weights(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    knowledge_cutoff: int,
    horizon: Literal["24h", "72h"],
) -> dict[str, np.ndarray]:
    """Build overlap-corrected weights with equal eligible mass per symbol."""

    symbols, blocks, _ = _validated_scope(train_symbols, samples)
    if horizon not in {"24h", "72h"}:
        raise ValueError("V3 horizon must be 24h or 72h")
    if isinstance(knowledge_cutoff, bool) or not isinstance(knowledge_cutoff, int):
        raise ValueError("knowledge_cutoff must be an integer")
    result: dict[str, np.ndarray] = {}
    for symbol, block in zip(symbols, blocks, strict=True):
        ends = (
            block.label_end_indices_24h
            if horizon == "24h"
            else block.label_end_indices_72h
        )
        weights = causal_alpha_overlap_uniqueness_weights(
            block.decision_indices,
            ends,
            knowledge_cutoff=knowledge_cutoff,
        )
        total = float(weights.sum(dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f"V3 {horizon} weights contain no eligible row for {symbol}")
        normalized = weights / total
        normalized.setflags(write=False)
        result[symbol] = normalized
    return result


def _pooled(
    blocks: tuple[CausalAlphaSymbolSamples, ...],
    field: str,
) -> np.ndarray:
    return np.concatenate(tuple(np.asarray(getattr(block, field)) for block in blocks), axis=0)


def _weight_digest(
    symbols: tuple[str, ...],
    weights: Mapping[str, np.ndarray],
    *,
    horizon: str,
    knowledge_cutoff: int,
) -> str:
    return content_and_arrays_digest(
        {
            "horizon": horizon,
            "knowledge_cutoff": knowledge_cutoff,
            "schema_version": "universal_causal_alpha_v3_weights_v1",
            "symbols": symbols,
        },
        tuple((f"weights:{symbol}", weights[symbol]) for symbol in symbols),
    )


def _weighted_residual_rmse(
    model: CausalAlphaRidgeModel,
    *,
    features: np.ndarray,
    feature_available: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    indices = model.eligible_indices
    selected_weights = weights[indices]
    total = float(selected_weights.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("V3 fitted model has no positive residual weight")
    prediction = model.predict(
        features[indices],
        feature_available=feature_available[indices],
    )
    residual = labels[indices] - prediction
    value = math.sqrt(
        float(
            np.sum(selected_weights * np.square(residual), dtype=np.float64) / total
        )
    )
    if not math.isfinite(value):
        raise ValueError("V3 residual RMSE is non-finite")
    return value


@dataclass(frozen=True, slots=True)
class CausalAlphaV3Fit:
    train_symbols: tuple[str, ...]
    knowledge_cutoff: int
    model_24h: CausalAlphaRidgeModel
    model_72h: CausalAlphaRidgeModel
    residual_rmse_24h: float
    residual_rmse_72h: float
    weight_digest_24h: str
    weight_digest_72h: str
    sample_scope_digest: str
    config: CausalAlphaV3FitConfig
    digest: str = ""

    def __post_init__(self) -> None:
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("V3 fit train_symbols must be unique")
        if self.knowledge_cutoff <= 0:
            raise ValueError("V3 fit knowledge_cutoff must be positive")
        for field in ("residual_rmse_24h", "residual_rmse_72h"):
            value = getattr(self, field)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"V3 fit {field} must be finite and non-negative")
        for field in (
            "weight_digest_24h",
            "weight_digest_72h",
            "sample_scope_digest",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"V3 fit {field} is invalid")
        if not isinstance(self.config, CausalAlphaV3FitConfig):
            raise TypeError("V3 fit config is invalid")
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 fit digest mismatch")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "digest", expected)

    def predict(
        self,
        features: object,
        *,
        feature_available: object | None = None,
    ) -> CausalAlphaV3Forecast:
        prediction_24h = self.model_24h.predict(
            features,
            feature_available=feature_available,
        )
        prediction_72h = self.model_72h.predict(
            features,
            feature_available=feature_available,
        )
        return causal_alpha_v3_forecast(
            prediction_24h,
            prediction_72h,
            residual_rmse_24h=self.residual_rmse_24h,
            residual_rmse_72h=self.residual_rmse_72h,
        )

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "config_digest": self.config.digest,
            "knowledge_cutoff": self.knowledge_cutoff,
            "model_24h_digest": self.model_24h.digest,
            "model_72h_digest": self.model_72h.digest,
            "residual_rmse_24h": self.residual_rmse_24h,
            "residual_rmse_72h": self.residual_rmse_72h,
            "sample_scope_digest": self.sample_scope_digest,
            "schema_version": _V3_FIT_EVIDENCE_SCHEMA,
            "train_symbols": self.train_symbols,
            "weight_digest_24h": self.weight_digest_24h,
            "weight_digest_72h": self.weight_digest_72h,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def fit_causal_alpha_v3(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaSymbolSamples],
    knowledge_cutoff: int,
    config: CausalAlphaV3FitConfig,
) -> CausalAlphaV3Fit:
    """Fit symbol-balanced overlap-aware 24h and 72h pooled ridge models."""

    symbols, blocks, scope_digest = _validated_scope(train_symbols, samples)
    if not isinstance(config, CausalAlphaV3FitConfig):
        raise TypeError("V3 fit requires CausalAlphaV3FitConfig")
    weights_24h = build_causal_alpha_v3_symbol_balanced_weights(
        train_symbols=symbols,
        samples=samples,
        knowledge_cutoff=knowledge_cutoff,
        horizon="24h",
    )
    weights_72h = build_causal_alpha_v3_symbol_balanced_weights(
        train_symbols=symbols,
        samples=samples,
        knowledge_cutoff=knowledge_cutoff,
        horizon="72h",
    )
    features = _pooled(blocks, "features").astype(np.float64, copy=False)
    available = _pooled(blocks, "feature_available").astype(np.bool_, copy=False)
    labels_24h = _pooled(blocks, "labels_24h").astype(np.float64, copy=False)
    labels_72h = _pooled(blocks, "labels_72h").astype(np.float64, copy=False)
    ends_24h = _pooled(blocks, "label_end_indices_24h").astype(np.int64, copy=False)
    ends_72h = _pooled(blocks, "label_end_indices_72h").astype(np.int64, copy=False)
    pooled_weights_24h = np.concatenate(tuple(weights_24h[symbol] for symbol in symbols))
    pooled_weights_72h = np.concatenate(tuple(weights_72h[symbol] for symbol in symbols))
    ridge = CausalAlphaRidgeConfig(ridge_strength=config.ridge_strength)
    feature_names = blocks[0].feature_names
    model_24h = fit_causal_alpha_ridge(
        features=features,
        labels=labels_24h,
        feature_available=available,
        label_end_indices=ends_24h,
        knowledge_cutoff=knowledge_cutoff,
        feature_names=feature_names,
        config=ridge,
        sample_weights=pooled_weights_24h,
        normalize_objective=True,
    )
    model_72h = fit_causal_alpha_ridge(
        features=features,
        labels=labels_72h,
        feature_available=available,
        label_end_indices=ends_72h,
        knowledge_cutoff=knowledge_cutoff,
        feature_names=feature_names,
        config=ridge,
        sample_weights=pooled_weights_72h,
        normalize_objective=True,
    )
    return CausalAlphaV3Fit(
        train_symbols=symbols,
        knowledge_cutoff=knowledge_cutoff,
        model_24h=model_24h,
        model_72h=model_72h,
        residual_rmse_24h=_weighted_residual_rmse(
            model_24h,
            features=features,
            feature_available=available,
            labels=labels_24h,
            weights=pooled_weights_24h,
        ),
        residual_rmse_72h=_weighted_residual_rmse(
            model_72h,
            features=features,
            feature_available=available,
            labels=labels_72h,
            weights=pooled_weights_72h,
        ),
        weight_digest_24h=_weight_digest(
            symbols,
            weights_24h,
            horizon="24h",
            knowledge_cutoff=knowledge_cutoff,
        ),
        weight_digest_72h=_weight_digest(
            symbols,
            weights_72h,
            horizon="72h",
            knowledge_cutoff=knowledge_cutoff,
        ),
        sample_scope_digest=scope_digest,
        config=config,
    )


__all__ = [
    "CausalAlphaV3Fit",
    "build_causal_alpha_v3_symbol_balanced_weights",
    "fit_causal_alpha_v3",
]
