"""Immutable label contracts for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.data.v4_context import V4ContextBlock
from trade_rl.domain.common import require_sha256

CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA: Final = "causal_alpha_v4_symbol_samples_v1"
CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA: Final = "causal_alpha_v4_residual_labels_v1"


def _readonly(value: object, *, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy(order="C")
    array.setflags(write=False)
    return array


def _validate_labels(
    *,
    labels: np.ndarray,
    ends: np.ndarray,
    rows: int,
    horizon: str,
) -> None:
    if labels.shape != (rows,) or ends.shape != (rows,):
        raise ValueError(f"V4 {horizon} label arrays must be sample aligned")
    valid = ends >= 0
    if np.any(valid & ~np.isfinite(labels)):
        raise ValueError(f"V4 {horizon} realized labels must be finite")
    if np.any(~valid & np.isfinite(labels)):
        raise ValueError(f"V4 {horizon} unavailable labels require non-finite storage")
    if np.any(~valid & (ends != -1)):
        raise ValueError(f"V4 {horizon} unavailable label ends must be -1")


@dataclass(frozen=True, slots=True)
class CausalAlphaV4SymbolSamples:
    """One train-symbol V4 feature/label table with persisted causal beta."""

    symbol: str
    dataset_id: str
    target_local_feature_names: tuple[str, ...]
    target_local_feature_schema_digest: str
    source_sample_digest: str
    source_context_digest: str
    decision_indices: np.ndarray
    target_local_features: np.ndarray
    target_local_available: np.ndarray
    instrument_descriptor_names: tuple[str, ...]
    instrument_descriptors: np.ndarray
    instrument_descriptor_available: np.ndarray
    local_context: V4ContextBlock
    global_context: V4ContextBlock
    beta: np.ndarray
    beta_available: np.ndarray
    labels_4h: np.ndarray
    label_end_indices_4h: np.ndarray
    labels_24h: np.ndarray
    label_end_indices_24h: np.ndarray
    labels_72h: np.ndarray
    label_end_indices_72h: np.ndarray
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("V4 sample symbol must be non-empty")
        for field in (
            "dataset_id",
            "target_local_feature_schema_digest",
            "source_sample_digest",
            "source_context_digest",
        ):
            require_sha256(getattr(self, field), field=f"V4 sample {field}")
        names = tuple(self.target_local_feature_names)
        if (
            not names
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "V4 target-local feature names must be non-empty and unique"
            )
        descriptor_names = tuple(self.instrument_descriptor_names)
        if descriptor_names != UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES:
            raise ValueError(
                "V4 instrument descriptor names must match the maintained order"
            )
        if set(names).intersection(descriptor_names):
            raise ValueError("V4 target-local features must not duplicate descriptors")

        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        rows = int(decisions.size)
        if rows == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("V4 sample decision indices must be strictly increasing")
        features = _readonly(self.target_local_features, dtype=np.float64)
        available = _readonly(self.target_local_available, dtype=np.bool_)
        if features.shape != (rows, len(names)) or available.shape != features.shape:
            raise ValueError("V4 target-local arrays do not match feature schema")
        if not np.isfinite(features).all():
            raise ValueError("V4 target-local features must be finite")
        descriptors = _readonly(self.instrument_descriptors, dtype=np.float64)
        descriptor_available = _readonly(
            self.instrument_descriptor_available, dtype=np.bool_
        )
        descriptor_shape = (rows, len(descriptor_names))
        if (
            descriptors.shape != descriptor_shape
            or descriptor_available.shape != descriptor_shape
        ):
            raise ValueError("V4 instrument descriptor arrays do not match schema")
        if not np.isfinite(descriptors).all():
            raise ValueError("V4 instrument descriptors must be finite")

        if not isinstance(self.local_context, V4ContextBlock) or not isinstance(
            self.global_context, V4ContextBlock
        ):
            raise TypeError("V4 samples require local/global context blocks")
        if not np.array_equal(self.local_context.decision_indices, decisions):
            raise ValueError("V4 local context decisions do not match samples")
        if not np.array_equal(self.global_context.decision_indices, decisions):
            raise ValueError("V4 global context decisions do not match samples")

        beta = _readonly(self.beta, dtype=np.float64).reshape(-1)
        beta_available = _readonly(self.beta_available, dtype=np.bool_).reshape(-1)
        if beta.shape != (rows,) or beta_available.shape != (rows,):
            raise ValueError("V4 persisted beta arrays must be sample aligned")
        if not np.isfinite(beta).all():
            raise ValueError("V4 persisted beta must be finite")
        if np.any(beta[beta_available] < -3.0) or np.any(beta[beta_available] > 3.0):
            raise ValueError("V4 persisted beta exceeds authored bounds")
        if self.symbol == "BTCUSDT" and np.any(beta[beta_available] != 1.0):
            raise ValueError("BTCUSDT available persisted beta must be exactly one")

        labels: dict[str, np.ndarray] = {}
        ends: dict[str, np.ndarray] = {}
        for horizon in ("4h", "24h", "72h"):
            label_values = _readonly(
                getattr(self, f"labels_{horizon}"), dtype=np.float64
            ).reshape(-1)
            label_ends = _readonly(
                getattr(self, f"label_end_indices_{horizon}"), dtype=np.int64
            ).reshape(-1)
            _validate_labels(
                labels=label_values,
                ends=label_ends,
                rows=rows,
                horizon=horizon,
            )
            labels[horizon] = label_values
            ends[horizon] = label_ends

        expected = content_and_arrays_digest(
            {
                "dataset_id": self.dataset_id,
                "global_context_digest": self.global_context.digest,
                "instrument_descriptor_names": descriptor_names,
                "local_context_digest": self.local_context.digest,
                "schema_version": CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA,
                "source_context_digest": self.source_context_digest,
                "source_sample_digest": self.source_sample_digest,
                "symbol": self.symbol,
                "target_local_feature_names": names,
                "target_local_feature_schema_digest": (
                    self.target_local_feature_schema_digest
                ),
            },
            (
                ("decision_indices", decisions),
                ("target_local_features", features),
                ("target_local_available", available),
                ("instrument_descriptors", descriptors),
                ("instrument_descriptor_available", descriptor_available),
                ("beta", beta),
                ("beta_available", beta_available),
                ("labels_4h", labels["4h"]),
                ("label_end_indices_4h", ends["4h"]),
                ("labels_24h", labels["24h"]),
                ("label_end_indices_24h", ends["24h"]),
                ("labels_72h", labels["72h"]),
                ("label_end_indices_72h", ends["72h"]),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 symbol sample digest mismatch")
        object.__setattr__(self, "target_local_feature_names", names)
        object.__setattr__(self, "instrument_descriptor_names", descriptor_names)
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "target_local_features", features)
        object.__setattr__(self, "target_local_available", available)
        object.__setattr__(self, "instrument_descriptors", descriptors)
        object.__setattr__(
            self, "instrument_descriptor_available", descriptor_available
        )
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "beta_available", beta_available)
        for horizon in ("4h", "24h", "72h"):
            object.__setattr__(self, f"labels_{horizon}", labels[horizon])
            object.__setattr__(
                self,
                f"label_end_indices_{horizon}",
                ends[horizon],
            )
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaV4ResidualLabels:
    """BTC market-proxy and beta-residual labels for all authored horizons."""

    symbol: str
    decision_indices: np.ndarray
    symbol_sample_digest: str
    market_proxy_sample_digest: str
    market_proxy_labels_4h: np.ndarray
    residual_labels_4h: np.ndarray
    available_4h: np.ndarray
    market_proxy_labels_24h: np.ndarray
    residual_labels_24h: np.ndarray
    available_24h: np.ndarray
    market_proxy_labels_72h: np.ndarray
    residual_labels_72h: np.ndarray
    available_72h: np.ndarray
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("V4 residual symbol must be non-empty")
        require_sha256(
            self.symbol_sample_digest,
            field="V4 residual symbol_sample_digest",
        )
        require_sha256(
            self.market_proxy_sample_digest,
            field="V4 residual market_proxy_sample_digest",
        )
        decisions = _readonly(self.decision_indices, dtype=np.int64).reshape(-1)
        rows = int(decisions.size)
        if rows == 0 or np.any(np.diff(decisions) <= 0):
            raise ValueError("V4 residual decisions must be strictly increasing")
        arrays: dict[str, np.ndarray] = {}
        for horizon in ("4h", "24h", "72h"):
            proxy = _readonly(
                getattr(self, f"market_proxy_labels_{horizon}"), dtype=np.float64
            ).reshape(-1)
            residual = _readonly(
                getattr(self, f"residual_labels_{horizon}"), dtype=np.float64
            ).reshape(-1)
            available = _readonly(
                getattr(self, f"available_{horizon}"), dtype=np.bool_
            ).reshape(-1)
            if (
                proxy.shape != (rows,)
                or residual.shape != (rows,)
                or available.shape != (rows,)
            ):
                raise ValueError(f"V4 residual {horizon} arrays are not aligned")
            if np.any(available & (~np.isfinite(proxy) | ~np.isfinite(residual))):
                raise ValueError(f"V4 residual {horizon} available rows must be finite")
            if np.any(~available & np.isfinite(residual)):
                raise ValueError(
                    f"V4 residual {horizon} unavailable rows require NaN residuals"
                )
            arrays[f"market_proxy_labels_{horizon}"] = proxy
            arrays[f"residual_labels_{horizon}"] = residual
            arrays[f"available_{horizon}"] = available
        expected = content_and_arrays_digest(
            {
                "market_proxy_sample_digest": self.market_proxy_sample_digest,
                "schema_version": CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA,
                "symbol": self.symbol,
                "symbol_sample_digest": self.symbol_sample_digest,
            },
            (
                ("decision_indices", decisions),
                *tuple((name, value) for name, value in arrays.items()),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V4 residual label digest mismatch")
        object.__setattr__(self, "decision_indices", decisions)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "digest", expected)


def _residual_for_horizon(
    *,
    symbol_labels: np.ndarray,
    market_labels: np.ndarray,
    beta: np.ndarray,
    beta_available: np.ndarray,
    symbol_ends: np.ndarray,
    market_ends: np.ndarray,
    horizon: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.array_equal(symbol_ends, market_ends):
        raise ValueError(f"V4 {horizon} symbol/BTC label ends are not aligned")
    available = (
        beta_available
        & (symbol_ends >= 0)
        & np.isfinite(symbol_labels)
        & np.isfinite(market_labels)
    )
    residual = np.full(symbol_labels.shape, np.nan, dtype=np.float64)
    residual[available] = (
        symbol_labels[available] - beta[available] * market_labels[available]
    )
    reconstructed = beta[available] * market_labels[available] + residual[available]
    if reconstructed.size and np.any(
        np.abs(reconstructed - symbol_labels[available]) > 1e-15
    ):
        raise RuntimeError(f"V4 {horizon} residual reconstruction drifted")
    return market_labels.copy(), residual, available


def build_causal_alpha_v4_residual_labels(
    *,
    symbol_samples: CausalAlphaV4SymbolSamples,
    btc_market_proxy_samples: CausalAlphaV4SymbolSamples,
) -> CausalAlphaV4ResidualLabels:
    """Decompose target returns with the persisted target beta and BTC labels."""

    if not isinstance(symbol_samples, CausalAlphaV4SymbolSamples) or not isinstance(
        btc_market_proxy_samples, CausalAlphaV4SymbolSamples
    ):
        raise TypeError("V4 residual decomposition requires V4 symbol samples")
    if btc_market_proxy_samples.symbol != "BTCUSDT":
        raise ValueError("V4 market proxy samples must be BTCUSDT")
    if not np.array_equal(
        symbol_samples.decision_indices,
        btc_market_proxy_samples.decision_indices,
    ):
        raise ValueError("V4 symbol/BTC sample decisions are not aligned")
    values: dict[str, np.ndarray] = {}
    for horizon in ("4h", "24h", "72h"):
        proxy, residual, available = _residual_for_horizon(
            symbol_labels=getattr(symbol_samples, f"labels_{horizon}"),
            market_labels=getattr(btc_market_proxy_samples, f"labels_{horizon}"),
            beta=symbol_samples.beta,
            beta_available=symbol_samples.beta_available,
            symbol_ends=getattr(symbol_samples, f"label_end_indices_{horizon}"),
            market_ends=getattr(
                btc_market_proxy_samples,
                f"label_end_indices_{horizon}",
            ),
            horizon=horizon,
        )
        values[f"market_proxy_labels_{horizon}"] = proxy
        values[f"residual_labels_{horizon}"] = residual
        values[f"available_{horizon}"] = available
    return CausalAlphaV4ResidualLabels(
        symbol=symbol_samples.symbol,
        decision_indices=symbol_samples.decision_indices,
        symbol_sample_digest=symbol_samples.digest,
        market_proxy_sample_digest=btc_market_proxy_samples.digest,
        market_proxy_labels_4h=values["market_proxy_labels_4h"],
        residual_labels_4h=values["residual_labels_4h"],
        available_4h=values["available_4h"],
        market_proxy_labels_24h=values["market_proxy_labels_24h"],
        residual_labels_24h=values["residual_labels_24h"],
        available_24h=values["available_24h"],
        market_proxy_labels_72h=values["market_proxy_labels_72h"],
        residual_labels_72h=values["residual_labels_72h"],
        available_72h=values["available_72h"],
    )


__all__ = [
    "CAUSAL_ALPHA_V4_RESIDUAL_LABELS_SCHEMA",
    "CAUSAL_ALPHA_V4_SYMBOL_SAMPLES_SCHEMA",
    "CausalAlphaV4ResidualLabels",
    "CausalAlphaV4SymbolSamples",
    "build_causal_alpha_v4_residual_labels",
]
