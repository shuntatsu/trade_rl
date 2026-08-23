"""Train-only sample assembly for the research-only Causal Alpha V4 lane."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.universal_features import UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
from trade_rl.data.v4_context import V4TargetContext
from trade_rl.learning.causal_alpha_teacher import forward_log_return_label
from trade_rl.learning.causal_alpha_v4 import CausalAlphaV4SymbolSamples
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples


class _PrefixLabelDataset:
    """Minimal label view that makes rows at/after train_stop physically absent."""

    def __init__(self, dataset: Any, *, train_stop: int) -> None:
        n_bars = getattr(dataset, "n_bars", None)
        if (
            isinstance(n_bars, bool)
            or not isinstance(n_bars, int)
            or n_bars <= 0
            or isinstance(train_stop, bool)
            or not isinstance(train_stop, int)
            or not 0 < train_stop <= n_bars
        ):
            raise ValueError("V4 prefix label view requires a valid train_stop")
        open_values = np.asarray(getattr(dataset, "open", None), dtype=np.float64)
        close_values = np.asarray(getattr(dataset, "close", None), dtype=np.float64)
        if open_values.ndim != 2 or close_values.shape != open_values.shape:
            raise ValueError("V4 label dataset must expose aligned price matrices")
        if len(open_values) != n_bars:
            raise ValueError("V4 label dataset price rows do not match n_bars")
        bars_for_hours = getattr(dataset, "bars_for_hours", None)
        if not callable(bars_for_hours):
            raise TypeError("V4 label dataset cannot resolve exact real-time horizons")
        self.regular_cadence = bool(getattr(dataset, "regular_cadence", False))
        self.n_bars = train_stop
        self.open = open_values[:train_stop].copy(order="C")
        self.close = close_values[:train_stop].copy(order="C")
        self._bars_for_hours = bars_for_hours

    def bars_for_hours(self, hours: float) -> int:
        return int(self._bars_for_hours(hours))


def _build_4h_labels(
    *,
    dataset: Any,
    decision_indices: np.ndarray,
    train_stop: int,
    signal_delay_decisions: int,
    decision_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = _PrefixLabelDataset(dataset, train_stop=train_stop)
    labels = np.full(decision_indices.shape, np.nan, dtype=np.float64)
    ends = np.full(decision_indices.shape, -1, dtype=np.int64)
    for row, raw_decision in enumerate(decision_indices):
        decision = int(raw_decision)
        if decision >= train_stop:
            raise ValueError("V4 sample decision lies outside train_stop")
        try:
            label = forward_log_return_label(
                prefix,
                decision_index=decision,
                horizon_hours=4.0,
                signal_delay_decisions=signal_delay_decisions,
                decision_bars=decision_bars,
            )
        except ValueError as error:
            if str(error) != "label horizon is incomplete inside the dataset":
                raise
            continue
        labels[row] = label.value
        ends[row] = label.label_end_index
    return labels, ends


def _split_v3_feature_surface(
    base_samples: CausalAlphaSymbolSamples,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    names = tuple(base_samples.feature_names)
    descriptor_names = UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
    descriptor_count = len(descriptor_names)
    if len(names) <= descriptor_count or names[-descriptor_count:] != descriptor_names:
        raise ValueError(
            "V4 source feature schema must end with the maintained instrument descriptors"
        )
    target_names = names[:-descriptor_count]
    split = len(target_names)
    target_features = np.asarray(
        base_samples.features[:, :split], dtype=np.float64
    ).copy(order="C")
    target_available = np.asarray(
        base_samples.feature_available[:, :split], dtype=np.bool_
    ).copy(order="C")
    descriptors = np.asarray(
        base_samples.features[:, split:], dtype=np.float64
    ).copy(order="C")
    descriptor_available = np.asarray(
        base_samples.feature_available[:, split:], dtype=np.bool_
    ).copy(order="C")
    if descriptors.shape[1] != descriptor_count:
        raise RuntimeError("V4 descriptor split width drifted")
    return (
        target_names,
        target_features,
        target_available,
        descriptors,
        descriptor_available,
    )


def build_causal_alpha_v4_symbol_samples(
    *,
    base_samples: CausalAlphaSymbolSamples,
    context: V4TargetContext,
    dataset: Any,
    train_stop: int,
    signal_delay_decisions: int,
    decision_bars: int,
) -> CausalAlphaV4SymbolSamples:
    """Add V4 context, persisted beta, and a 4h label to train-only base samples."""

    if not isinstance(base_samples, CausalAlphaSymbolSamples):
        raise TypeError("V4 base_samples must be CausalAlphaSymbolSamples")
    if not isinstance(context, V4TargetContext):
        raise TypeError("V4 context must be V4TargetContext")
    if base_samples.symbol != context.symbol:
        raise ValueError("V4 sample/context symbol identity drifted")
    if not np.array_equal(
        base_samples.decision_indices, context.local.decision_indices
    ):
        raise ValueError("V4 sample/local-context decision indices drifted")
    if not np.array_equal(
        base_samples.decision_indices,
        context.global_market.decision_indices,
    ):
        raise ValueError("V4 sample/global-context decision indices drifted")
    if context.beta.shape != base_samples.decision_indices.shape or (
        context.beta_available.shape != base_samples.decision_indices.shape
    ):
        raise ValueError("V4 persisted beta is not sample aligned")
    if np.any(context.beta[context.beta_available] < -3.0) or np.any(
        context.beta[context.beta_available] > 3.0
    ):
        raise ValueError("V4 persisted beta exceeds authored bounds")
    if context.symbol == "BTCUSDT" and np.any(
        context.beta[context.beta_available] != 1.0
    ):
        raise ValueError("BTCUSDT available persisted beta must be exactly one")
    if signal_delay_decisions not in {0, 1}:
        raise ValueError("V4 signal_delay_decisions must be zero or one")
    if (
        isinstance(decision_bars, bool)
        or not isinstance(decision_bars, int)
        or decision_bars <= 0
    ):
        raise ValueError("V4 decision_bars must be a positive integer")

    (
        target_names,
        target_features,
        target_available,
        descriptors,
        descriptor_available,
    ) = _split_v3_feature_surface(base_samples)
    labels_4h, ends_4h = _build_4h_labels(
        dataset=dataset,
        decision_indices=base_samples.decision_indices,
        train_stop=train_stop,
        signal_delay_decisions=signal_delay_decisions,
        decision_bars=decision_bars,
    )
    feature_schema_digest = content_digest(
        {
            "base_feature_schema_digest": base_samples.feature_schema_digest,
            "instrument_descriptor_names": UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
            "schema_version": "causal_alpha_v4_target_local_feature_schema_v2",
            "source_context_digest": base_samples.context_digest,
            "target_local_feature_names": target_names,
        }
    )
    return CausalAlphaV4SymbolSamples(
        symbol=base_samples.symbol,
        dataset_id=base_samples.dataset_id,
        target_local_feature_names=target_names,
        target_local_feature_schema_digest=feature_schema_digest,
        source_sample_digest=base_samples.digest,
        source_context_digest=context.digest,
        decision_indices=base_samples.decision_indices,
        target_local_features=target_features,
        target_local_available=target_available,
        instrument_descriptor_names=UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
        instrument_descriptors=descriptors,
        instrument_descriptor_available=descriptor_available,
        local_context=context.local,
        global_context=context.global_market,
        beta=context.beta,
        beta_available=context.beta_available,
        labels_4h=labels_4h,
        label_end_indices_4h=ends_4h,
        labels_24h=base_samples.labels_24h,
        label_end_indices_24h=base_samples.label_end_indices_24h,
        labels_72h=base_samples.labels_72h,
        label_end_indices_72h=base_samples.label_end_indices_72h,
    )


def validate_causal_alpha_v4_train_sample_scope(
    *,
    train_symbols: tuple[str, ...],
    samples: Mapping[str, CausalAlphaV4SymbolSamples],
) -> dict[str, CausalAlphaV4SymbolSamples]:
    """Fail closed unless V4 fitting receives exactly the authored train symbols."""

    symbols = tuple(train_symbols)
    if (
        not symbols
        or len(set(symbols)) != len(symbols)
        or any(not isinstance(symbol, str) or not symbol for symbol in symbols)
    ):
        raise ValueError("V4 train_symbols must be non-empty and unique")
    if set(samples) != set(symbols):
        raise ValueError("V4 samples must exactly match train_symbols")
    ordered: dict[str, CausalAlphaV4SymbolSamples] = {}
    for symbol in symbols:
        sample = samples[symbol]
        if (
            not isinstance(sample, CausalAlphaV4SymbolSamples)
            or sample.symbol != symbol
        ):
            raise ValueError("V4 train sample symbol identity drifted")
        ordered[symbol] = sample
    names = {sample.target_local_feature_names for sample in ordered.values()}
    schemas = {sample.target_local_feature_schema_digest for sample in ordered.values()}
    descriptor_names = {
        sample.instrument_descriptor_names for sample in ordered.values()
    }
    local_names = {sample.local_context.feature_names for sample in ordered.values()}
    global_names = {sample.global_context.feature_names for sample in ordered.values()}
    if len(names) != 1 or len(schemas) != 1:
        raise ValueError("V4 target-local feature schema drifted across train symbols")
    if descriptor_names != {UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES}:
        raise ValueError("V4 descriptor schema drifted across train symbols")
    if len(local_names) != 1 or len(global_names) != 1:
        raise ValueError("V4 context feature schema drifted across train symbols")
    return ordered


__all__ = [
    "build_causal_alpha_v4_symbol_samples",
    "validate_causal_alpha_v4_train_sample_scope",
]
