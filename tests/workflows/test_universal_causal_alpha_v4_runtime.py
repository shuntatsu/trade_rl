from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.universal_features import (
    UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
)
from trade_rl.data.v4_context import V4ContextBlock, V4TargetContext
from trade_rl.learning.causal_alpha_teacher import forward_log_return_label
from trade_rl.workflows.universal_causal_alpha_contracts import CausalAlphaSymbolSamples
from trade_rl.workflows.universal_causal_alpha_v4_runtime import (
    build_causal_alpha_v4_symbol_samples,
    validate_causal_alpha_v4_train_sample_scope,
)


def _digest(char: str) -> str:
    return char * 64


class _Dataset:
    regular_cadence = True

    def __init__(self, *, rows: int = 80) -> None:
        self.n_bars = rows
        index = np.arange(rows, dtype=np.float64)
        self.open = np.exp(0.001 * index)[:, None]
        self.close = np.exp(0.001 * index + 0.0002)[:, None]

    def bars_for_hours(self, hours: float) -> int:
        bars = int(round(hours * 4.0))
        if not np.isclose(bars / 4.0, hours):
            raise ValueError("hours must align to 15-minute bars")
        return bars


def _base_samples(*, symbol: str = "ETHUSDT") -> CausalAlphaSymbolSamples:
    decisions = np.arange(10, 41, dtype=np.int64)
    rows = len(decisions)
    market = decisions.astype(np.float64)[:, None]
    descriptors = np.column_stack(
        tuple(
            decisions.astype(np.float64) + float(index + 1)
            for index in range(len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES))
        )
    )
    features = np.column_stack((market, descriptors))
    unavailable = np.full(rows, np.nan, dtype=np.float64)
    missing_end = np.full(rows, -1, dtype=np.int64)
    return CausalAlphaSymbolSamples(
        symbol=symbol,
        dataset_id=_digest("a" if symbol == "BTCUSDT" else "b"),
        feature_names=("target_x", *UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES),
        feature_schema_digest=_digest("c"),
        context_digest=_digest("d"),
        reference_equity_mode="initial_capital",
        reference_equity=100_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=np.ones(features.shape, dtype=np.bool_),
        labels_24h=unavailable.copy(),
        label_end_indices_24h=missing_end.copy(),
        labels_72h=unavailable.copy(),
        label_end_indices_72h=missing_end.copy(),
    )


def _context(*, symbol: str = "ETHUSDT") -> V4TargetContext:
    decisions = np.arange(10, 41, dtype=np.int64)
    local_values = decisions.astype(np.float64)[:, None]
    global_values = (decisions.astype(np.float64) + 100.0)[:, None]
    local = V4ContextBlock(
        feature_names=("local_x",),
        decision_indices=decisions,
        values=local_values,
        available=np.ones(local_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(local_values.shape, dtype=np.float64),
        source_digest=_digest("1"),
    )
    global_market = V4ContextBlock(
        feature_names=("global_x",),
        decision_indices=decisions,
        values=global_values,
        available=np.ones(global_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(global_values.shape, dtype=np.float64),
        source_digest=_digest("2"),
    )
    beta = np.linspace(0.5, 1.5, len(decisions), dtype=np.float64)
    if symbol == "BTCUSDT":
        beta[:] = 1.0
    return V4TargetContext(
        symbol=symbol,
        local=local,
        global_market=global_market,
        beta=beta,
        beta_available=np.ones(len(decisions), dtype=np.bool_),
        beta_source_digest=_digest("3"),
        profile_name="test_v1",
    )


def test_v4_sample_builder_uses_existing_forward_label_timing_for_4h() -> None:
    dataset = _Dataset()
    base = _base_samples()
    context = _context()

    result = build_causal_alpha_v4_symbol_samples(
        base_samples=base,
        context=context,
        dataset=dataset,
        train_stop=50,
        signal_delay_decisions=1,
        decision_bars=1,
    )

    expected = forward_log_return_label(
        dataset,
        decision_index=10,
        horizon_hours=4.0,
        signal_delay_decisions=1,
        decision_bars=1,
    )
    assert result.labels_4h[0] == pytest.approx(expected.value)
    assert result.label_end_indices_4h[0] == expected.label_end_index
    late = int(np.flatnonzero(result.decision_indices == 40)[0])
    assert np.isnan(result.labels_4h[late])
    assert result.label_end_indices_4h[late] == -1


def test_v4_sample_builder_uses_persisted_context_beta_exactly() -> None:
    context = _context()
    result = build_causal_alpha_v4_symbol_samples(
        base_samples=_base_samples(),
        context=context,
        dataset=_Dataset(),
        train_stop=50,
        signal_delay_decisions=1,
        decision_bars=1,
    )
    np.testing.assert_array_equal(result.beta, context.beta)
    np.testing.assert_array_equal(result.beta_available, context.beta_available)
    assert result.source_context_digest == context.digest


def test_v4_sample_builder_separates_market_features_and_descriptors_exactly() -> None:
    base = _base_samples()
    result = build_causal_alpha_v4_symbol_samples(
        base_samples=base,
        context=_context(),
        dataset=_Dataset(),
        train_stop=50,
        signal_delay_decisions=1,
        decision_bars=1,
    )
    assert result.target_local_feature_names == ("target_x",)
    assert result.instrument_descriptor_names == UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES
    np.testing.assert_array_equal(result.target_local_features, base.features[:, :1])
    np.testing.assert_array_equal(result.instrument_descriptors, base.features[:, 1:])
    np.testing.assert_array_equal(
        np.column_stack((result.target_local_features, result.instrument_descriptors)),
        base.features,
    )
    np.testing.assert_array_equal(
        np.column_stack(
            (result.target_local_available, result.instrument_descriptor_available)
        ),
        base.feature_available,
    )


def test_v4_sample_builder_rejects_descriptor_suffix_drift() -> None:
    base = _base_samples()
    names = list(base.feature_names)
    names[-1] = "wrong_descriptor"
    drifted = replace(base, feature_names=tuple(names), digest="")
    with pytest.raises(ValueError, match="descriptor"):
        build_causal_alpha_v4_symbol_samples(
            base_samples=drifted,
            context=_context(),
            dataset=_Dataset(),
            train_stop=50,
            signal_delay_decisions=1,
            decision_bars=1,
        )


def test_v4_sample_builder_future_price_mutation_after_train_stop_changes_nothing() -> (
    None
):
    first_dataset = _Dataset()
    second_dataset = _Dataset()
    second_dataset.open[50:] *= 100.0
    second_dataset.close[50:] *= 0.01
    base = _base_samples()
    context = _context()

    first = build_causal_alpha_v4_symbol_samples(
        base_samples=base,
        context=context,
        dataset=first_dataset,
        train_stop=50,
        signal_delay_decisions=1,
        decision_bars=1,
    )
    second = build_causal_alpha_v4_symbol_samples(
        base_samples=base,
        context=context,
        dataset=second_dataset,
        train_stop=50,
        signal_delay_decisions=1,
        decision_bars=1,
    )

    np.testing.assert_array_equal(first.labels_4h, second.labels_4h)
    np.testing.assert_array_equal(
        first.label_end_indices_4h,
        second.label_end_indices_4h,
    )


def test_v4_sample_builder_rejects_context_decision_drift() -> None:
    context = _context()
    decisions = context.local.decision_indices.copy() + 1
    drifted_local = V4ContextBlock(
        feature_names=context.local.feature_names,
        decision_indices=decisions,
        values=context.local.values,
        available=context.local.available,
        staleness_hours=context.local.staleness_hours,
        source_digest=_digest("4"),
    )
    drifted_global = V4ContextBlock(
        feature_names=context.global_market.feature_names,
        decision_indices=decisions,
        values=context.global_market.values,
        available=context.global_market.available,
        staleness_hours=context.global_market.staleness_hours,
        source_digest=_digest("5"),
    )
    drifted = replace(
        context, local=drifted_local, global_market=drifted_global, digest=""
    )

    with pytest.raises(ValueError, match="decision"):
        build_causal_alpha_v4_symbol_samples(
            base_samples=_base_samples(),
            context=drifted,
            dataset=_Dataset(),
            train_stop=50,
            signal_delay_decisions=1,
            decision_bars=1,
        )


def test_v4_btc_samples_preserve_persisted_unit_beta() -> None:
    context = _context(symbol="BTCUSDT")
    result = build_causal_alpha_v4_symbol_samples(
        base_samples=_base_samples(symbol="BTCUSDT"),
        context=context,
        dataset=_Dataset(),
        train_stop=50,
        signal_delay_decisions=1,
        decision_bars=1,
    )
    np.testing.assert_array_equal(result.beta, np.ones(len(result.beta)))
    np.testing.assert_array_equal(result.beta, context.beta)


def test_v4_train_scope_rejects_validation_or_test_samples() -> None:
    common = dict(
        dataset=_Dataset(),
        train_stop=50,
        signal_delay_decisions=1,
        decision_bars=1,
    )
    train = build_causal_alpha_v4_symbol_samples(
        base_samples=_base_samples(symbol="ETHUSDT"),
        context=_context(symbol="ETHUSDT"),
        **common,
    )
    validation = build_causal_alpha_v4_symbol_samples(
        base_samples=_base_samples(symbol="SOLUSDT"),
        context=_context(symbol="SOLUSDT"),
        **common,
    )

    with pytest.raises(ValueError, match="exactly match train_symbols"):
        validate_causal_alpha_v4_train_sample_scope(
            train_symbols=("ETHUSDT",),
            samples={"ETHUSDT": train, "SOLUSDT": validation},
        )
