from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.v4_context import V4ContextBlock
from trade_rl.data.universal_features import (
    UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
)
from trade_rl.learning.causal_alpha_v4 import (
    CausalAlphaV4SymbolSamples,
    build_causal_alpha_v4_residual_labels,
)


def _digest(char: str) -> str:
    return char * 64


def _block(*, decisions: np.ndarray, source: str, offset: float) -> V4ContextBlock:
    values = (decisions.astype(np.float64)[:, None] + offset).copy()
    return V4ContextBlock(
        feature_names=(f"feature_{source}",),
        decision_indices=decisions,
        values=values,
        available=np.ones(values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(values.shape, dtype=np.float64),
        source_digest=_digest(source),
    )


def _samples(
    *,
    symbol: str,
    beta: np.ndarray,
    labels_4h: np.ndarray,
    labels_24h: np.ndarray,
    labels_72h: np.ndarray,
) -> CausalAlphaV4SymbolSamples:
    decisions = np.asarray([10, 11, 12], dtype=np.int64)
    target = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)
    return CausalAlphaV4SymbolSamples(
        symbol=symbol,
        dataset_id=_digest("a" if symbol == "BTCUSDT" else "b"),
        target_local_feature_names=("target_x",),
        target_local_feature_schema_digest=_digest("c"),
        source_sample_digest=_digest("d" if symbol == "BTCUSDT" else "e"),
        source_context_digest=_digest("f" if symbol == "BTCUSDT" else "0"),
        decision_indices=decisions,
        target_local_features=target,
        target_local_available=np.ones(target.shape, dtype=np.bool_),
        instrument_descriptor_names=UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES,
        instrument_descriptors=np.ones(
            (len(decisions), len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)),
            dtype=np.float64,
        ),
        instrument_descriptor_available=np.ones(
            (len(decisions), len(UNIVERSAL_INSTRUMENT_DESCRIPTOR_NAMES)),
            dtype=np.bool_,
        ),
        local_context=_block(decisions=decisions, source="1", offset=10.0),
        global_context=_block(decisions=decisions, source="2", offset=20.0),
        beta=beta,
        beta_available=np.asarray([True, True, True], dtype=np.bool_),
        labels_4h=labels_4h,
        label_end_indices_4h=np.asarray([27, 28, 29], dtype=np.int64),
        labels_24h=labels_24h,
        label_end_indices_24h=np.asarray([107, 108, 109], dtype=np.int64),
        labels_72h=labels_72h,
        label_end_indices_72h=np.asarray([299, 300, 301], dtype=np.int64),
    )


def test_residual_labels_reconstruct_symbol_returns_exactly() -> None:
    btc = _samples(
        symbol="BTCUSDT",
        beta=np.ones(3, dtype=np.float64),
        labels_4h=np.asarray([0.01, -0.02, 0.03]),
        labels_24h=np.asarray([0.04, 0.01, -0.02]),
        labels_72h=np.asarray([0.09, 0.03, -0.06]),
    )
    symbol = _samples(
        symbol="ETHUSDT",
        beta=np.asarray([0.5, 1.5, 2.0]),
        labels_4h=np.asarray([0.015, -0.01, 0.08]),
        labels_24h=np.asarray([0.03, 0.025, -0.01]),
        labels_72h=np.asarray([0.10, 0.07, -0.02]),
    )

    residual = build_causal_alpha_v4_residual_labels(
        symbol_samples=symbol,
        btc_market_proxy_samples=btc,
    )

    for horizon in ("4h", "24h", "72h"):
        proxy = getattr(residual, f"market_proxy_labels_{horizon}")
        values = getattr(residual, f"residual_labels_{horizon}")
        available = getattr(residual, f"available_{horizon}")
        original = getattr(symbol, f"labels_{horizon}")
        reconstructed = symbol.beta[available] * proxy[available] + values[available]
        np.testing.assert_allclose(
            reconstructed,
            original[available],
            atol=1e-15,
            rtol=0.0,
        )


def test_btc_residual_is_exact_zero_when_beta_is_one() -> None:
    btc = _samples(
        symbol="BTCUSDT",
        beta=np.ones(3, dtype=np.float64),
        labels_4h=np.asarray([0.01, -0.02, 0.03]),
        labels_24h=np.asarray([0.04, 0.01, -0.02]),
        labels_72h=np.asarray([0.09, 0.03, -0.06]),
    )

    residual = build_causal_alpha_v4_residual_labels(
        symbol_samples=btc,
        btc_market_proxy_samples=btc,
    )

    np.testing.assert_array_equal(residual.residual_labels_4h, np.zeros(3))
    np.testing.assert_array_equal(residual.residual_labels_24h, np.zeros(3))
    np.testing.assert_array_equal(residual.residual_labels_72h, np.zeros(3))


def test_unavailable_persisted_beta_makes_residual_unavailable() -> None:
    btc = _samples(
        symbol="BTCUSDT",
        beta=np.ones(3, dtype=np.float64),
        labels_4h=np.asarray([0.01, -0.02, 0.03]),
        labels_24h=np.asarray([0.04, 0.01, -0.02]),
        labels_72h=np.asarray([0.09, 0.03, -0.06]),
    )
    symbol = _samples(
        symbol="ETHUSDT",
        beta=np.asarray([0.5, 1.5, 2.0]),
        labels_4h=np.asarray([0.015, -0.01, 0.08]),
        labels_24h=np.asarray([0.03, 0.025, -0.01]),
        labels_72h=np.asarray([0.10, 0.07, -0.02]),
    )
    unavailable = symbol.beta_available.copy()
    unavailable[1] = False
    symbol = replace(symbol, beta_available=unavailable, digest="")

    residual = build_causal_alpha_v4_residual_labels(
        symbol_samples=symbol,
        btc_market_proxy_samples=btc,
    )

    assert not residual.available_4h[1]
    assert not residual.available_24h[1]
    assert not residual.available_72h[1]
    assert np.isnan(residual.residual_labels_4h[1])
    assert np.isnan(residual.residual_labels_24h[1])
    assert np.isnan(residual.residual_labels_72h[1])


def test_residual_labels_reject_btc_label_end_mismatch() -> None:
    btc = _samples(
        symbol="BTCUSDT",
        beta=np.ones(3, dtype=np.float64),
        labels_4h=np.asarray([0.01, -0.02, 0.03]),
        labels_24h=np.asarray([0.04, 0.01, -0.02]),
        labels_72h=np.asarray([0.09, 0.03, -0.06]),
    )
    symbol = _samples(
        symbol="ETHUSDT",
        beta=np.asarray([0.5, 1.5, 2.0]),
        labels_4h=np.asarray([0.015, -0.01, 0.08]),
        labels_24h=np.asarray([0.03, 0.025, -0.01]),
        labels_72h=np.asarray([0.10, 0.07, -0.02]),
    )
    mismatched_ends = btc.label_end_indices_4h.copy()
    mismatched_ends[1] += 1
    btc = replace(btc, label_end_indices_4h=mismatched_ends, digest="")

    with pytest.raises(ValueError, match="label ends"):
        build_causal_alpha_v4_residual_labels(
            symbol_samples=symbol,
            btc_market_proxy_samples=btc,
        )
