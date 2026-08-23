from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.v4_context import (
    CROSS_MARKET_CORE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    V4ContextBlock,
    V4TargetContext,
)
from trade_rl.rl.universal_v4_context import V4ContextProvider


def _digest(char: str) -> str:
    return char * 64


def _context(
    *,
    symbol: str = "ETHUSDT",
    local_names: tuple[str, ...] = CROSS_MARKET_CORE_NAMES,
    global_names: tuple[str, ...] = GLOBAL_MARKET_CORE_NAMES,
    profile_name: str = "cross_market_core_v1",
) -> V4TargetContext:
    decisions = np.asarray([100, 101], dtype=np.int64)
    local_values = np.arange(2 * len(local_names), dtype=np.float64).reshape(
        2, len(local_names)
    )
    global_values = np.arange(2 * len(global_names), dtype=np.float64).reshape(
        2, len(global_names)
    ) + 1000.0
    local = V4ContextBlock(
        feature_names=local_names,
        decision_indices=decisions,
        values=local_values,
        available=np.ones(local_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(local_values.shape, dtype=np.float64),
        source_digest=_digest("1"),
    )
    global_market = V4ContextBlock(
        feature_names=global_names,
        decision_indices=decisions,
        values=global_values,
        available=np.ones(global_values.shape, dtype=np.bool_),
        staleness_hours=np.zeros(global_values.shape, dtype=np.float64),
        source_digest=_digest("2"),
    )
    return V4TargetContext(
        symbol=symbol,
        local=local,
        global_market=global_market,
        beta=np.asarray([0.75, 1.25], dtype=np.float64),
        beta_available=np.asarray([True, True], dtype=np.bool_),
        beta_source_digest=_digest("3"),
        profile_name=profile_name,
    )


def test_v4_context_provider_resolves_exact_artifact_backed_row() -> None:
    context = _context()
    provider = V4ContextProvider(contexts={context.symbol: context})

    resolved = provider.resolve(symbol="ETHUSDT", decision_index=101)

    assert provider.local_width == len(CROSS_MARKET_CORE_NAMES)
    assert provider.global_width == len(GLOBAL_MARKET_CORE_NAMES)
    assert len(provider.schema_digest) == 64
    assert len(provider.digest) == 64
    assert resolved.digest == context.policy_row_digest(1)
    assert resolved.local_values.dtype == np.float32
    assert resolved.local_values.shape == (1, len(CROSS_MARKET_CORE_NAMES))
    assert resolved.global_values.shape == (1, len(GLOBAL_MARKET_CORE_NAMES))
    assert resolved.beta.shape == (1, 1)
    assert resolved.beta_available.shape == (1, 1)
    np.testing.assert_allclose(resolved.local_values, context.local.values[1:2])
    np.testing.assert_allclose(
        resolved.global_values, context.global_market.values[1:2]
    )
    np.testing.assert_allclose(resolved.beta, [[1.25]])
    np.testing.assert_array_equal(resolved.beta_available, [[1.0]])


def test_v4_context_provider_rejects_unknown_symbol() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    with pytest.raises(ValueError, match="symbol"):
        provider.resolve(symbol="SOLUSDT", decision_index=100)


def test_v4_context_provider_rejects_missing_decision_index() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    with pytest.raises(ValueError, match="decision"):
        provider.resolve(symbol="ETHUSDT", decision_index=99)


def test_v4_context_provider_does_not_accept_external_beta() -> None:
    provider = V4ContextProvider(contexts={"ETHUSDT": _context()})
    with pytest.raises(TypeError):
        provider.resolve(  # type: ignore[call-arg]
            symbol="ETHUSDT",
            decision_index=100,
            beta=9.0,
            beta_available=True,
        )


def test_v4_context_provider_rejects_feature_order_drift() -> None:
    left = _context(symbol="ETHUSDT")
    right = _context(symbol="SOLUSDT")
    names = tuple(reversed(right.local.feature_names))
    drifted_local = V4ContextBlock(
        feature_names=names,
        decision_indices=right.local.decision_indices,
        values=right.local.values[:, ::-1],
        available=right.local.available[:, ::-1],
        staleness_hours=right.local.staleness_hours[:, ::-1],
        source_digest=_digest("4"),
    )
    right = replace(right, local=drifted_local, digest="")

    with pytest.raises(ValueError, match="schema|feature"):
        V4ContextProvider(contexts={left.symbol: left, right.symbol: right})


def test_v4_context_provider_rejects_profile_width_drift() -> None:
    context = _context(local_names=CROSS_MARKET_CORE_NAMES[:-1])
    with pytest.raises(ValueError, match="width|feature"):
        V4ContextProvider(contexts={context.symbol: context})
