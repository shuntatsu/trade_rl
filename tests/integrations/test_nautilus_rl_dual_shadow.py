from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nautilus_trader")

from trade_rl.data.market import MarketDataset
from trade_rl.integrations.nautilus.rl_dual_shadow import NautilusEnvironmentDualShadow
from trade_rl.rl.environment_execution import ExecutionDualShadowRequest


def _market(*, symbol: str = "BTCUSDT") -> MarketDataset:
    n_bars = 6
    shape = (n_bars, 1)
    close = np.full(shape, 100.0)
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=(symbol,),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close.copy(),
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close.copy(),
        index_price=close.copy(),
    )


@pytest.mark.nautilus
def test_rl_dual_shadow_rejects_non_maintained_symbol() -> None:
    with pytest.raises(ValueError, match="BTCUSDT"):
        NautilusEnvironmentDualShadow(_market(symbol="ETHUSDT"))


@pytest.mark.nautilus
def test_rl_dual_shadow_replays_prefixes_in_fresh_children() -> None:
    runtime = NautilusEnvironmentDualShadow(_market(), no_trade_band=0.0)
    runtime.reset(
        start_index=0,
        initial_capital=1_000.0,
        initial_quantities=(0.0,),
    )

    first = runtime.observe(
        ExecutionDualShadowRequest(
            target=(0.1,),
            start_index=0,
            end_index=1,
            allocated_equity=1_000.0,
            legacy_terminal_quantities=(1.0,),
        )
    )
    second = runtime.observe(
        ExecutionDualShadowRequest(
            target=(0.0,),
            start_index=1,
            end_index=2,
            allocated_equity=1_000.0,
            legacy_terminal_quantities=(0.0,),
        )
    )

    assert first.worker_pid != second.worker_pid
    assert first.structural_parity is True
    assert first.candidate_terminal_quantities == (1.0,)
    assert second.structural_parity is True
    assert second.candidate_terminal_quantities == (0.0,)
    assert runtime.step_count == 2
