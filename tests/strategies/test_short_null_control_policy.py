"""Native-portfolio comparators; no shadow account or future market data."""

from __future__ import annotations

import numpy as np
import pytest

from tests.strategies.test_divergence_cloud_follow import (
    _native_market,
    _signals,
    _snapshot,
)
from trade_rl.strategies.short_null_control import (
    ShortNullControlConfig,
    ShortNullControlPolicy,
)


def test_long_only_blocks_short_and_permits_c2_long() -> None:
    downside = _signals([-1] * 55)
    no_short = ShortNullControlPolicy(
        symbol="BTCUSDT",
        signals=downside,
        short_permission=np.zeros(55, dtype=bool),
        config=ShortNullControlConfig(mode="long_only"),
    )
    assert no_short.decide_portfolio(_snapshot(downside, 6)).target_weights == (0.0,)
    assert no_short.blocked_short_proposals == 1
    upside = _signals([1] * 55)
    long_only = ShortNullControlPolicy(
        symbol="BTCUSDT",
        signals=upside,
        short_permission=np.zeros(55, dtype=bool),
        config=ShortNullControlConfig(mode="long_only"),
    )
    assert long_only.decide_portfolio(_snapshot(upside, 6)).target_weights == (0.25,)


def test_weekly_short_permission_exits_when_blocked() -> None:
    s = _signals([-1] * 55)
    allowed = np.ones(55, dtype=bool)
    allowed[12:] = False
    policy = ShortNullControlPolicy(
        symbol="BTCUSDT",
        signals=s,
        short_permission=allowed,
        config=ShortNullControlConfig(mode="weekly_random"),
    )
    assert policy.decide_portfolio(_snapshot(s, 6)).target_weights == (-0.25,)
    assert policy.decide_portfolio(
        _snapshot(s, 11, quantity=-3, weight=-0.2)
    ).hold_current_quantity_symbols == (0,)
    assert policy.decide_portfolio(
        _snapshot(s, 12, quantity=-3, weight=-0.2)
    ).target_weights == (0.0,)
    assert policy.blocked_short_proposals == 0


def test_native_long_only_runs_zero_short_fills_and_pays_long_fee() -> None:
    from trade_rl.evaluation.replay import run_shared_portfolio_replay
    from trade_rl.simulation import ExecutionCostConfig

    market = _native_market()
    s = _signals([1] * 25 + [-1] * 30)
    policy = ShortNullControlPolicy(
        symbol="BTCUSDT",
        signals=s,
        short_permission=np.zeros(55, dtype=bool),
        config=ShortNullControlConfig(mode="long_only"),
    )
    out = run_shared_portfolio_replay(
        market,
        policy,
        start_index=0,
        stop_index=54,
        execution_cost=ExecutionCostConfig(),
        initial_capital=10000,
        capture_ledger_evidence=True,
    )
    assert out.book.fill_count == 2
    assert out.book.total_cost > 0
    assert out.ledger_evidence is not None


@pytest.mark.parametrize("bad", ["missing", 0, None])
def test_unknown_mode_rejected(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ShortNullControlConfig(mode=bad)


def test_terminated_account_is_passed_to_native_C2_without_new_flat_order() -> None:
    from dataclasses import replace

    s = _signals([-1] * 55)
    policy = ShortNullControlPolicy(
        symbol="BTCUSDT",
        signals=s,
        short_permission=np.zeros(55, dtype=bool),
        config=ShortNullControlConfig(mode="long_only"),
    )
    snap = _snapshot(s, 9, quantity=-2, weight=-0.2)
    ended = replace(
        snap,
        account=replace(snap.account, insolvent=True, termination_reason="insolvency"),
    )
    action = policy.decide_portfolio(ended)
    assert action.hold_current_quantity_symbols == (0,)
    assert action.target_weights == (-0.2,)
    assert policy.ineligible_short_flat_requests == 0


def test_long_only_rejects_true_mask_and_non_boolean_mask() -> None:
    s = _signals([-1] * 55)
    with pytest.raises(ValueError, match="long-only"):
        ShortNullControlPolicy(
            symbol="BTCUSDT",
            signals=s,
            short_permission=np.ones(55, dtype=bool),
            config=ShortNullControlConfig(mode="long_only"),
        )
    with pytest.raises(ValueError, match="boolean"):
        ShortNullControlPolicy(
            symbol="BTCUSDT",
            signals=s,
            short_permission=np.zeros(55, dtype=np.int64),
            config=ShortNullControlConfig(mode="weekly_random"),
        )
