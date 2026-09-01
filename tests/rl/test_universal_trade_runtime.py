from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from tests.rl.universal_trade_test_support import make_u1_base_env, make_u1_market
from trade_rl.rl.universal_trade_environment import UniversalTradeMarketEnv
from trade_rl.simulation.execution import ExecutionCostConfig


def test_u1_environment_rejects_multi_symbol_before_base_initialization() -> None:
    dataset = cast(Any, SimpleNamespace(n_symbols=2))
    with pytest.raises(ValueError, match="single-symbol"):
        UniversalTradeMarketEnv(dataset)


def test_runtime_separates_submission_pending_risk_and_realized_weight() -> None:
    env = make_u1_base_env(
        dataset=make_u1_market(volume=100.0),
        max_abs_weight=0.35,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=0.01,
            maintenance_margin_rate=0.0,
        ),
    )
    env.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    env.step(np.asarray([0.60], dtype=np.float32))
    env.step(np.asarray([0.80], dtype=np.float32))

    snapshot = env.universal_trade_runtime_snapshot()
    assert snapshot.policy_requested_weight == pytest.approx(0.80)
    assert snapshot.pending_target_active is True
    assert snapshot.pending_target_weight == pytest.approx(0.80)
    assert snapshot.risk_projected_weight == pytest.approx(0.35)
    assert abs(snapshot.current_weight) < abs(snapshot.risk_projected_weight)
    assert 0.0 <= snapshot.fill_ratio < 1.0


def test_pending_flat_is_distinct_from_no_pending_target() -> None:
    env = make_u1_base_env()
    env.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    assert env.universal_trade_runtime_snapshot().pending_target_active is False

    env.step(np.asarray([0.0], dtype=np.float32))
    snapshot = env.universal_trade_runtime_snapshot()
    assert snapshot.pending_target_active is True
    assert snapshot.pending_target_weight == pytest.approx(0.0)
