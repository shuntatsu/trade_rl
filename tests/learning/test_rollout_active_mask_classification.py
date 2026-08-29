from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning import rollout_evaluation
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig


class _Trades:
    def __init__(self, multipliers: np.ndarray) -> None:
        del multipliers

    def seed_positions(self, *, quantities: np.ndarray, prices: np.ndarray) -> None:
        del quantities, prices

    def ingest_stateful(self, execution: object) -> None:
        del execution

    def ingest_liquidation(self, liquidation: object) -> None:
        del liquidation

    def diagnostics(self) -> SimpleNamespace:
        return SimpleNamespace(closed_trades=0, winning_trades=0)


class _Dataset:
    def resolved_array(self, name: str) -> np.ndarray:
        assert name == "contract_multipliers"
        return np.ones(1, dtype=np.float64)


class _Environment:
    dataset = _Dataset()

    def __init__(self) -> None:
        self.current_index = 0
        self._offset = 0
        self._weight = 0.0
        self.pre_trade_risk = PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.5,
                max_turnover=1.0,
            )
        )
        self.hybrid = SimpleNamespace(
            max_drawdown=0.0,
            quantities=np.zeros(1, dtype=np.float64),
            mark_prices=np.ones(1, dtype=np.float64),
        )
        self._active = (False, True, True, True)

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "current_weights": np.asarray([self._weight], dtype=np.float32),
            "active": np.asarray([self._active[self._offset]], dtype=np.float32),
        }

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        start = options["start_idx"]
        assert isinstance(start, int)
        self.current_index = start
        self._offset = 0
        self._weight = 0.0
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        if self._offset == 0:
            constrained = np.asarray([0.0], dtype=np.float64)
            filled_turnover = 0.0
        else:
            constrained = np.asarray(action, dtype=np.float64).reshape(-1).copy()
            filled_turnover = abs(float(action[0]) - self._weight)
            self._weight = float(action[0])
        execution = SimpleNamespace(
            requested_turnover=filled_turnover,
            filled_turnover=filled_turnover,
            rejected_count=0,
            order_events=(),
        )
        info: dict[str, object] = {
            "hybrid_execution": execution,
            "hybrid_risk": SimpleNamespace(
                weights=constrained,
                risk_scale=1.0,
                reasons=(),
            ),
            "hybrid_liquidation": None,
            "interval_gross_return": 0.0,
            "interval_net_return": 0.0,
            "interval_cost": 0.0,
        }
        self._offset += 1
        self.current_index += 1
        terminal = self._offset == 3
        return self._observation(), 0.0, terminal, False, info


def test_newly_active_target_is_fresh_intent_not_reassertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    result = rollout_evaluation.evaluate_action_path(
        _Environment(),
        evaluation_range=(0, 4),
        actions=np.asarray([[0.4], [0.4], [0.4]], dtype=np.float32),
    )

    trace = result.execution_trace
    assert trace is not None
    assert trace.strategy_intent_changes.tolist() == [False, True, False]
    assert trace.realized_state_follows.tolist() == [False, False, False]
    assert trace.rebalance_reassertions.tolist() == [False, False, False]
