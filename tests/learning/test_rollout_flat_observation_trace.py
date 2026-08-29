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
        self.hybrid = SimpleNamespace(
            weights=np.asarray([0.0], dtype=np.float64),
            quantities=np.asarray([0.0], dtype=np.float64),
            mark_prices=np.asarray([1.0], dtype=np.float64),
            max_drawdown=0.0,
        )
        self.pre_trade_risk = PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.5,
                max_turnover=1.0,
            )
        )

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[np.ndarray, dict[str, object]]:
        start = options["start_idx"]
        assert isinstance(start, int)
        self.current_index = start
        self._offset = 0
        self.hybrid.weights = np.asarray([0.0], dtype=np.float64)
        return np.asarray([1.0, 2.0], dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        target = np.asarray(action, dtype=np.float64).reshape(-1)
        before = self.hybrid.weights.copy()
        # First fill undershoots the requested target; the second decision repeats
        # the same strategic target and should therefore be a trace reassertion.
        post = np.asarray([0.09 if self._offset == 0 else float(target[0])])
        self.hybrid.weights = post
        execution = SimpleNamespace(
            requested_turnover=float(np.abs(target - before).sum()),
            filled_turnover=float(np.abs(post - before).sum()),
            rejected_count=0,
            order_events=(),
        )
        info: dict[str, object] = {
            "hybrid_execution": execution,
            "hybrid_risk": SimpleNamespace(
                proposal_weights=target.copy(),
                weights=target.copy(),
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
        terminal = self._offset == 2
        return np.asarray([3.0, 4.0], dtype=np.float32), 0.0, terminal, False, info


def test_flat_observation_keeps_generic_evidence_and_records_book_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    result = rollout_evaluation.evaluate_action_path(
        _Environment(),
        evaluation_range=(0, 3),
        actions=np.asarray([[0.10], [0.10]], dtype=np.float32),
    )

    # Main's generic collapse contract treats the repeated raw action as unchanged
    # when a flat observation carries no explicit current-weight field.
    assert result.collapse_evidence.submitted_change_count == 1

    trace = result.execution_trace
    assert trace is not None
    np.testing.assert_allclose(trace.pre_action_weights[:, 0], (0.0, 0.09))
    np.testing.assert_allclose(trace.risk_constrained_weights[:, 0], (0.10, 0.10))
    np.testing.assert_allclose(trace.post_step_weights[:, 0], (0.09, 0.10))
    assert trace.strategy_intent_changes.tolist() == [True, False]
    assert trace.realized_state_follows.tolist() == [False, False]
    assert trace.rebalance_reassertions.tolist() == [False, True]
