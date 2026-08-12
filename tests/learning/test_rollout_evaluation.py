from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning import rollout_evaluation


class _Trades:
    def __init__(self, multipliers: np.ndarray) -> None:
        del multipliers

    def ingest_stateful(self, execution: object) -> None:
        del execution

    def ingest_liquidation(self, liquidation: object) -> None:
        del liquidation

    def diagnostics(self) -> SimpleNamespace:
        return SimpleNamespace(closed_trades=0, winning_trades=0)


class _Dataset:
    def resolved_array(self, name: str) -> np.ndarray:
        assert name == "contract_multipliers"
        return np.ones(2, dtype=np.float64)


class _Environment:
    dataset = _Dataset()

    def __init__(self) -> None:
        self.current_index = 0
        self._offset = 0
        self._weights = np.zeros(2, dtype=np.float32)

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "current_weights": self._weights.copy(),
            "active": np.array([1.0, 0.0], dtype=np.float32),
        }

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        self.current_index = int(options["start_idx"])
        self._offset = 0
        self._weights[:] = 0.0
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        requested = 0.0 if self._offset == 0 else 0.2
        filled = 0.0 if self._offset == 0 else 0.2
        rejected = 1 if self._offset == 0 else 0
        if filled > 0.0:
            self._weights[0] = float(action[0])
        execution = SimpleNamespace(
            requested_turnover=requested,
            filled_turnover=filled,
            rejected_count=rejected,
        )
        info = {
            "hybrid_execution": execution,
            "hybrid_liquidation": None,
            "interval_gross_return": 0.0,
            "interval_net_return": 0.0,
            "interval_cost": 0.0,
        }
        self._offset += 1
        self.current_index += 1
        terminal = self._offset == 3
        return self._observation(), 0.0, terminal, False, info


def test_action_path_reports_where_submitted_changes_disappear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment()

    result = rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=np.array(
            [[0.4, 0.9], [0.4, 0.9], [0.0, -0.9]],
            dtype=np.float32,
        ),
        action_change_tolerance=0.01,
    )

    evidence = result.collapse_evidence
    assert evidence.active_dimension_count == 3
    assert evidence.inactive_dimension_count == 3
    assert evidence.proposal_distance_count == 3
    assert evidence.submitted_change_count == 3
    assert evidence.downstream_no_trade_suppression_count == 1
    assert evidence.execution_rejection_count == 1
    assert evidence.executed_change_count == 2
    assert evidence.constant_submitted_actions is False
    assert evidence.inactive_mask_rate == 0.5


def test_action_path_can_request_stochastic_model_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment()

    class _Model:
        modes: list[bool] = []

        def predict(
            self, observation: object, *, deterministic: bool
        ) -> tuple[np.ndarray, None]:
            del observation
            self.modes.append(deterministic)
            return np.asarray([0.2, 0.0], dtype=np.float32), None

    model = _Model()
    rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        model=model,
        deterministic=False,
    )

    assert model.modes == [False, False, False]
