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
        return np.ones(1, dtype=np.float64)


class _Environment:
    dataset = _Dataset()

    def __init__(
        self,
        *,
        risk_weights: float = 0.10,
        realized_weights: tuple[float, float, float] = (0.10, 0.10, 0.10),
        submitted_targets: tuple[float, float, float] = (0.10, 0.10, 0.10),
        executed_targets: tuple[float, float, float] = (0.0, 0.10, 0.10),
        risk_reasons: tuple[tuple[str, ...], ...] = ((), (), ()),
        liquidation_terminal: tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        self.current_index = 0
        self._offset = 0
        self._current = 0.0
        self._risk_weights = float(risk_weights)
        self._realized_weights = realized_weights
        self._submitted_targets = submitted_targets
        self._executed_targets = executed_targets
        self._risk_reasons = risk_reasons
        self._liquidation_terminal = liquidation_terminal

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "current_weights": np.array([self._current], dtype=np.float32),
            "active": np.array([1.0], dtype=np.float32),
        }

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        start = options["start_idx"]
        assert isinstance(start, int)
        self.current_index = start
        self._offset = 0
        self._current = 0.0
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        offset = self._offset
        risk_weight = self._risk_weights if offset == 0 else float(action[0])
        risk = SimpleNamespace(
            reasons=self._risk_reasons[offset],
            pretrade_weights=np.array([risk_weight], dtype=np.float64),
            weights=np.array([risk_weight], dtype=np.float64),
            risk_scale=1.0,
            max_abs_weight=0.10,
            max_gross=1.0,
            fail_closed_tolerance=1e-10,
        )
        self._current = float(self._realized_weights[offset])
        execution = SimpleNamespace(
            requested_turnover=abs(float(action[0])),
            filled_turnover=0.10 if offset in (0, 2) else 0.0,
            rejected_count=0,
            order_events=(),
        )
        liquidation = (
            SimpleNamespace(
                interval_gross_return=0.0,
                interval_net_return=0.0,
                interval_cost=0.0,
                filled_turnover=0.0,
            )
            if self._liquidation_terminal[offset]
            else None
        )
        info: dict[str, object] = {
            "hybrid_execution": execution,
            "hybrid_risk": risk,
            "hybrid_liquidation": liquidation,
            "interval_gross_return": 0.0,
            "interval_net_return": 0.0,
            "interval_cost": 0.0,
            "submitted_target": np.array(
                [self._submitted_targets[offset]], dtype=np.float64
            ),
            "executed_target": np.array(
                [self._executed_targets[offset]], dtype=np.float64
            ),
            "effective_filled_weights": np.array([self._current], dtype=np.float64),
            "liquidation_terminal": self._liquidation_terminal[offset],
            "termination_reason": "liquidation" if self._liquidation_terminal[offset] else None,
        }
        self._offset += 1
        self.current_index += 1
        terminal = self._offset == 3
        return self._observation(), 0.0, terminal, False, info


def _evaluate(monkeypatch: pytest.MonkeyPatch, environment: _Environment):
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    return rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=np.array([[0.10], [0.10], [0.0]], dtype=np.float32),
    )


def test_hard_risk_violation_uses_final_risk_projection_not_post_step_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = _evaluate(
        monkeypatch,
        _Environment(risk_weights=0.1002, realized_weights=(0.05, 0.10, 0.0)),
    )
    assert invalid.collapse_evidence.hard_risk_violation is True

    valid_with_market_drift = _evaluate(
        monkeypatch,
        _Environment(risk_weights=0.10, realized_weights=(0.1002, 0.10, 0.0)),
    )
    assert valid_with_market_drift.collapse_evidence.hard_risk_violation is False


def test_step_trace_preserves_submitted_and_delayed_execution_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate(monkeypatch, _Environment())
    trace = result.step_trace
    assert trace is not None

    np.testing.assert_allclose(trace.submitted_targets[:, 0], (0.10, 0.10, 0.10))
    np.testing.assert_allclose(trace.execution_intent_targets[:, 0], (0.0, 0.10, 0.10))
    assert not np.array_equal(trace.submitted_targets, trace.execution_intent_targets)


def test_step_trace_explains_nonflat_to_flat_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate(
        monkeypatch,
        _Environment(
            realized_weights=(0.10, 0.10, 0.0),
            risk_reasons=((), (), ("emergency_flatten",)),
        ),
    )
    trace = result.step_trace
    assert trace is not None

    assert trace.transition_classes[-1] == "exit"
    assert trace.flatten_initiators[-1] == "risk:emergency_flatten"


def test_liquidation_has_explicit_flatten_initiator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _evaluate(
        monkeypatch,
        _Environment(
            realized_weights=(0.10, 0.10, 0.0),
            liquidation_terminal=(False, False, True),
        ),
    )
    trace = result.step_trace
    assert trace is not None
    assert trace.flatten_initiators[-1] == "liquidation"
