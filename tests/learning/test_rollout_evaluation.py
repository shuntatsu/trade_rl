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
        return np.ones(2, dtype=np.float64)


class _Environment:
    dataset = _Dataset()

    def __init__(
        self,
        *,
        omit_rejected_event: bool = False,
        executed_turnover: float = 0.2,
        gross_returns: tuple[float, ...] = (0.0, 0.0, 0.0),
        net_returns: tuple[float, ...] = (0.0, 0.0, 0.0),
        costs: tuple[float, ...] = (0.0, 0.0, 0.0),
        post_weights: tuple[tuple[float, float], ...] | None = None,
        risk_weights: tuple[tuple[float, float], ...] | None = None,
        risk_scales: tuple[float, ...] = (1.0, 1.0, 1.0),
        drawdowns: tuple[float, ...] = (0.0, 0.0, 0.0),
        risk_config: PreTradeRiskConfig | None = None,
    ) -> None:
        self.current_index = 0
        self._offset = 0
        self._weights = np.zeros(2, dtype=np.float32)
        self._omit_rejected_event = omit_rejected_event
        self._executed_turnover = executed_turnover
        self._gross_returns = gross_returns
        self._net_returns = net_returns
        self._costs = costs
        self._post_weights = post_weights
        self._risk_weights = risk_weights
        self._risk_scales = risk_scales
        self._drawdowns = drawdowns
        self.pre_trade_risk = PreTradeRisk(
            risk_config
            or PreTradeRiskConfig(
                max_gross=1.0,
                max_abs_weight=0.5,
                max_turnover=2.0,
            )
        )
        self.hybrid = SimpleNamespace(
            max_drawdown=0.0,
            quantities=np.zeros(2, dtype=np.float64),
            mark_prices=np.ones(2, dtype=np.float64),
        )

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "current_weights": self._weights.copy(),
            "active": np.array([1.0, 0.0], dtype=np.float32),
        }

    def reset(
        self, *, options: dict[str, object]
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        start_index = options["start_idx"]
        assert isinstance(start_index, int)
        self.current_index = start_index
        self._offset = 0
        self._weights[:] = 0.0
        self.hybrid.max_drawdown = 0.0
        return self._observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, object]]:
        requested = 0.0 if self._offset == 0 else self._executed_turnover
        filled = 0.0 if self._offset == 0 else self._executed_turnover
        rejected = 1 if self._offset == 0 else 0
        risk_weights = (
            np.asarray(self._risk_weights[self._offset], dtype=np.float64)
            if self._risk_weights is not None
            else np.asarray(action, dtype=np.float64).copy()
        )
        if self._post_weights is not None:
            self._weights[:] = np.asarray(
                self._post_weights[self._offset], dtype=np.float32
            )
        elif filled > 0.0:
            self._weights[0] = float(action[0])
        self.hybrid.max_drawdown = float(self._drawdowns[self._offset])
        execution = SimpleNamespace(
            requested_turnover=requested,
            filled_turnover=filled,
            rejected_count=rejected,
            order_events=(
                ()
                if rejected == 0 or self._omit_rejected_event
                else (
                    SimpleNamespace(event_type="rejected", reason="minimum_notional"),
                )
            ),
        )
        info: dict[str, object] = {
            "hybrid_execution": execution,
            "hybrid_risk": SimpleNamespace(
                weights=risk_weights,
                risk_scale=float(self._risk_scales[self._offset]),
                reasons=("no_trade_band",) if self._offset == 0 else (),
            ),
            "hybrid_liquidation": None,
            "interval_gross_return": self._gross_returns[self._offset],
            "interval_net_return": self._net_returns[self._offset],
            "interval_cost": self._costs[self._offset],
        }
        self._offset += 1
        self.current_index += 1
        terminal = self._offset == 3
        return self._observation(), 0.0, terminal, False, info


def test_action_path_reports_where_submitted_changes_disappear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment(
        risk_weights=((0.4, 0.0), (0.4, 0.0), (0.0, 0.0)),
    )

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
    assert evidence.execution_rejection_reason_counts == (("minimum_notional", 1),)
    assert evidence.risk_projection_reason_counts == (("no_trade_band", 1),)
    assert evidence.hard_risk_violation is False
    assert evidence.executed_change_count == 2
    assert evidence.constant_submitted_actions is False
    assert evidence.inactive_mask_rate == 0.5


def test_action_path_uses_one_tolerance_for_execution_and_traded_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    tolerance = 1e-6
    environment = _Environment(executed_turnover=5e-7)

    result = rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=np.array(
            [[0.4, 0.0], [0.2, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
        action_change_tolerance=tolerance,
    )

    assert result.collapse_evidence.executed_change_count == 0
    assert result.performance.traded_step_count == 0


def test_action_path_preserves_immutable_step_economics_for_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment(
        gross_returns=(0.01, -0.02, 0.03),
        net_returns=(0.009, -0.021, 0.029),
        costs=(0.001, 0.001, 0.001),
    )

    result = rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=np.array([[0.2, 0.0], [0.2, 0.0], [0.0, 0.0]], dtype=np.float32),
    )

    economics = result.step_economics
    assert economics is not None
    np.testing.assert_allclose(economics.gross_returns, (0.01, -0.02, 0.03))
    np.testing.assert_allclose(economics.net_returns, (0.009, -0.021, 0.029))
    np.testing.assert_allclose(economics.costs, (0.001, 0.001, 0.001))
    assert economics.gross_returns.flags.writeable is False
    assert economics.net_returns.flags.writeable is False
    assert economics.costs.flags.writeable is False


def test_action_path_records_execution_boundary_weights_without_changing_economics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment(
        gross_returns=(0.01, -0.02, 0.03),
        net_returns=(0.009, -0.021, 0.029),
        costs=(0.001, 0.001, 0.001),
        post_weights=((0.09, 0.0), (0.08, 0.0), (0.0, 0.0)),
        risk_weights=((0.10, 0.0), (0.09, 0.0), (0.0, 0.0)),
        risk_scales=(1.0, 0.8, 0.6),
    )
    actions = np.asarray([[0.10, 0.0], [0.09, 0.0], [0.0, 0.0]], dtype=np.float32)

    result = rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=actions,
    )

    trace = result.execution_trace
    assert trace is not None
    np.testing.assert_allclose(trace.pre_action_weights[:, 0], (0.0, 0.09, 0.08))
    np.testing.assert_allclose(trace.risk_constrained_weights[:, 0], (0.10, 0.09, 0.0))
    np.testing.assert_allclose(trace.post_step_weights[:, 0], (0.09, 0.08, 0.0))
    np.testing.assert_allclose(trace.applied_risk_scales, (1.0, 0.8, 0.6))
    np.testing.assert_allclose(result.actions, actions)
    assert result.step_economics is not None
    np.testing.assert_allclose(result.step_economics.gross_returns, (0.01, -0.02, 0.03))
    np.testing.assert_allclose(
        result.step_economics.net_returns, (0.009, -0.021, 0.029)
    )
    np.testing.assert_allclose(result.step_economics.costs, (0.001, 0.001, 0.001))


def test_action_path_distinguishes_intent_realized_state_follow_and_reassertion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment(
        post_weights=((0.09, 0.0), (0.08, 0.0), (0.09, 0.0)),
    )

    result = rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=np.asarray([[0.10, 0.0], [0.09, 0.0], [0.09, 0.0]], dtype=np.float32),
    )

    trace = result.execution_trace
    assert trace is not None
    assert trace.strategy_intent_changes.tolist() == [True, False, False]
    assert trace.realized_state_follows.tolist() == [False, True, False]
    assert trace.rebalance_reassertions.tolist() == [False, False, True]


def test_generic_collapse_evidence_schema_does_not_gain_v10_change_class_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    result = rollout_evaluation.evaluate_action_path(
        _Environment(post_weights=((0.09, 0.0), (0.08, 0.0), (0.09, 0.0))),
        evaluation_range=(0, 4),
        actions=np.asarray([[0.10, 0.0], [0.09, 0.0], [0.09, 0.0]], dtype=np.float32),
    )

    payload = result.collapse_evidence.to_dict()
    assert "strategy_intent_change_count" not in payload
    assert "realized_state_follow_count" not in payload
    assert "passive_weight_drift_count" not in payload
    assert "rebalance_reassertion_count" not in payload


def _risk_config_for_projection_test() -> PreTradeRiskConfig:
    return PreTradeRiskConfig(
        max_gross=1.0,
        max_abs_weight=0.10,
        max_turnover=2.0,
        drawdown_start=0.10,
        drawdown_stop=0.20,
    )


def test_action_path_does_not_treat_post_step_price_drift_as_hard_risk_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment(
        post_weights=((0.06, 0.0), (0.04, 0.0), (0.0, 0.0)),
        risk_weights=((0.05, 0.0), (0.04, 0.0), (0.0, 0.0)),
        risk_scales=(0.5, 0.5, 0.0),
        drawdowns=(0.15, 0.15, 0.20),
        risk_config=_risk_config_for_projection_test(),
    )

    result = rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=np.asarray([[0.05, 0.0], [0.04, 0.0], [0.0, 0.0]], dtype=np.float32),
    )

    trace = result.execution_trace
    assert trace is not None
    np.testing.assert_allclose(trace.applied_risk_scales, (0.5, 0.5, 0.0))
    assert trace.hard_risk_violations.tolist() == [False, False, False]
    assert result.collapse_evidence.hard_risk_violation is False


def test_action_path_detects_hard_risk_projection_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)
    environment = _Environment(
        post_weights=((0.04, 0.0), (0.04, 0.0), (0.0, 0.0)),
        risk_weights=((0.06, 0.0), (0.04, 0.0), (0.0, 0.0)),
        risk_scales=(0.5, 0.5, 0.0),
        drawdowns=(0.15, 0.15, 0.20),
        risk_config=_risk_config_for_projection_test(),
    )

    result = rollout_evaluation.evaluate_action_path(
        environment,
        evaluation_range=(0, 4),
        actions=np.asarray([[0.06, 0.0], [0.04, 0.0], [0.0, 0.0]], dtype=np.float32),
    )

    trace = result.execution_trace
    assert trace is not None
    assert trace.hard_risk_violations.tolist() == [True, False, False]
    assert result.collapse_evidence.hard_risk_violation is True


def test_action_path_rejects_unreconciled_execution_rejection_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_evaluation, "ClosedTradeTracker", _Trades)

    with pytest.raises(ValueError, match="rejected_count"):
        rollout_evaluation.evaluate_action_path(
            _Environment(omit_rejected_event=True),
            evaluation_range=(0, 4),
            actions=np.zeros((3, 2), dtype=np.float32),
        )


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
