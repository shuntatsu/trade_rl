"""Synthetic-only regression action head: input and artifact contract tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.strategies.decision_value import (
    LinearDecisionValueModel,
    LinearDecisionValuePolicy,
    load_linear_decision_value,
    serialize_linear_decision_value,
)
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)


def _snapshot(*, signal: float = 0.0) -> PortfolioDecisionSnapshot:
    account = PortfolioAccountSnapshot(
        cash=1_000.0,
        portfolio_value=1_000.0,
        peak_value=1_000.0,
        max_drawdown=0.0,
        exact_quantities=("0", "0"),
        quantities=(0.0, 0.0),
        weights=(0.0, 0.0),
        mark_prices=(100.0, 200.0),
        contract_multipliers=(1.0, 1.0),
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        margin_used=0.0,
        maintenance_margin=0.0,
        maintenance_requirement=0.0,
        margin_deficit=0.0,
        insolvent=False,
        termination_reason=None,
    )
    return PortfolioDecisionSnapshot(
        index=1,
        timestamp_ns=1_771_000_000_000_000_000,
        agent_stop_index=6,
        remaining_decisions=5,
        symbols=("AAA", "BBB"),
        feature_names=("signal",),
        global_feature_names=("regime",),
        features=((signal,), (0.0,)),
        feature_available=((True,), (True,)),
        feature_staleness=((0.0,), (0.0,)),
        global_features=(0.0,),
        global_feature_available=(True,),
        account=account,
        active_orders=(),
        risk=PortfolioRiskLimits(
            max_gross=0.5,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.20,
            drawdown_stop=0.20,
            emergency_turnover_override=True,
            fail_closed_tolerance=1e-10,
        ),
    )


def _model() -> LinearDecisionValueModel:
    # Features are intercept, AAA signal, BBB signal, global regime, AAA weight,
    # BBB weight. The HOLD candidate has a structural zero score.
    return LinearDecisionValueModel(
        schema_version="synthetic_linear_decision_value_v1",
        source_scope="synthetic_only",
        symbols=("AAA", "BBB"),
        feature_names=("signal",),
        global_feature_names=("regime",),
        feature_config_digest="0" * 64,
        normalization_digest="0" * 64,
        execution_policy_digest="a" * 64,
        reference_policy_digest="b" * 64,
        target_abs_weights=(0.4, 0.4),
        candidate_actions=((ACTION_HOLD, ACTION_HOLD), (ACTION_LONG, ACTION_HOLD),
                           (ACTION_SHORT, ACTION_HOLD)),
        fit_cutoff_ns=1_770_000_000_000_000_000,
        evaluation_start_ns=1_770_100_000_000_000_000,
        ridge_penalty=1e-4,
        training_rows=18,
        overlap_groups=1,
        coefficients=((-0.001, 0.010, 0.0, 0.0, 0.0, 0.0),
                      (-0.001, -0.010, 0.0, 0.0, 0.0, 0.0)),
    )


def test_action_head_prefers_hold_without_positive_increment_and_keeps_quantity_hold() -> None:
    policy = LinearDecisionValuePolicy(_model())
    for signal, expected in ((0.0, ACTION_HOLD), (0.7, ACTION_LONG),
                             (-0.7, ACTION_SHORT)):
        snapshot = _snapshot(signal=signal)
        assert policy.decide_action(snapshot) == (expected, ACTION_HOLD)
        proposal = policy.decide_portfolio(snapshot)
        if expected == ACTION_HOLD:
            assert proposal.hold_current_quantity_symbols == (0, 1)
            assert proposal.target_weights == snapshot.account.weights
        else:
            assert proposal.target_weights[0] == pytest.approx(
                0.4 if expected == ACTION_LONG else -0.4
            )
            assert proposal.hold_current_quantity_symbols == (1,)
    assert policy.value_for_action(_snapshot(), (ACTION_HOLD, ACTION_HOLD)) == 0.0


def test_model_rejects_unsupported_input_instead_of_faking_safe_profit() -> None:
    policy = LinearDecisionValuePolicy(_model())
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="feature"):
        policy.decide_action(replace(snapshot, feature_available=((False,), (True,))))
    with pytest.raises(ValueError, match="feature"):
        policy.decide_action(replace(snapshot, global_feature_available=(False,)))
    with pytest.raises(ValueError, match="roster|symbols"):
        policy.decide_action(replace(snapshot, symbols=("BBB", "AAA")))
    with pytest.raises(ValueError, match="evaluation"):
        policy.decide_action(replace(snapshot, timestamp_ns=1_769_000_000_000_000_000))
    with pytest.raises(ValueError, match="pending"):
        policy.decide_action(
            replace(snapshot, active_orders=(_pending_order(),))
        )
    with pytest.raises(ValueError, match="candidate"):
        policy.value_for_action(snapshot, (ACTION_LONG, ACTION_SHORT))


def _pending_order():
    from trade_rl.strategies.portfolio import PortfolioPendingOrderSnapshot

    return PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type="market",
        time_in_force="gtc",
        requested_quantity=1.0,
        remaining_quantity=1.0,
        status="submitted",
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
        submission_reference_price=100.0,
    )


def test_serialization_preserves_action_and_fails_closed_on_tamper(
    tmp_path: Path,
) -> None:
    model = _model()
    payload, digest = serialize_linear_decision_value(model)
    path = tmp_path / "synthetic-linear-value.json"
    path.write_text(payload, encoding="utf-8")
    restored = load_linear_decision_value(path.read_text(encoding="utf-8"),
                                          expected_sha256=digest)
    assert restored == model
    for signal in (-1.0, 0.0, 0.8):
        assert LinearDecisionValuePolicy(restored).decide_action(_snapshot(signal=signal)) == (
            LinearDecisionValuePolicy(model).decide_action(_snapshot(signal=signal))
        )
    with pytest.raises(ValueError, match="digest"):
        load_linear_decision_value(payload.replace("-0.001", "0.9000"),
                                   expected_sha256=digest)
    with pytest.raises(ValueError, match="digest"):
        load_linear_decision_value(payload, expected_sha256="c" * 64)
    with pytest.raises(ValueError, match="canonical"):
        load_linear_decision_value(payload + "\n", expected_sha256=digest)


def test_invalid_model_schema_or_coefficients_are_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="finite"):
        replace(model, coefficients=((float("nan"),) * 6,) * 2)
    with pytest.raises(ValueError, match="candidate"):
        replace(model, candidate_actions=((ACTION_HOLD, ACTION_HOLD),
                                          (ACTION_LONG, ACTION_HOLD),
                                          (ACTION_LONG, ACTION_HOLD)))
    with pytest.raises(ValueError, match="dimension|coefficients"):
        replace(model, coefficients=((1.0,), (1.0,)))
    with pytest.raises(ValueError, match="synthetic"):
        replace(model, source_scope="real_market")
    with pytest.raises(ValueError, match="cutoff|evaluation"):
        replace(model, evaluation_start_ns=model.fit_cutoff_ns)
