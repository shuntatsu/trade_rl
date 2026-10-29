"""Synthetic-only regression action head: input and artifact contract tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trade_rl.strategies.decision_value import (
    ConservativeLinearDecisionValueModel,
    ConservativeLinearDecisionValuePolicy,
    LinearDecisionValueModel,
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
    compute_pending_order_weights,
    encode_decision_value_state_v2,
    load_conservative_linear_decision_value,
    load_linear_decision_value,
    serialize_conservative_linear_decision_value,
    serialize_linear_decision_value,
)
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioPendingOrderSnapshot,
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
        candidate_actions=(
            (ACTION_HOLD, ACTION_HOLD),
            (ACTION_LONG, ACTION_HOLD),
            (ACTION_SHORT, ACTION_HOLD),
        ),
        fit_cutoff_ns=1_770_000_000_000_000_000,
        evaluation_start_ns=1_770_100_000_000_000_000,
        ridge_penalty=1e-4,
        training_rows=18,
        overlap_groups=1,
        coefficients=(
            (-0.001, 0.010, 0.0, 0.0, 0.0, 0.0),
            (-0.001, -0.010, 0.0, 0.0, 0.0, 0.0),
        ),
    )


def _context(model: LinearDecisionValueModel) -> LinearDecisionValueRuntimeContext:
    return LinearDecisionValueRuntimeContext(
        source_scope="synthetic_only",
        feature_config_digest=model.feature_config_digest,
        normalization_digest=model.normalization_digest,
        execution_policy_digest=model.execution_policy_digest,
        reference_policy_digest=model.reference_policy_digest,
    )


def _policy(model: LinearDecisionValueModel) -> LinearDecisionValuePolicy:
    return LinearDecisionValuePolicy(model, runtime_context=_context(model))


def test_runtime_binding_rejects_different_economics_or_source_scope() -> None:
    model = _model()
    context = _context(model)
    with pytest.raises(ValueError, match="execution_policy_digest"):
        LinearDecisionValuePolicy(
            model,
            runtime_context=replace(context, execution_policy_digest="f" * 64),
        )
    with pytest.raises(ValueError, match="synthetic"):
        replace(context, source_scope="real_market")
    with pytest.raises(TypeError):
        LinearDecisionValuePolicy(model)


def test_action_head_prefers_hold_without_positive_increment_and_keeps_quantity_hold() -> (
    None
):
    policy = _policy(_model())
    for signal, expected in (
        (0.0, ACTION_HOLD),
        (0.7, ACTION_LONG),
        (-0.7, ACTION_SHORT),
    ):
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
    policy = _policy(_model())
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
        policy.decide_action(replace(snapshot, active_orders=(_pending_order(),)))
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
    restored = load_linear_decision_value(
        path.read_text(encoding="utf-8"), expected_sha256=digest
    )
    assert restored == model
    for signal in (-1.0, 0.0, 0.8):
        assert _policy(restored).decide_action(_snapshot(signal=signal)) == (
            _policy(model).decide_action(_snapshot(signal=signal))
        )
    with pytest.raises(ValueError, match="digest"):
        load_linear_decision_value(
            payload.replace("-0.001", "0.9"), expected_sha256=digest
        )
    with pytest.raises(ValueError, match="digest"):
        load_linear_decision_value(payload, expected_sha256="c" * 64)
    with pytest.raises(ValueError, match="canonical"):
        load_linear_decision_value(payload + "\n", expected_sha256=digest)


def test_invalid_model_schema_or_coefficients_are_rejected() -> None:
    model = _model()
    with pytest.raises(ValueError, match="finite"):
        replace(model, coefficients=((float("nan"),) * 6,) * 2)
    with pytest.raises(ValueError, match="candidate"):
        replace(
            model,
            candidate_actions=(
                (ACTION_HOLD, ACTION_HOLD),
                (ACTION_LONG, ACTION_HOLD),
                (ACTION_LONG, ACTION_HOLD),
            ),
        )
    with pytest.raises(ValueError, match="dimension|coefficients"):
        replace(model, coefficients=((1.0,), (1.0,)))
    with pytest.raises(ValueError, match="synthetic"):
        replace(model, source_scope="real_market")
    with pytest.raises(ValueError, match="cutoff|evaluation"):
        replace(model, evaluation_start_ns=model.fit_cutoff_ns)


def _conservative_model(
    *,
    uncertainty_penalty_weight: float = 1.0,
    one_way_switch_cost: float = 0.0,
    cov_diag_val: float = 0.01,
) -> ConservativeLinearDecisionValueModel:
    d = 8
    cov = tuple(
        tuple(cov_diag_val if i == j else 0.0 for j in range(d)) for i in range(d)
    )
    return ConservativeLinearDecisionValueModel(
        schema_version="conservative_linear_decision_value_v2",
        source_scope="synthetic_only",
        symbols=("AAA", "BBB"),
        feature_names=("signal",),
        global_feature_names=("regime",),
        feature_config_digest="0" * 64,
        normalization_digest="0" * 64,
        execution_policy_digest="a" * 64,
        reference_policy_digest="b" * 64,
        target_abs_weights=(0.4, 0.4),
        candidate_actions=(
            (ACTION_HOLD, ACTION_HOLD),
            (ACTION_LONG, ACTION_HOLD),
            (ACTION_SHORT, ACTION_HOLD),
        ),
        fit_cutoff_ns=1_770_000_000_000_000_000,
        evaluation_start_ns=1_770_100_000_000_000_000,
        ridge_penalty=1e-4,
        training_rows=18,
        overlap_groups=1,
        coefficients=(
            (-0.001, 0.010, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-0.001, -0.010, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        covariance_matrices=(cov, cov),
        uncertainty_penalty_weight=uncertainty_penalty_weight,
        one_way_switch_cost=one_way_switch_cost,
    )


def test_compute_pending_order_weights() -> None:
    snap = _snapshot()
    # Empty active orders
    assert compute_pending_order_weights(snap) == (0.0, 0.0)

    # AAA mark=100, multiplier=1, portfolio_value=1000.
    # Order 1: symbol 0, remaining=2.0 -> value = 2.0 * 100 * 1 = 200 -> weight = 200/1000 = 0.20
    # Order 2: symbol 1, remaining=-1.0 -> value = -1.0 * 200 * 1 = -200 -> weight = -200/1000 = -0.20
    ord0 = PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type="market",
        time_in_force="gtc",
        requested_quantity=2.0,
        remaining_quantity=2.0,
        status="submitted",
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
        submission_reference_price=100.0,
    )
    ord1 = PortfolioPendingOrderSnapshot(
        symbol_index=1,
        order_type="market",
        time_in_force="gtc",
        requested_quantity=-1.0,
        remaining_quantity=-1.0,
        status="submitted",
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
        submission_reference_price=200.0,
    )
    snap_orders = replace(snap, active_orders=(ord0, ord1))
    pw = compute_pending_order_weights(snap_orders)
    assert pytest.approx(pw[0]) == 0.20
    assert pytest.approx(pw[1]) == -0.20

    # Test accumulation: add second order for symbol 0 with remaining 1.0 -> total 300 / 1000 = 0.30
    ord0_extra = replace(ord0, remaining_quantity=1.0)
    snap_accum = replace(snap, active_orders=(ord0, ord0_extra, ord1))
    pw_accum = compute_pending_order_weights(snap_accum)
    assert pytest.approx(pw_accum[0]) == 0.30
    assert pytest.approx(pw_accum[1]) == -0.20

    # Zero or negative portfolio value
    with pytest.raises(ValueError, match="portfolio_value"):
        compute_pending_order_weights(
            replace(
                snap_orders, account=replace(snap_orders.account, portfolio_value=0.0)
            )
        )

    # Symbol index out of range
    bad_ord = replace(ord0, symbol_index=5)
    with pytest.raises(ValueError, match="symbol"):
        compute_pending_order_weights(replace(snap, active_orders=(bad_ord,)))


def test_encode_decision_value_state_v2() -> None:
    snap = _snapshot(signal=0.5)
    ord0 = PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type="market",
        time_in_force="gtc",
        requested_quantity=2.0,
        remaining_quantity=2.0,
        status="submitted",
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
        submission_reference_price=100.0,
    )
    snap_orders = replace(snap, active_orders=(ord0,))
    encoded = encode_decision_value_state_v2(
        snap_orders,
        symbols=("AAA", "BBB"),
        feature_names=("signal",),
        global_feature_names=("regime",),
    )
    assert encoded.shape == (8,)
    # Intercept = 1.0
    assert encoded[0] == 1.0
    # AAA feature = 0.5, BBB feature = 0.0
    assert encoded[1] == 0.5
    assert encoded[2] == 0.0
    # Regime = 0.0
    assert encoded[3] == 0.0
    # Weights AAA = 0.0, BBB = 0.0
    assert encoded[4] == 0.0
    assert encoded[5] == 0.0
    # Pending weights AAA = 0.20, BBB = 0.0
    assert pytest.approx(encoded[6]) == 0.20
    assert encoded[7] == 0.0


def test_conservative_linear_decision_value_model_validation() -> None:
    model = _conservative_model()
    assert model.schema_version == "conservative_linear_decision_value_v2"

    # Schema version
    with pytest.raises(ValueError, match="schema"):
        replace(model, schema_version="synthetic_linear_decision_value_v1")

    # Negative uncertainty penalty or switch cost
    with pytest.raises(ValueError, match="uncertainty_penalty_weight"):
        replace(model, uncertainty_penalty_weight=-0.1)
    with pytest.raises(ValueError, match="one_way_switch_cost"):
        replace(model, one_way_switch_cost=-0.01)

    # Covariance dimension mismatch
    with pytest.raises(ValueError, match="covariance"):
        replace(model, covariance_matrices=(model.covariance_matrices[0],))
    bad_shape_cov = tuple(tuple(0.0 for _ in range(7)) for _ in range(8))
    with pytest.raises(ValueError, match="covariance"):
        replace(model, covariance_matrices=(bad_shape_cov, bad_shape_cov))

    # Asymmetric covariance
    asym_cov = [list(row) for row in model.covariance_matrices[0]]
    asym_cov[0][1] = 1.0
    asym_cov[1][0] = 2.0
    asym_tuple = tuple(tuple(r) for r in asym_cov)
    with pytest.raises(ValueError, match="symmetric"):
        replace(model, covariance_matrices=(asym_tuple, model.covariance_matrices[1]))

    # Negative diagonal in covariance
    neg_diag = [list(row) for row in model.covariance_matrices[0]]
    neg_diag[0][0] = -0.5
    neg_diag_tuple = tuple(tuple(r) for r in neg_diag)
    with pytest.raises(ValueError, match="non-negative"):
        replace(
            model, covariance_matrices=(neg_diag_tuple, model.covariance_matrices[1])
        )


def test_conservative_policy_uncertainty_and_hold_suppression() -> None:
    model_low_unc = _conservative_model(
        uncertainty_penalty_weight=0.0, cov_diag_val=0.0
    )
    ctx = LinearDecisionValueRuntimeContext(
        source_scope="synthetic_only",
        feature_config_digest=model_low_unc.feature_config_digest,
        normalization_digest=model_low_unc.normalization_digest,
        execution_policy_digest=model_low_unc.execution_policy_digest,
        reference_policy_digest=model_low_unc.reference_policy_digest,
    )
    pol_low = ConservativeLinearDecisionValuePolicy(model_low_unc, runtime_context=ctx)
    snap = _snapshot(signal=0.7)
    # With zero uncertainty penalty, signal 0.7 gives positive value for LONG
    assert pol_low.decide_action(snap) == (ACTION_LONG, ACTION_HOLD)

    # Now with high uncertainty penalty and positive variance, LCB is negative -> falls back to HOLD
    model_high_unc = _conservative_model(
        uncertainty_penalty_weight=10.0, cov_diag_val=1.0
    )
    pol_high = ConservativeLinearDecisionValuePolicy(
        model_high_unc, runtime_context=ctx
    )
    assert pol_high.decide_action(snap) == (ACTION_HOLD, ACTION_HOLD)

    # Test in-flight duplicate order suppression:
    # Target weight for LONG AAA is 0.4.
    # Suppose current account weight is 0.0, but there is an active order for 4.0 shares @ 100 -> pending weight = 0.40.
    # Effective weight = 0.0 + 0.40 = 0.40, which matches target 0.40!
    # Therefore decide_portfolio must NOT issue another BUY:
    # it must add symbol 0 to hold_current_quantity_symbols and set target_weights[0] = account.weights[0] (0.0).
    ord_inflight = PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type="market",
        time_in_force="gtc",
        requested_quantity=4.0,
        remaining_quantity=4.0,
        status="submitted",
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
        submission_reference_price=100.0,
    )
    snap_inflight = replace(snap, active_orders=(ord_inflight,))
    proposal = pol_low.decide_portfolio(snap_inflight)
    assert 0 in proposal.hold_current_quantity_symbols
    assert proposal.target_weights[0] == 0.0


def test_conservative_serialization_and_tamper(tmp_path: Path) -> None:
    model = _conservative_model()
    payload, digest = serialize_conservative_linear_decision_value(model)
    path = tmp_path / "conservative-linear-value.json"
    path.write_text(payload, encoding="utf-8")
    restored = load_conservative_linear_decision_value(
        path.read_text(encoding="utf-8"), expected_sha256=digest
    )
    assert restored == model

    ctx = LinearDecisionValueRuntimeContext(
        source_scope="synthetic_only",
        feature_config_digest=model.feature_config_digest,
        normalization_digest=model.normalization_digest,
        execution_policy_digest=model.execution_policy_digest,
        reference_policy_digest=model.reference_policy_digest,
    )
    pol_orig = ConservativeLinearDecisionValuePolicy(model, runtime_context=ctx)
    pol_rest = ConservativeLinearDecisionValuePolicy(restored, runtime_context=ctx)
    snap = _snapshot(signal=0.7)
    assert pol_orig.decide_action(snap) == pol_rest.decide_action(snap)

    with pytest.raises(ValueError, match="digest"):
        load_conservative_linear_decision_value(
            payload.replace("0.01", "0.99"), expected_sha256=digest
        )
    with pytest.raises(ValueError, match="digest"):
        load_conservative_linear_decision_value(payload, expected_sha256="c" * 64)
    with pytest.raises(ValueError, match="canonical"):
        load_conservative_linear_decision_value(payload + "\n", expected_sha256=digest)
