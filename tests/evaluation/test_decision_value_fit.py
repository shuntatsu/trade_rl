"""One bounded synthetic model fit, frozen forward decisions and native E2E."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_fit import fit_synthetic_linear_decision_value
from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label,
    build_synthetic_paired_label_batch,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.decision_value import (
    LinearDecisionValueModel,
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    portfolio_action_to_proposal,
)

_HOLD = (ACTION_HOLD, ACTION_HOLD)
_LONG = (ACTION_LONG, ACTION_HOLD)
_SHORT = (ACTION_SHORT, ACTION_HOLD)
_CANDIDATES = (_HOLD, _LONG, _SHORT)
_COST = replace(ExecutionCostConfig.zero(), fee_rate=0.001, spread_rate=0.0002)
_RISK = PreTradeRisk(
    PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.20,
        drawdown_stop=0.20,
    )
)
_TRAIN_FIT = np.datetime64("2026-02-01T00:00:00", "ns")
_EVALUATE = np.datetime64("2026-03-01T00:00:00", "ns")


def _market(*, signal: float, move: float, day: str, tag: str) -> MarketDataset:
    # One observable synthetic signal, followed by a deliberately known trend.
    marks = np.array(
        [
            100,
            100,
            100,
            100 * (1 + move / 3),
            100 * (1 + 2 * move / 3),
            100 * (1 + move),
            100 * (1 + move),
            100 * (1 + move),
        ],
        dtype=np.float64,
    )
    closes = np.column_stack((marks, np.full(8, 200.0)))
    opens = np.vstack((closes[0], closes[:-1]))
    features = np.zeros((8, 2, 1), dtype=np.float32)
    features[:, 0, 0] = signal
    digest = hashlib.sha256(
        f"issue1025-td3/synthetic/{tag}/{day}/{signal}/{move}".encode()
    ).hexdigest()
    return MarketDataset(
        dataset_id=digest,
        symbols=("AAA", "BBB"),
        timestamps=np.datetime64(f"{day}T00:00:00", "ns")
        + np.arange(8) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((8, 1), dtype=np.float32),
        open=opens,
        high=np.maximum(opens, closes),
        low=np.minimum(opens, closes),
        close=closes,
        volume=np.full_like(closes, 1_000_000.0),
        funding_rate=np.zeros_like(closes),
        tradable=np.ones_like(closes, dtype=np.bool_),
        feature_available=np.ones((8, 2, 1), dtype=np.bool_),
        feature_names=("observed_signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


class _Hold:
    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            snapshot.account.weights, hold_current_quantity_symbols=(0, 1)
        )


class _AlwaysLong:
    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return portfolio_action_to_proposal(_LONG, snapshot, (0.4, 0.4))


def _policy(
    model: LinearDecisionValueModel, market: MarketDataset
) -> LinearDecisionValuePolicy:
    context = LinearDecisionValueRuntimeContext(
        source_scope="synthetic_only",
        feature_config_digest=market.feature_config_digest,
        normalization_digest=market.normalization_digest,
        execution_policy_digest=model.execution_policy_digest,
        reference_policy_digest=model.reference_policy_digest,
    )
    return LinearDecisionValuePolicy(model, runtime_context=context)


def _native(market: MarketDataset, policy: object) -> float:
    result = run_shared_portfolio_replay(
        market,
        policy,
        start_index=0,
        stop_index=7,
        initial_capital=1_000.0,
        execution_cost=_COST,
        risk=_RISK,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert result.ledger_evidence is not None
    assert result.book.exact_quantities == (0, 0)
    assert result.book.termination_reason is None
    return result.book.portfolio_value


def _fit_batch():
    samples = []
    for magnitude in (0.018, 0.020, 0.022):
        for signal in (-1.0, 0.0, 1.0):
            market = _market(
                signal=signal,
                move=signal * magnitude,
                day="2026-01-01",
                tag=f"train-{magnitude}-{signal}",
            )
            for action in (_LONG, _SHORT):
                pair = run_paired_portfolio_action(
                    market,
                    reference_policy_factory=_Hold,
                    start_index=0,
                    stop_index=7,
                    intervention_index=1,
                    baseline_action=_HOLD,
                    candidate_action=action,
                    target_abs_weights=(0.4, 0.4),
                    initial_capital=1_000.0,
                    execution_cost=_COST,
                    risk=_RISK,
                )
                ledger = pair.baseline_result.ledger_evidence
                assert ledger is not None
                execution_digest = ledger.to_mapping()["execution_policy_digest"]
                assert isinstance(execution_digest, str)
                samples.append(
                    build_synthetic_paired_label(
                        market,
                        pair,
                        baseline_action=_HOLD,
                        candidate_action=action,
                        target_abs_weights=(0.4, 0.4),
                        reference_policy_digest="a" * 64,
                        expected_execution_policy_digest=execution_digest,
                        observation_record_available_at=market.timestamps[1],
                        settlement_record_available_at=market.timestamps[7]
                        + np.timedelta64(30, "m"),
                        fit_cutoff=_TRAIN_FIT,
                    )
                )
    return build_synthetic_paired_label_batch(
        tuple(samples),
        fit_cutoff=_TRAIN_FIT,
        evaluation_start=_EVALUATE,
        expected_reference_policy_digest="a" * 64,
    )


def test_real_small_fit_replays_frozen_policy_on_separate_synthetic_periods() -> None:
    batch = _fit_batch()
    assert len(batch.samples) == 18
    assert (
        batch.group_ids == (0,) * 18
    )  # correlated windows, NOT 18 independent markets
    model = fit_synthetic_linear_decision_value(batch, candidate_actions=_CANDIDATES)
    assert model.training_rows == 18
    assert model.overlap_groups == 1

    learned_values = []
    cash_values = []
    always_long_values = []
    for signal, move in ((0.0, 0.0), (0.8, 0.018), (-0.8, -0.018)):
        market = _market(signal=signal, move=move, day="2026-03-03", tag="unused-toy")
        policy = _policy(model, market)
        learned = _native(market, policy)
        cash = _native(market, _Hold())
        always_long = _native(market, _AlwaysLong())
        learned_values.append(learned)
        cash_values.append(cash)
        always_long_values.append(always_long)
        if signal == 0.0:
            assert learned == pytest.approx(cash)
        else:
            assert learned > cash
    assert sum(learned_values) > sum(cash_values)
    assert sum(learned_values) > sum(always_long_values)

    # A change to the *future* payoff cannot alter the already frozen action.
    bad_tail = _market(signal=0.8, move=-0.018, day="2026-03-03", tag="bad-tail")
    good_tail = _market(signal=0.8, move=0.018, day="2026-03-03", tag="good-tail")

    class _Capture:
        def __init__(self, market: MarketDataset) -> None:
            self.actor = _policy(model, market)
            self.first: tuple[int, ...] | None = None

        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            if self.first is None:
                self.first = self.actor.decide_action(snapshot)
            return self.actor.decide_portfolio(snapshot)

    first, second = _Capture(good_tail), _Capture(bad_tail)
    good = _native(good_tail, first)
    bad = _native(bad_tail, second)
    assert first.first == second.first == _LONG
    assert bad < good


def test_fit_fail_closed_when_maturity_or_action_roster_is_forged() -> None:
    batch = _fit_batch()
    with pytest.raises(ValueError, match="matur|cutoff"):
        fit_synthetic_linear_decision_value(
            replace(
                batch,
                samples=(
                    replace(
                        batch.samples[0], label_first_available_ns=batch.fit_cutoff_ns
                    ),
                )
                + batch.samples[1:],
            ),
            candidate_actions=_CANDIDATES,
        )
    with pytest.raises(ValueError, match="candidate|coverage"):
        fit_synthetic_linear_decision_value(batch, candidate_actions=(_HOLD, _LONG))
    with pytest.raises(ValueError, match="reference|digest"):
        fit_synthetic_linear_decision_value(
            replace(
                batch,
                samples=(replace(batch.samples[0], reference_policy_digest="f" * 64),)
                + batch.samples[1:],
            ),
            candidate_actions=_CANDIDATES,
        )


def test_fit_standardized_ridge_and_affine_folding() -> None:
    from trade_rl.evaluation.decision_value_fit import _fit_standardized_ridge

    n = 100
    p = 6
    rng = np.random.default_rng(20261029)
    x = rng.normal(size=(n, p))
    x[:, 0] = 1.0  # bias
    x[:, 3] = 1.0  # constant feature
    w_true = np.array([0.5, 1.2, -0.8, 0.0, 0.4, -0.3])
    y = x @ w_true + rng.normal(scale=0.01, size=n)

    intervals = [(i * 6, i * 6 + 24) for i in range(n)]
    rows = [x[i] for i in range(n)]
    labels = [float(v) for v in y]

    coefs, cov = _fit_standardized_ridge(
        rows, labels, penalty=1e-4, intervals=intervals, compute_covariance=True
    )
    assert len(coefs) == p
    assert cov is not None
    assert len(cov) == p
    assert len(cov[0]) == p

    # Constant feature 3 must have zero weight
    assert coefs[3] == pytest.approx(0.0)

    # For any new test point with constant feature = 1.0, prediction and variance must match standardized computation
    x_test = np.array([1.0, 0.5, -0.2, 1.0, 0.8, -0.1])
    pred = float(x_test @ np.asarray(coefs))
    quad = float(x_test @ np.asarray(cov) @ x_test)
    assert quad > 0.0
    assert abs(pred - float(x_test @ w_true)) < 0.05


def test_standardized_ridge_covariance_calibration_and_collinearity() -> None:
    from trade_rl.evaluation.decision_value_fit import _fit_standardized_ridge

    rng = np.random.default_rng(42)
    n = 80
    # Create collinear features: feature 2 is collinear with feature 1 (corr ~ 1 - 1e-8)
    base_feat = rng.normal(size=n)
    collinear_feat = base_feat + 1e-8 * rng.normal(size=n)
    other_feat = rng.normal(size=n)

    x = np.column_stack([np.ones(n), base_feat, collinear_feat, other_feat])
    # intervals with moderate overlap (4 bars each, step 2 bars)
    intervals = [(i * 2, i * 2 + 8) for i in range(n)]

    w_true = np.array([0.5, 1.0, 0.0, -0.5])
    sim_coefs: list[np.ndarray] = []
    sim_ses: list[np.ndarray] = []

    for _ in range(60):
        # Generate underlying bar-level returns so overlapping intervals genuinely share noise
        delta = rng.normal(scale=0.05, size=200)
        noise = np.array([np.sum(delta[s:e]) / np.sqrt(e - s) for s, e in intervals])
        y = x @ w_true + noise
        rows = [x[i] for i in range(n)]
        labels = [float(v) for v in y]
        coefs, cov = _fit_standardized_ridge(
            rows, labels, penalty=0.05, intervals=intervals, compute_covariance=True
        )
        assert cov is not None
        c_arr = np.asarray(coefs)
        # Weights must be bounded under 1e-8 collinearity (rank-truncation prevents blowup)
        assert np.all(np.abs(c_arr) < 10.0)
        sim_coefs.append(c_arr)
        cov_arr = np.asarray(cov)
        diag = np.maximum(0.0, np.diag(cov_arr))
        sim_ses.append(np.sqrt(diag))

    emp_sd = np.std(sim_coefs, axis=0)
    mean_se = np.mean(sim_ses, axis=0)

    # For non-collinear features (bias and other_feat), SE / empirical SD within [0.75, 1.35]
    for idx in (0, 3):
        ratio = mean_se[idx] / emp_sd[idx]
        assert 0.75 <= ratio <= 1.35


def test_shallow_policy_zero_sensitivity_on_unexcited_weights() -> None:
    from trade_rl.evaluation.decision_value_fit import (
        fit_synthetic_shallow_decision_value,
    )
    from trade_rl.strategies.decision_value_shallow import ShallowDecisionValuePolicy

    batch = _fit_batch()
    shallow_model = fit_synthetic_shallow_decision_value(
        batch, candidate_actions=_CANDIDATES
    )
    # The last 2 features are account.weights for AAA and BBB.
    # Since _fit_batch training was conducted from flat start, weights had 0 variance
    assert shallow_model.input_scales is not None
    assert shallow_model.input_scales[-2:] == (0.0, 0.0)

    context = LinearDecisionValueRuntimeContext(
        source_scope="synthetic_only",
        feature_config_digest=shallow_model.feature_config_digest,
        normalization_digest=shallow_model.normalization_digest,
        execution_policy_digest=shallow_model.execution_policy_digest,
        reference_policy_digest=shallow_model.reference_policy_digest,
    )
    policy = ShallowDecisionValuePolicy(
        shallow_model,
        runtime_context=context,
        eval_one_way_cost=0.0,
        train_one_way_cost=0.0,
    )

    # Create two snapshots with same market signals but differing non-zero weights
    market = _market(signal=0.8, move=0.015, day="2026-03-03", tag="sh-sens")
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
    from trade_rl.strategies.portfolio_observation import (
        build_portfolio_decision_snapshot,
    )

    session = SharedPortfolioSession(
        market,
        start_index=1,
        stop_index=6,
        initial_capital=1000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    snap_flat = build_portfolio_decision_snapshot(
        market,
        index=1,
        agent_stop_index=6,
        book=session.book,
        active_orders=session.active_orders,
        risk_config=_RISK.config,
    )
    snap_holding = replace(
        snap_flat,
        account=replace(
            snap_flat.account,
            weights=(0.25, 0.0),
            quantities=(2.5, 0.0),
            exact_quantities=("2.5", "0"),
        ),
    )

    # Hidden basis layer zeros out unexcited features
    basis_flat = policy._basis(snap_flat)
    basis_holding = policy._basis(snap_holding)
    np.testing.assert_allclose(basis_flat, basis_holding)

    # At zero switch cost, value_for_action is identically equal
    val_flat = policy.value_for_action(snap_flat, _LONG)
    val_holding = policy.value_for_action(snap_holding, _LONG)
    assert val_flat == pytest.approx(val_holding)


def test_conservative_policy_epistemic_prior_preserves_entry_hysteresis() -> None:
    from trade_rl.evaluation.decision_value_fit import (
        fit_conservative_linear_decision_value,
    )
    from trade_rl.simulation.orders.model import (
        OrderStatus,
        OrderType,
        TimeInForce,
    )
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
    from trade_rl.strategies.decision_value import (
        ConservativeLinearDecisionValuePolicy,
    )
    from trade_rl.strategies.portfolio import PortfolioPendingOrderSnapshot
    from trade_rl.strategies.portfolio_observation import (
        build_portfolio_decision_snapshot,
    )

    batch = _fit_batch()
    model = fit_conservative_linear_decision_value(
        batch,
        candidate_actions=_CANDIDATES,
        uncertainty_penalty_weight=1.0,
        one_way_switch_cost=0.0008,
        unexcited_prior_std=1e-3,
    )
    context = LinearDecisionValueRuntimeContext(
        source_scope="synthetic_only",
        feature_config_digest=model.feature_config_digest,
        normalization_digest=model.normalization_digest,
        execution_policy_digest=model.execution_policy_digest,
        reference_policy_digest=model.reference_policy_digest,
    )
    pol = ConservativeLinearDecisionValuePolicy(
        model,
        runtime_context=context,
        allow_active_orders=True,
        eval_one_way_cost=0.0008,
        train_one_way_cost=0.0008,
    )

    market = _market(signal=0.8, move=0.015, day="2026-03-03", tag="cons-hyst")
    session = SharedPortfolioSession(
        market,
        start_index=1,
        stop_index=6,
        initial_capital=1000.0,
        execution_cost=_COST,
        risk=_RISK,
    )
    snap_flat = build_portfolio_decision_snapshot(
        market,
        index=1,
        agent_stop_index=6,
        book=session.book,
        active_orders=session.active_orders,
        risk_config=_RISK.config,
    )
    snap_held = replace(
        snap_flat,
        account=replace(
            snap_flat.account,
            weights=(0.25, 0.0),
            quantities=(2.5, 0.0),
            exact_quantities=("2.5", "0"),
        ),
    )
    order = PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type=OrderType.MARKET.value,
        time_in_force=TimeInForce.IOC.value,
        status=OrderStatus.SUBMITTED.value,
        requested_quantity=2.5,
        remaining_quantity=2.5,
        submit_index=1,
        eligible_index=2,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
        limit_price=None,
        stop_price=None,
        submission_reference_price=100.0,
    )
    snap_inflight = replace(snap_flat, active_orders=(order,))

    action = _LONG
    # 1. Epistemic uncertainty on unexcited coordinates is strictly positive
    std_flat = pol.predictive_std_for_action(snap_flat, action)
    std_held = pol.predictive_std_for_action(snap_held, action)
    std_inflight = pol.predictive_std_for_action(snap_inflight, action)
    assert std_held > std_flat
    assert std_inflight > std_flat

    # 2. Sunk-cost hysteresis dominates the small OOD state prior
    val_flat = pol.conservative_value_for_action(snap_flat, action)
    val_held = pol.conservative_value_for_action(snap_held, action)
    val_inflight = pol.conservative_value_for_action(snap_inflight, action)
    assert val_held > val_flat
    assert val_inflight > val_flat
