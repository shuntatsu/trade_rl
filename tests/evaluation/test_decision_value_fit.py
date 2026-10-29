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
