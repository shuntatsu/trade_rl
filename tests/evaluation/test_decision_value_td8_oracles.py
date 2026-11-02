"""Sanity oracles for the corrected HOLD-relative rule (#1025 TD8).

* An intercept-only value model in an up-drift market must behave like
  ``fixed_long`` (the decision machinery adds nothing beyond its labels).
* On pure noise, a zero margin churns while the pre-registered margin does not.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.decision_value_td8_support import (
    COST,
    ONE_WAY,
    RISK_CONFIG,
    TARGETS,
    risk,
    signal_market,
)
from trade_rl.evaluation.decision_value_dev_study import (
    _FixedEntryPolicy,
    _HoldQuantityPolicy,
)
from trade_rl.evaluation.decision_value_fit import fit_synthetic_linear_decision_value
from trade_rl.evaluation.decision_value_reachable_labels import (
    collect_reachable_state_labels,
)
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label_batch,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.strategies.decision_value import (
    LinearDecisionValueModel,
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)

_H = ACTION_HOLD
_CANDIDATES = (
    (_H, _H),
    (ACTION_FLAT, ACTION_FLAT),
    (ACTION_LONG, ACTION_LONG),
    (ACTION_SHORT, ACTION_SHORT),
)
_REF = "d" * 64


def _policy(model, *, k: float) -> LinearDecisionValuePolicy:
    return LinearDecisionValuePolicy(
        model,
        runtime_context=LinearDecisionValueRuntimeContext(
            source_scope="real_development",
            feature_config_digest=model.feature_config_digest,
            normalization_digest=model.normalization_digest,
            execution_policy_digest=model.execution_policy_digest,
            reference_policy_digest=model.reference_policy_digest,
        ),
        allow_active_orders=True,
        eval_one_way_cost=ONE_WAY,
        train_one_way_cost=ONE_WAY,
        hold_relative=True,
        decision_margin_multiplier=k,
    )


def _replay(market, policy, start: int, stop: int):
    return run_shared_portfolio_replay(
        market,
        policy,
        start_index=start,
        stop_index=stop,
        initial_capital=100_000.0,
        execution_cost=COST,
        risk=risk(),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )


def test_intercept_only_value_model_reproduces_fixed_long_in_an_uptrend() -> None:
    market = signal_market(300, drift=0.002, signal_strength=0.0, noise=0.003)
    horizon, fit_index = 6, 150
    collection = collect_reachable_state_labels(
        market,
        reference_policy_provider=lambda _t, _s: lambda: _HoldQuantityPolicy(TARGETS),
        reference_policy_digest=_REF,
        branch_indices=tuple(range(20, 120, horizon)),
        train_start_index=20,
        prefix_bars=12,
        horizon_bars=horizon,
        candidate_actions=_CANDIDATES,
        target_abs_weights=TARGETS,
        initial_capital=100_000.0,
        execution_cost=COST,
        risk_config=RISK_CONFIG,
        fit_cutoff=market.timestamps[fit_index],
        source_scope="real_development",
    )
    batch = build_synthetic_paired_label_batch(
        collection.samples,
        fit_cutoff=market.timestamps[fit_index],
        evaluation_start=market.timestamps[fit_index + 1],
        expected_reference_policy_digest=_REF,
    )
    fitted = fit_synthetic_linear_decision_value(
        batch,
        candidate_actions=_CANDIDATES,
        ridge_penalty=10.0,
        solver_version="td8_v2",
    )
    width = len(fitted.coefficients[0])
    by_action = {}
    for sample in collection.samples:
        by_action.setdefault(sample.candidate_action, []).append(sample.label_delta)
    coefs = []
    for action in _CANDIDATES[1:]:
        row = [0.0] * width
        row[0] = float(np.mean(by_action[action]))
        coefs.append(tuple(row))
    intercept_only = replace(fitted, coefficients=tuple(coefs))
    assert isinstance(intercept_only, LinearDecisionValueModel)
    ll = _CANDIDATES.index((ACTION_LONG, ACTION_LONG)) - 1
    ss = _CANDIDATES.index((ACTION_SHORT, ACTION_SHORT)) - 1
    assert coefs[ll][0] > 0.0 > coefs[ss][0]

    start, stop = fit_index + 1, 290
    learned = _replay(market, _policy(intercept_only, k=0.0), start, stop)
    passive = _replay(
        market,
        _FixedEntryPolicy(
            action=(ACTION_LONG, ACTION_LONG),
            target_abs_weights=TARGETS,
            proposal_version="td8_v2",
        ),
        start,
        stop,
    )
    assert float(learned.book.portfolio_value) == pytest.approx(
        float(passive.book.portfolio_value), rel=1e-12
    )
    assert float(learned.book.total_cost) == pytest.approx(
        float(passive.book.total_cost), rel=1e-12
    )
    assert float(passive.book.portfolio_value) > 100_000.0


def _noise_model(market, scale: float) -> LinearDecisionValueModel:
    n_features = len(market.feature_names)
    width = 1 + 2 * n_features + len(market.global_feature_names) + 2
    long_row = [0.0] * width
    long_row[1] = scale  # BTC feature 0
    short_row = [0.0] * width
    short_row[1] = -scale
    flat_row = [0.0] * width
    return LinearDecisionValueModel(
        schema_version="synthetic_linear_decision_value_v2",
        source_scope="real_development",
        symbols=market.symbols,
        feature_names=market.feature_names,
        global_feature_names=market.global_feature_names,
        feature_config_digest=market.feature_config_digest,
        normalization_digest=market.normalization_digest,
        execution_policy_digest=COST.execution_policy_digest,
        reference_policy_digest=_REF,
        target_abs_weights=TARGETS,
        candidate_actions=_CANDIDATES,
        fit_cutoff_ns=int(market.timestamps[150].astype(np.int64)),
        evaluation_start_ns=int(market.timestamps[151].astype(np.int64)),
        ridge_penalty=1.0,
        training_rows=20,
        overlap_groups=1,
        coefficients=(tuple(flat_row), tuple(long_row), tuple(short_row)),
    )


def test_zero_margin_churns_on_noise_and_the_margin_suppresses_it() -> None:
    market = signal_market(500, signal_strength=0.0, noise=0.003, seed=11)
    rng = np.random.default_rng(3)
    features = market.features.copy()
    features[:, :, 0] = rng.normal(0.0, 0.004, size=features.shape[:2])
    market = replace(market, features=features, identity_payload_json=None)
    model = _noise_model(market, scale=0.25)

    zero = _replay(market, _policy(model, k=0.0), 151, 480)
    margin = _replay(market, _policy(model, k=2.0), 151, 480)

    assert float(zero.book.turnover_total) > 0.0
    # The noise score flips sign on ~half the bars; the margin ignores weak flips.
    assert float(margin.book.turnover_total) < 0.4 * float(zero.book.turnover_total)
    assert float(margin.book.total_cost) < 0.4 * float(zero.book.total_cost)
    assert float(zero.book.total_cost) > 0.0
