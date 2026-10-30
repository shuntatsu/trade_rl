"""HOLD-relative decision rule with a pre-registered decision margin (#1025 TD8).

The label of the corrected design is a native paired increment over exact
quantity-HOLD from a reachable (usually non-flat) state, so Q(s, HOLD) is
structurally zero and an action is taken only when its increment beats a margin
proportional to the candidate's round-trip cost. Ties go to HOLD.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.strategies.decision_value import (
    ConservativeLinearDecisionValueModel,
    ConservativeLinearDecisionValuePolicy,
    LinearDecisionValueModel,
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
    candidate_target_weights,
    compute_hold_relative_cost_adjustment,
    round_trip_cost_fraction,
    select_hold_relative_action,
)
from trade_rl.strategies.decision_value_shallow import (
    ShallowDecisionValueModel,
    ShallowDecisionValuePolicy,
    fixed_hidden_weights,
)
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)

_HOLD = (ACTION_HOLD, ACTION_HOLD)
_LONG = (ACTION_LONG, ACTION_HOLD)
_SHORT = (ACTION_SHORT, ACTION_HOLD)
_FLAT = (ACTION_FLAT, ACTION_HOLD)
_ONE_WAY = 0.0008


def _snapshot(*, long_first: bool = False, signal: float = 0.0):
    qty = ("2.5", "0") if long_first else ("0", "0")
    weights = (0.25, 0.0) if long_first else (0.0, 0.0)
    account = PortfolioAccountSnapshot(
        cash=1_000.0 - (250.0 if long_first else 0.0),
        portfolio_value=1_000.0,
        peak_value=1_000.0,
        max_drawdown=0.0,
        exact_quantities=qty,
        quantities=(float(qty[0]), 0.0),
        weights=weights,
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


def _model(*, long_intercept: float = 0.0, flat_intercept: float = 0.0):
    # [intercept, AAA signal, BBB signal, regime, AAA weight, BBB weight]
    return LinearDecisionValueModel(
        schema_version="synthetic_linear_decision_value_v2",
        source_scope="real_development",
        symbols=("AAA", "BBB"),
        feature_names=("signal",),
        global_feature_names=("regime",),
        feature_config_digest="0" * 64,
        normalization_digest="0" * 64,
        execution_policy_digest="a" * 64,
        reference_policy_digest="b" * 64,
        target_abs_weights=(0.4, 0.4),
        candidate_actions=(_HOLD, _LONG, _SHORT, _FLAT),
        fit_cutoff_ns=1_770_000_000_000_000_000,
        evaluation_start_ns=1_770_100_000_000_000_000,
        ridge_penalty=1e-4,
        training_rows=18,
        overlap_groups=1,
        coefficients=(
            (long_intercept, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (flat_intercept, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )


def _context(model) -> LinearDecisionValueRuntimeContext:
    return LinearDecisionValueRuntimeContext(
        source_scope="real_development",
        feature_config_digest=model.feature_config_digest,
        normalization_digest=model.normalization_digest,
        execution_policy_digest=model.execution_policy_digest,
        reference_policy_digest=model.reference_policy_digest,
    )


def _policy(
    model, *, k: float, eval_cost: float = _ONE_WAY
) -> LinearDecisionValuePolicy:
    return LinearDecisionValuePolicy(
        model,
        runtime_context=_context(model),
        allow_active_orders=True,
        eval_one_way_cost=eval_cost,
        train_one_way_cost=_ONE_WAY,
        hold_relative=True,
        decision_margin_multiplier=k,
    )


def test_round_trip_cost_fraction_scales_with_gross_target() -> None:
    assert round_trip_cost_fraction((0.25, 0.25), 0.001) == pytest.approx(0.001)
    assert round_trip_cost_fraction((0.0, 0.0), 0.001) == 0.0
    assert round_trip_cost_fraction((-0.25, 0.25), 0.002) == pytest.approx(0.002)
    with pytest.raises(ValueError):
        round_trip_cost_fraction((0.1,), -0.001)


def test_candidate_target_weights_resolve_hold_and_flat_per_symbol() -> None:
    assert candidate_target_weights(
        (ACTION_LONG, ACTION_HOLD), (0.0, -0.1), (0.4, 0.4)
    ) == (0.4, -0.1)
    assert candidate_target_weights(
        (ACTION_FLAT, ACTION_SHORT), (0.3, 0.3), (0.4, 0.4)
    ) == (0.0, -0.4)


def test_margin_is_proportional_to_round_trip_cost_and_ties_go_to_hold() -> None:
    # LONG target weights are (0.4, 0.0): round trip = 2 * 0.0008 * 0.4.
    margin_one = round_trip_cost_fraction((0.4, 0.0), _ONE_WAY)
    snapshot = _snapshot()

    # Clearly above 2x margin: acts.
    model = _model(long_intercept=2.0 * margin_one + 1e-4)
    assert _policy(model, k=2.0).decide_action(snapshot) == _LONG
    # Same increment but a 4x margin demands more: HOLD.
    assert _policy(model, k=4.0).decide_action(snapshot) == _HOLD
    # Exactly equal to the margin is a tie, and ties go to HOLD.
    tie = _model(long_intercept=2.0 * margin_one)
    assert _policy(tie, k=2.0).decide_action(snapshot) == _HOLD
    # Zero margin keeps the only legacy floor: strictly positive increment.
    assert _policy(_model(long_intercept=1e-9), k=0.0).decide_action(snapshot) == _LONG
    assert _policy(_model(long_intercept=0.0), k=0.0).decide_action(snapshot) == _HOLD
    assert _policy(_model(long_intercept=-1e-3), k=0.0).decide_action(snapshot) == _HOLD


def test_hold_has_structural_zero_value_and_matching_posture_is_never_chosen() -> None:
    model = _model(long_intercept=0.5)
    policy = _policy(model, k=0.0)
    flat = _snapshot()
    held_long = _snapshot(long_first=True)
    assert policy.value_for_action(flat, _HOLD) == 0.0
    assert policy.value_for_action(held_long, _HOLD) == 0.0
    # Already long in AAA: LONG maps to exactly the HOLD proposal, so it is a
    # no-op, not an edge to be re-collected every bar.
    assert policy.decide_action(held_long) == _HOLD
    assert policy.decide_action(flat) == _LONG


def test_flat_is_a_learned_head_not_an_implicit_liquidation() -> None:
    held_long = _snapshot(long_first=True)
    # Negative FLAT increment -> keep the position (legacy rule would flatten).
    negative = _policy(_model(flat_intercept=-0.01), k=0.0)
    assert negative.decide_action(held_long) == _HOLD
    proposal = negative.decide_portfolio(held_long)
    assert proposal.hold_current_quantity_symbols == (0, 1)
    # Positive FLAT increment above margin -> flatten (margin of a flat target is 0).
    positive = _policy(_model(flat_intercept=0.01), k=2.0)
    assert positive.decide_action(held_long) == _FLAT
    assert positive.decide_portfolio(held_long).target_weights[0] == 0.0


def test_stress_cost_adjusts_only_the_incremental_turnover() -> None:
    train = 0.0008
    # From flat, LONG(AAA) trades 0.4 now and (by construction) closes 0.4 later.
    same = compute_hold_relative_cost_adjustment(
        _LONG, (0.0, 0.0), (0.4, 0.4), eval_one_way_cost=train, train_one_way_cost=train
    )
    assert same == 0.0
    stressed = compute_hold_relative_cost_adjustment(
        _LONG,
        (0.0, 0.0),
        (0.4, 0.4),
        eval_one_way_cost=2 * train,
        train_one_way_cost=train,
    )
    assert stressed == pytest.approx(train * (0.4 + 0.4))
    # From a held 0.25 long, HOLD-relative extra turnover of staying long is 0.
    staying = compute_hold_relative_cost_adjustment(
        (ACTION_LONG, ACTION_HOLD),
        (0.4, 0.0),
        (0.4, 0.4),
        eval_one_way_cost=2 * train,
        train_one_way_cost=train,
    )
    assert staying == pytest.approx(0.0)
    model = _model(long_intercept=0.01)
    value = _policy(model, k=0.0, eval_cost=2 * train).value_for_action(
        _snapshot(), _LONG
    )
    assert value == pytest.approx(0.01 - train * 0.8)


def test_selector_rejects_noop_equivalents_and_uses_margin_per_candidate() -> None:
    snapshot = _snapshot(long_first=True)
    chosen = select_hold_relative_action(
        (_HOLD, _LONG, _SHORT, _FLAT),
        snapshot,
        lambda action: {_LONG: 9.0, _SHORT: 0.0004, _FLAT: 0.0}[action],
        target_abs_weights=(0.4, 0.4),
        eval_one_way_cost=_ONE_WAY,
        margin_multiplier=1.0,
    )
    # LONG is the HOLD no-op (ignored); SHORT's increment is below its margin.
    assert chosen == _HOLD


def test_legacy_policy_semantics_are_untouched_by_default() -> None:
    model = _model(long_intercept=1e-9)
    legacy_model = replace(model, schema_version="synthetic_linear_decision_value_v1")
    legacy = LinearDecisionValuePolicy(
        legacy_model, runtime_context=_context(legacy_model)
    )
    assert legacy.decide_action(_snapshot()) == _LONG
    # Default legacy FLAT sentinel still means structural zero for all-FLAT only.
    assert legacy.value_for_action(_snapshot(), _HOLD) == 0.0


def test_conservative_and_shallow_use_the_same_rule() -> None:
    base = _model(long_intercept=0.005)
    width = 1 + 2 * 1 + 1 + 2 + 2
    cons = ConservativeLinearDecisionValueModel(
        schema_version="conservative_linear_decision_value_v3",
        source_scope=base.source_scope,
        symbols=base.symbols,
        feature_names=base.feature_names,
        global_feature_names=base.global_feature_names,
        feature_config_digest=base.feature_config_digest,
        normalization_digest=base.normalization_digest,
        execution_policy_digest=base.execution_policy_digest,
        reference_policy_digest=base.reference_policy_digest,
        target_abs_weights=base.target_abs_weights,
        candidate_actions=base.candidate_actions,
        fit_cutoff_ns=base.fit_cutoff_ns,
        evaluation_start_ns=base.evaluation_start_ns,
        ridge_penalty=10.0,
        uncertainty_penalty_weight=1.0,
        one_way_switch_cost=_ONE_WAY,
        training_rows=18,
        overlap_groups=1,
        coefficients=(
            (0.005,) + (0.0,) * (width - 1),
            (-1.0,) + (0.0,) * (width - 1),
            (0.0,) + (0.0,) * (width - 1),
        ),
        covariance_matrices=tuple(
            tuple(tuple(0.0 for _ in range(width)) for _ in range(width))
            for _ in range(3)
        ),
    )
    cons_policy = ConservativeLinearDecisionValuePolicy(
        cons,
        runtime_context=_context(cons),
        eval_one_way_cost=_ONE_WAY,
        train_one_way_cost=_ONE_WAY,
        hold_relative=True,
        decision_margin_multiplier=2.0,
    )
    margin = 2.0 * round_trip_cost_fraction((0.4, 0.0), _ONE_WAY)
    assert cons_policy.value_for_action(_snapshot(), _HOLD) == 0.0
    assert cons_policy.decide_action(_snapshot()) == (
        _LONG if 0.005 > margin else _HOLD
    )
    state_width = 1 + 2 * 1 + 1 + 2
    hidden = fixed_hidden_weights(state_width)
    shallow = ShallowDecisionValueModel(
        schema_version="synthetic_shallow_decision_value_v2",
        source_scope=base.source_scope,
        symbols=base.symbols,
        feature_names=base.feature_names,
        global_feature_names=base.global_feature_names,
        feature_config_digest=base.feature_config_digest,
        normalization_digest=base.normalization_digest,
        execution_policy_digest=base.execution_policy_digest,
        reference_policy_digest=base.reference_policy_digest,
        target_abs_weights=base.target_abs_weights,
        candidate_actions=base.candidate_actions,
        fit_cutoff_ns=base.fit_cutoff_ns,
        evaluation_start_ns=base.evaluation_start_ns,
        ridge_penalty=1e-3,
        training_rows=18,
        overlap_groups=1,
        hidden_weights=hidden,
        input_means=(0.0,) * (state_width - 1),
        input_scales=(1.0,) * (state_width - 1),
        coefficients=(
            (0.005,) + (0.0,) * (state_width + 8 - 1),
            (-1.0,) + (0.0,) * (state_width + 8 - 1),
            (0.0,) + (0.0,) * (state_width + 8 - 1),
        ),
    )
    shallow_policy = ShallowDecisionValuePolicy(
        shallow,
        runtime_context=_context(shallow),
        allow_active_orders=True,
        eval_one_way_cost=_ONE_WAY,
        train_one_way_cost=_ONE_WAY,
        hold_relative=True,
        decision_margin_multiplier=2.0,
    )
    assert shallow_policy.decide_action(_snapshot()) == (
        _LONG if 0.005 > margin else _HOLD
    )
    assert shallow_policy.decide_action(_snapshot(long_first=True)) == _HOLD
