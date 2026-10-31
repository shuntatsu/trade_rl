"""Labels from reachable non-flat states under a frozen B0 continuation (#1025 TD8)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.decision_value_td8_support import (
    COST,
    ONE_WAY,
    RISK_CONFIG,
    TARGETS,
    signal_market,
)
from trade_rl.evaluation.decision_value_dev_study import (
    _HoldQuantityPolicy,
    _perturb_after,
    causal_b0_reference_provider,
)
from trade_rl.evaluation.decision_value_fit import (
    fit_conservative_linear_decision_value,
    fit_synthetic_linear_decision_value,
)
from trade_rl.evaluation.decision_value_reachable_labels import (
    collect_reachable_state_labels,
)
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label_batch,
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
    (ACTION_LONG, ACTION_FLAT),
    (ACTION_SHORT, ACTION_FLAT),
    (ACTION_FLAT, ACTION_LONG),
    (ACTION_FLAT, ACTION_SHORT),
    (ACTION_LONG, ACTION_LONG),
    (ACTION_SHORT, ACTION_SHORT),
    (ACTION_LONG, ACTION_SHORT),
    (ACTION_SHORT, ACTION_LONG),
)
_HORIZON = 6
_PREFIX = 12
_REF_DIGEST = "d" * 64


def _provider(dataset, *, min_fit_bars: int = 12, first_fit_index: int = 10):
    return causal_b0_reference_provider(
        dataset,
        feature_indices=(0, 1, 2),
        horizon_hours=1,
        alpha=1.0,
        entry_threshold=0.0004,
        exit_threshold=0.0001,
        target_abs_weights=TARGETS,
        one_way_switch_cost=ONE_WAY,
        min_fit_bars=min_fit_bars,
        first_fit_index=first_fit_index,
    )


def _collect(market, provider, *, branch_indices, train_start=10, fit_cutoff_index=300):
    return collect_reachable_state_labels(
        market,
        reference_policy_provider=provider,
        reference_policy_digest=_REF_DIGEST,
        branch_indices=branch_indices,
        train_start_index=train_start,
        prefix_bars=_PREFIX,
        horizon_bars=_HORIZON,
        candidate_actions=_CANDIDATES,
        target_abs_weights=TARGETS,
        initial_capital=100_000.0,
        execution_cost=COST,
        risk_config=RISK_CONFIG,
        fit_cutoff=market.timestamps[min(fit_cutoff_index, market.n_bars - 1)],
        source_scope="real_development",
    )


def test_b0_prefix_yields_non_flat_predecision_states_and_prefix_windows() -> None:
    market = signal_market(360)
    provider = _provider(market)
    branches = tuple(range(40, 160, _HORIZON))
    collection = _collect(market, provider, branch_indices=branches)

    assert collection.skipped_branches_without_reference == 0
    assert collection.nonflat_branch_states > 0
    assert collection.paired_replay_runs == 2 * len(branches) * (len(_CANDIDATES) - 1)
    non_flat = [
        s
        for s in collection.samples
        if any(Fraction(q) != 0 for q in s.observation.account.exact_quantities)
    ]
    assert non_flat, "B0 prefix must reach non-flat predecision states"
    assert any(any(w != 0.0 for w in s.observation.account.weights) for s in non_flat)

    # Window bookkeeping: start = max(train_start, t - prefix), stop = t + horizon.
    for t, start, stop in collection.branch_windows:
        assert start == max(10, t - _PREFIX)
        assert stop == t + _HORIZON
    # Every candidate of every branch is retained (losses and zeros included).
    labelled = {(s.decision_time_ns, s.candidate_action) for s in collection.samples}
    assert len(labelled) == len(collection.samples)
    assert len(collection.samples) + collection.ineligible_pairs == len(branches) * (
        len(_CANDIDATES) - 1
    )
    assert any(s.label_delta < 0.0 for s in collection.samples)
    assert all(s.baseline_action == (_H, _H) for s in collection.samples)


def test_reference_policy_is_a_causal_fresh_refit_at_the_prefix_start() -> None:
    market = signal_market(200)
    seen: list[tuple[int, int]] = []
    base = _provider(market)

    def spying_provider(t: int, start: int):
        factory = base(t, start)
        assert factory is not None
        first, second = factory(), factory()
        assert first is not second  # TD1: a fresh instance per branch arm
        model = first._strategies[0].model  # noqa: SLF001
        seen.append((t, start))
        assert model.fit_cutoff == market.timestamps[start]
        return factory

    _collect(market, spying_provider, branch_indices=(40, 52, 64))
    assert [s[1] for s in seen] == [28, 40, 52]


def test_branches_without_a_sufficiently_informed_causal_reference_are_counted() -> (
    None
):
    market = signal_market(200)
    provider = _provider(market, min_fit_bars=30)
    collection = _collect(market, provider, branch_indices=(14, 30, 48, 70, 90))
    # start = max(10, t-12): 10, 18, 36, 58, 78 -> bars of history 0, 8, 26, 48, 68.
    assert collection.skipped_branches_without_reference == 3
    assert [t for t, _s, _e in collection.branch_windows] == [70, 90]
    assert {s.decision_time_ns for s in collection.samples} == {
        int(market.timestamps[t].astype(np.int64)) for t in (70, 90)
    }


def test_first_branch_after_train_start_has_a_zero_length_flat_prefix() -> None:
    market = signal_market(200)
    provider = _provider(market, min_fit_bars=0)
    collection = _collect(market, provider, branch_indices=(10, 16, 22))
    windows = {t: (start, stop) for t, start, stop in collection.branch_windows}
    assert windows[10][0] == 10  # zero-length prefix: a flat 100k account
    assert windows[16][0] == 10
    assert windows[22][0] == 10
    first = [
        s
        for s in collection.samples
        if s.decision_time_ns == int(market.timestamps[10].astype(np.int64))
    ]
    assert first and all(
        all(Fraction(q) == 0 for q in s.observation.account.exact_quantities)
        for s in first
    )


def test_holding_the_existing_posture_has_zero_increment_and_flatten_is_labelled() -> (
    None
):
    market = signal_market(360, drift=0.001)
    collection = _collect(
        market, _provider(market), branch_indices=tuple(range(60, 160, _HORIZON))
    )
    long_long = [
        s
        for s in collection.samples
        if s.candidate_action == (ACTION_LONG, ACTION_LONG)
        and all(Fraction(q) > 0 for q in s.observation.account.exact_quantities)
    ]
    # A candidate that reproduces the current posture is the HOLD no-op.
    for sample in long_long:
        w = sample.observation.account.weights
        if all(abs(x - 0.25) < 0.01 for x in w):
            assert abs(sample.label_delta) < 5e-3
    flats = [s for s in collection.samples if s.candidate_action == (ACTION_FLAT,) * 2]
    assert flats
    assert any(abs(s.label_delta) > 0.0 for s in flats), (
        "FLAT must differ from HOLD once the book is non-flat"
    )


def test_account_weight_coefficients_are_not_identically_zero_when_excited() -> None:
    market = signal_market(420)
    branches = tuple(range(30, 250, _HORIZON))
    collection = _collect(
        market, _provider(market), branch_indices=branches, fit_cutoff_index=300
    )
    batch = build_synthetic_paired_label_batch(
        collection.samples,
        fit_cutoff=market.timestamps[300],
        evaluation_start=market.timestamps[301],
        expected_reference_policy_digest=_REF_DIGEST,
    )
    n_symbols = 2
    b1 = fit_synthetic_linear_decision_value(
        batch, candidate_actions=_CANDIDATES, ridge_penalty=1.0, solver_version="td8_v2"
    )
    d_feat = len(b1.coefficients[0]) - 1
    weight_cols = slice(1 + d_feat - n_symbols, 1 + d_feat)
    weight_coefs = np.asarray([row[weight_cols] for row in b1.coefficients])
    assert np.any(weight_coefs != 0.0)
    b1c = fit_conservative_linear_decision_value(
        batch,
        candidate_actions=_CANDIDATES,
        ridge_penalty=1.0,
        solver_version="td8_v2",
        one_way_switch_cost=ONE_WAY,
    )
    d_feat_c = len(b1c.coefficients[0]) - 1
    actual_cols = slice(1 + d_feat_c - 2 * n_symbols, 1 + d_feat_c - n_symbols)
    assert np.any(np.asarray([row[actual_cols] for row in b1c.coefficients]) != 0.0)


def test_labels_do_not_depend_on_data_after_the_label_horizon() -> None:
    market = signal_market(360)
    branches = tuple(range(40, 100, _HORIZON))
    last_needed = max(branches) + _HORIZON
    left = _collect(market, _provider(market), branch_indices=branches)
    future_changed = _perturb_after(market, last_needed)
    right = _collect(future_changed, _provider(future_changed), branch_indices=branches)
    assert [s.label_delta for s in left.samples] == [
        s.label_delta for s in right.samples
    ]
    assert [s.candidate_action for s in left.samples] == [
        s.candidate_action for s in right.samples
    ]


def test_hold_reference_gives_flat_states_and_matches_flat_start_label() -> None:
    market = signal_market(200, drift=0.002)
    collection = _collect(
        market,
        lambda _t, _s: lambda: _HoldQuantityPolicy(TARGETS),
        branch_indices=tuple(range(40, 100, _HORIZON)),
    )
    assert collection.nonflat_branch_states == 0
    long_long = [
        s.label_delta
        for s in collection.samples
        if s.candidate_action == (ACTION_LONG, ACTION_LONG)
    ]
    assert long_long and np.mean(long_long) > 0.0


def test_invalid_arguments_fail_closed() -> None:
    market = signal_market(200)
    provider = _provider(market)
    with pytest.raises(ValueError, match="prefix"):
        collect_reachable_state_labels(
            market,
            reference_policy_provider=provider,
            reference_policy_digest=_REF_DIGEST,
            branch_indices=(40,),
            train_start_index=10,
            prefix_bars=-1,
            horizon_bars=_HORIZON,
            candidate_actions=_CANDIDATES,
            target_abs_weights=TARGETS,
            initial_capital=100_000.0,
            execution_cost=COST,
            risk_config=RISK_CONFIG,
            fit_cutoff=market.timestamps[150],
            source_scope="real_development",
        )
    with pytest.raises(ValueError, match="branch"):
        _collect(market, provider, branch_indices=())


def test_td8_reference_vintage_is_immune_to_future_train_window_features() -> None:
    """A teacher refitted at prefix start must not see later train-window stats."""
    from dataclasses import replace

    from trade_rl.evaluation.decision_value_feature_transform import (
        apply_feature_transform,
        fit_train_feature_transform,
    )
    from trade_rl.risk.pretrade import PreTradeRisk
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
    from trade_rl.strategies.portfolio_observation import (
        build_portfolio_decision_snapshot,
    )

    raw = signal_market(420, seed=17)
    mutated_features = raw.features.copy()
    mutated_features[201:301] = mutated_features[201:301] * 18 + 12
    future_changed = replace(raw, features=mutated_features, identity_payload_json=None)
    assert np.array_equal(raw.features[:201], future_changed.features[:201])
    proposals = []
    fitted_coefs = []
    for market in (raw, future_changed):
        transform = fit_train_feature_transform(
            market, train_start_index=24, train_stop_index=300, clip=3.0
        )
        block_view = apply_feature_transform(market, transform)
        provider = causal_b0_reference_provider(
            block_view,
            raw_dataset=market,
            causal_feature_clip=3.0,
            feature_indices=(0, 1, 2),
            horizon_hours=24,
            alpha=1.0,
            entry_threshold=0.002,
            exit_threshold=0.0005,
            target_abs_weights=TARGETS,
            one_way_switch_cost=ONE_WAY,
            min_fit_bars=96,
            first_fit_index=24,
        )
        factory = provider(248, 200)
        assert factory is not None
        policy = factory()
        session = SharedPortfolioSession(
            block_view,
            start_index=200,
            stop_index=248,
            initial_capital=100_000.0,
            execution_cost=COST,
            risk=PreTradeRisk(RISK_CONFIG),
        )
        snapshot = build_portfolio_decision_snapshot(
            block_view,
            index=200,
            agent_stop_index=248,
            book=session.book,
            active_orders=session.active_orders,
            risk_config=RISK_CONFIG,
        )
        fitted_coefs.append(policy._strategies[0].model.coefficients)  # noqa: SLF001
        proposals.append(policy.decide_portfolio(snapshot))
    assert np.array_equal(fitted_coefs[0], fitted_coefs[1])
    assert proposals[0] == proposals[1]
