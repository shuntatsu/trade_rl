"""Train-internal pre-registered decision-margin selection (#1025 TD8)."""

from __future__ import annotations

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
    _perturb_after,
    causal_b0_reference_provider,
)
from trade_rl.evaluation.decision_value_fit import fit_synthetic_linear_decision_value
from trade_rl.evaluation.decision_value_margin import (
    MarginSelection,
    select_margin_multiplier,
)
from trade_rl.evaluation.decision_value_reachable_labels import (
    collect_reachable_state_labels,
)
from trade_rl.strategies.decision_value import (
    LinearDecisionValuePolicy,
    LinearDecisionValueRuntimeContext,
)
from trade_rl.strategies.portfolio_adapter import CashPortfolioPolicy
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
_HORIZON = 6
_GRID = (0.0, 0.5, 1.0, 2.0)
_REF = "d" * 64
_FIT_CUTOFF_INDEX = 330


def _labels(market):
    provider = causal_b0_reference_provider(
        market,
        feature_indices=(0, 1, 2),
        horizon_hours=1,
        alpha=1.0,
        entry_threshold=0.0004,
        exit_threshold=0.0001,
        target_abs_weights=TARGETS,
        one_way_switch_cost=ONE_WAY,
        min_fit_bars=12,
        first_fit_index=20,
    )
    branches = tuple(range(20, 240, _HORIZON))
    collection = collect_reachable_state_labels(
        market,
        reference_policy_provider=provider,
        reference_policy_digest=_REF,
        branch_indices=branches,
        train_start_index=20,
        prefix_bars=12,
        horizon_bars=_HORIZON,
        candidate_actions=_CANDIDATES,
        target_abs_weights=TARGETS,
        initial_capital=100_000.0,
        execution_cost=COST,
        risk_config=RISK_CONFIG,
        fit_cutoff=market.timestamps[_FIT_CUTOFF_INDEX],
        source_scope="real_development",
    )
    return collection.samples


def _fit(batch):
    return fit_synthetic_linear_decision_value(
        batch,
        candidate_actions=_CANDIDATES,
        ridge_penalty=10.0,
        solver_version="td8_v2",
    )


def _policy(model, k, _dataset=None):
    return LinearDecisionValuePolicy(
        model,
        runtime_context=LinearDecisionValueRuntimeContext(
            source_scope=model.source_scope,
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


def _select(market, samples, **overrides) -> MarginSelection:
    kwargs = dict(
        dataset=market,
        samples=samples,
        fit_model=_fit,
        make_policy=_policy,
        grid=_GRID,
        calibration_fraction=0.25,
        horizon_bars=_HORIZON,
        initial_capital=100_000.0,
        execution_cost=COST,
        risk_config=RISK_CONFIG,
        reference_policy_digest=_REF,
    )
    kwargs.update(overrides)
    return select_margin_multiplier(**kwargs)


def _setup(seed: int = 7):
    market = signal_market(420, seed=seed)
    return market, _labels(market)


def test_selection_records_every_grid_point_and_uses_a_time_ordered_split() -> None:
    market, samples = _setup()
    seen: dict[str, object] = {}

    def spy_fit(batch):
        seen["fit_times"] = sorted({s.decision_time_ns for s in batch.samples})
        seen["max_terminal"] = max(s.terminal_time_ns for s in batch.samples)
        return _fit(batch)

    selection = _select(market, samples, fit_model=spy_fit)
    assert selection.multiplier in _GRID
    assert tuple(k for k, _ in selection.net_returns) == _GRID
    all_times = sorted({s.decision_time_ns for s in samples})
    n_fit = int(0.75 * len(all_times))
    assert selection.n_fit_branches == n_fit
    assert selection.n_calibration_branches == len(all_times) - n_fit
    assert seen["fit_times"] == all_times[:n_fit]
    # Fit labels matured strictly before the calibration replay begins.
    cal_start_ns = int(
        market.timestamps[selection.calibration_start_index].astype(np.int64)
    )
    assert int(seen["max_terminal"]) < cal_start_ns  # type: ignore[arg-type]
    last_t = int(np.searchsorted(market.timestamps.astype(np.int64), all_times[-1]))
    assert selection.calibration_stop_index == last_t + _HORIZON


def test_selection_is_unchanged_by_perturbing_the_future_tail() -> None:
    market, samples = _setup()
    baseline = _select(market, samples)
    last_used = baseline.calibration_stop_index
    tail_changed = _perturb_after(market, last_used)
    # Same label rows are re-collected on the perturbed dataset (they are causal).
    samples_pert = _labels(tail_changed)
    again = _select(tail_changed, samples_pert)
    assert again.multiplier == baseline.multiplier
    assert again.net_returns == baseline.net_returns


def test_exact_ties_go_to_the_larger_margin() -> None:
    market, samples = _setup()

    class _Cash:
        def __init__(self, symbols):
            self._cash = CashPortfolioPolicy(symbols=symbols)

        def decide_portfolio(self, snapshot):
            return self._cash.decide_portfolio(snapshot)

    selection = _select(
        market,
        samples,
        make_policy=lambda _model, _k, _market: _Cash(market.symbols),
    )
    assert {r for _, r in selection.net_returns} == {0.0}
    assert selection.multiplier == max(_GRID)


def test_selection_prefers_the_best_after_cost_return_and_rejects_bad_grid() -> None:
    market, samples = _setup()
    best = _select(market, samples)
    assert best.multiplier == max(
        (k for k, r in best.net_returns if r == max(rr for _, rr in best.net_returns)),
    )
    with pytest.raises(ValueError, match="grid"):
        _select(market, samples, grid=())
    with pytest.raises(ValueError, match="grid"):
        _select(market, samples, grid=(1.0, 0.5))
    with pytest.raises(ValueError, match="fraction"):
        _select(market, samples, calibration_fraction=1.0)
    with pytest.raises(ValueError, match="labels"):
        _select(market, ())


def test_temporary_margin_fit_is_immune_to_calibration_future_features() -> None:
    """Calibration features must not leak through the earlier fit normalizer."""
    from dataclasses import replace

    from trade_rl.evaluation.decision_value_dev_study import _HoldQuantityPolicy
    from trade_rl.evaluation.decision_value_feature_transform import (
        apply_feature_transform,
        fit_train_feature_transform,
    )

    raw = signal_market(420, seed=29)
    changed_features = raw.features.copy()
    changed_features[190:241] = changed_features[190:241] * 30.0 + 15.0
    future_changed = replace(raw, features=changed_features, identity_payload_json=None)
    assert np.array_equal(raw.features[:190], future_changed.features[:190])

    models = []
    for source in (raw, future_changed):
        block_transform = fit_train_feature_transform(
            source,
            train_start_index=20,
            train_stop_index=240,
            clip=3.0,
            as_of_index=330,
        )
        view = apply_feature_transform(source, block_transform)
        collection = collect_reachable_state_labels(
            view,
            reference_policy_provider=lambda _t, _start: (
                lambda: _HoldQuantityPolicy(TARGETS)
            ),
            reference_policy_digest=_REF,
            branch_indices=tuple(range(20, 240, _HORIZON)),
            train_start_index=20,
            prefix_bars=12,
            horizon_bars=_HORIZON,
            candidate_actions=_CANDIDATES,
            target_abs_weights=TARGETS,
            initial_capital=100_000.0,
            execution_cost=COST,
            risk_config=RISK_CONFIG,
            fit_cutoff=source.timestamps[330],
            source_scope="synthetic_only",
        )
        captured = []

        def fit(batch):
            model = _fit(batch)
            captured.append(model)
            return model

        select_margin_multiplier(
            dataset=view,
            raw_dataset=source,
            train_start_index=20,
            feature_clip=3.0,
            samples=collection.samples,
            fit_model=fit,
            make_policy=_policy,
            grid=_GRID,
            calibration_fraction=0.25,
            horizon_bars=_HORIZON,
            initial_capital=100_000.0,
            execution_cost=COST,
            risk_config=RISK_CONFIG,
            reference_policy_digest=_REF,
        )
        assert len(captured) == 1
        models.append(captured[0])
    np.testing.assert_array_equal(models[0].coefficients, models[1].coefficients)
