"""Contract and native-integration tests for issue #1025's paired action replay.

All prices, observations and episodes here are artificial. No market fit is run.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label,
    build_synthetic_paired_label_batch,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
)

_FEE = 0.001
_SPREAD = 0.0002
_CAPITAL = 1_000.0
_TARGET_WEIGHT = 0.4
_COST = replace(ExecutionCostConfig.zero(), fee_rate=_FEE, spread_rate=_SPREAD)
_RISK = PreTradeRisk(
    PreTradeRiskConfig(
        max_gross=0.5,
        max_abs_weight=0.5,
        max_turnover=None,
        drawdown_start=0.20,
        drawdown_stop=0.20,
    )
)


def _dataset(*, rise: float = 10.0) -> MarketDataset:
    # The entry at decision 0 fills at row 1 for only ONE AAA unit.
    # Decision 1 sees a genuine position AND an unfinished entry order.
    close = np.array(
        [
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0, 200.0],
            [100.0 + rise, 200.0],
            [100.0 + rise, 200.0],
            [100.0 + rise, 200.0],
            [100.0 + rise, 200.0],
            [100.0 + rise, 200.0],
        ],
        dtype=np.float64,
    )
    opening = np.vstack((close[0], close[:-1]))
    volume = np.full_like(close, 1_000_000.0)
    volume[1, 0] = 1.0
    return MarketDataset(
        dataset_id="1" * 64,
        symbols=("AAA", "BBB"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(len(close)) * np.timedelta64(1, "h"),
        features=np.zeros((len(close), 2, 1), dtype=np.float32),
        global_features=np.zeros((len(close), 1), dtype=np.float32),
        open=opening,
        high=np.maximum(opening, close),
        low=np.minimum(opening, close),
        close=close,
        volume=volume,
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((len(close), 2, 1), dtype=np.bool_),
        feature_names=("fixed_observation",),
        global_feature_names=("fixed_global",),
        periods_per_year=8_760,
    )


class _EnterThenHold:
    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        if snapshot.index == 0:
            return PortfolioTargetProposal((_TARGET_WEIGHT, 0.0))
        return PortfolioTargetProposal(
            snapshot.account.weights,
            hold_current_quantity_symbols=(0, 1),
        )


def _paired(
    *,
    market: MarketDataset | None = None,
    baseline: tuple[int, int] = (ACTION_HOLD, ACTION_HOLD),
    candidate: tuple[int, int] = (ACTION_FLAT, ACTION_HOLD),
    factory: object = _EnterThenHold,
    execution_cost: ExecutionCostConfig = _COST,
) -> object:
    return run_paired_portfolio_action(
        _dataset() if market is None else market,
        reference_policy_factory=factory,
        start_index=0,
        stop_index=7,
        intervention_index=1,
        baseline_action=baseline,
        candidate_action=candidate,
        target_abs_weights=(_TARGET_WEIGHT, _TARGET_WEIGHT),
        initial_capital=_CAPITAL,
        execution_cost=execution_cost,
        risk=_RISK,
    )


def test_pending_prefix_and_native_money_oracle() -> None:
    paired = _paired()
    before = paired.baseline_snapshot
    assert before == paired.candidate_snapshot
    assert before.index == 1
    assert before.account.exact_quantities == ("1", "0")
    assert len(before.active_orders) == 1
    assert before.active_orders[0].symbol_index == 0
    assert before.active_orders[0].remaining_quantity == pytest.approx(3.0)

    assert paired.baseline_result.ledger_evidence is not None
    assert paired.candidate_result.ledger_evidence is not None
    ledger_a = paired.baseline_result.ledger_evidence.to_mapping()
    ledger_b = paired.candidate_result.ledger_evidence.to_mapping()
    assert ledger_a["intervals"][:1] == ledger_b["intervals"][:1]
    assert len(ledger_a["intervals"]) == len(ledger_b["intervals"]) == 7
    assert paired.baseline_result.book.exact_quantities == (0, 0)
    assert paired.candidate_result.book.exact_quantities == (0, 0)
    assert paired.eligible_for_label

    # Independent hand money check (no production account formula used):
    # entry: +1 AAA at 100; fee+spread = 100 * .0012 = .12;
    # baseline closes 1 at 110: proceeds 110, cost .132;
    # candidate flattens at 100: proceeds 100, cost .12.
    entry_cost = 100.0 * (_FEE + _SPREAD)
    baseline_close_cost = 110.0 * (_FEE + _SPREAD)
    candidate_close_cost = 100.0 * (_FEE + _SPREAD)
    expected_baseline = _CAPITAL - 100.0 - entry_cost + 110.0 - baseline_close_cost
    expected_candidate = _CAPITAL - entry_cost - candidate_close_cost
    assert paired.baseline_result.book.portfolio_value == pytest.approx(
        expected_baseline, abs=1e-8
    )
    assert paired.candidate_result.book.portfolio_value == pytest.approx(
        expected_candidate, abs=1e-8
    )
    assert paired.delta_per_initial_capital == pytest.approx(
        (expected_candidate - expected_baseline) / _CAPITAL, abs=1e-10
    )
    assert paired.delta_per_initial_capital < 0.0


def test_null_intervention_and_branch_order_invariance() -> None:
    hold = (ACTION_HOLD, ACTION_HOLD)
    flat = (ACTION_FLAT, ACTION_HOLD)
    null = _paired(baseline=hold, candidate=hold)
    assert null.delta_per_initial_capital == pytest.approx(0.0, abs=1e-12)
    assert (
        null.baseline_result.ledger_evidence.to_mapping()
        == null.candidate_result.ledger_evidence.to_mapping()
    )

    forward = _paired(baseline=hold, candidate=flat)
    reversed_pair = _paired(baseline=flat, candidate=hold)
    assert forward.delta_per_initial_capital == pytest.approx(
        -reversed_pair.delta_per_initial_capital, abs=1e-12
    )
    assert (
        forward.candidate_result.book.portfolio_value
        == reversed_pair.baseline_result.book.portfolio_value
    )
    assert (
        forward.baseline_result.book.portfolio_value
        == reversed_pair.candidate_result.book.portfolio_value
    )


def test_future_price_change_preserves_predecision_but_changes_label() -> None:
    up = _paired(market=_dataset(rise=10.0))
    down = _paired(market=_dataset(rise=-10.0))
    assert up.baseline_snapshot == down.baseline_snapshot
    assert up.candidate_snapshot == down.candidate_snapshot
    assert up.delta_per_initial_capital < 0.0
    assert down.delta_per_initial_capital > 0.0


def test_mismatched_reference_prefix_is_rejected() -> None:
    calls = 0

    def unsafe_factory() -> _EnterThenHold:
        nonlocal calls
        calls += 1

        # Capture the branch-specific choice, not a mutable free variable.
        branch = calls

        class _Fixed(_EnterThenHold):
            def decide_portfolio(
                self, snapshot: PortfolioDecisionSnapshot
            ) -> PortfolioTargetProposal:
                if snapshot.index == 0 and branch == 2:
                    return PortfolioTargetProposal((0.0, 0.0))
                return super().decide_portfolio(snapshot)

        return _Fixed()

    with pytest.raises(RuntimeError, match="prefix"):
        _paired(factory=unsafe_factory)


def test_preflight_rejects_invalid_actions_and_stochastic_execution() -> None:
    calls: list[int] = []

    def factory() -> _EnterThenHold:
        calls.append(1)
        return _EnterThenHold()

    with pytest.raises(ValueError, match="exactly one code"):
        _paired(factory=factory, candidate=(ACTION_FLAT,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="stochastic"):
        _paired(
            factory=factory,
            execution_cost=replace(_COST, slippage_std=0.02),
        )
    assert not calls


def test_reference_factory_must_return_independent_actors() -> None:
    actor = _EnterThenHold()
    with pytest.raises(ValueError, match="distinct"):
        _paired(factory=lambda: actor)


def test_unsettled_terminal_keeps_raw_arms_but_blocks_label() -> None:
    market = _dataset()
    volume = np.asarray(market.volume).copy()
    volume[-1, 0] = 0.0  # no capacity to liquidate the held AAA unit
    blocked = replace(market, volume=volume)
    paired = _paired(market=blocked)

    assert not paired.eligible_for_label
    assert paired.delta_per_initial_capital is None
    assert paired.baseline_result.book.exact_quantities != (0, 0)
    assert paired.candidate_result.book.exact_quantities == (0, 0)
    assert paired.baseline_result.ledger_evidence is not None
    retained = paired.baseline_result.ledger_evidence.to_mapping()
    assert retained["active_order_remainders"]


# TD2: labels are post-settlement observations, not as-of-decision features.
# Synthetic-only scope deliberately cannot authorize real-data training.


def _sample(
    *,
    market: MarketDataset | None = None,
    paired: object | None = None,
    observation_record_available_at: np.datetime64 = np.datetime64(
        "2026-01-01T01:00:00", "ns"
    ),
    settlement_record_available_at: np.datetime64 = np.datetime64(
        "2026-01-01T07:30:00", "ns"
    ),
    fit_cutoff: np.datetime64 = np.datetime64("2026-01-01T10:00:00", "ns"),
    reference_policy_digest: str = "a" * 64,
    expected_execution_policy_digest: str | None = None,
    baseline_action: tuple[int, int] = (ACTION_HOLD, ACTION_HOLD),
    candidate_action: tuple[int, int] = (ACTION_FLAT, ACTION_HOLD),
):
    market = _dataset() if market is None else market
    paired = _paired(market=market) if paired is None else paired
    evidence = paired.baseline_result.ledger_evidence
    assert evidence is not None
    digest = evidence.to_mapping()["execution_policy_digest"]
    assert isinstance(digest, str)
    return build_synthetic_paired_label(
        market,
        paired,
        baseline_action=baseline_action,
        candidate_action=candidate_action,
        target_abs_weights=(_TARGET_WEIGHT, _TARGET_WEIGHT),
        reference_policy_digest=reference_policy_digest,
        expected_execution_policy_digest=(
            digest
            if expected_execution_policy_digest is None
            else expected_execution_policy_digest
        ),
        observation_record_available_at=observation_record_available_at,
        settlement_record_available_at=settlement_record_available_at,
        fit_cutoff=fit_cutoff,
    )


def test_synthetic_label_is_frozen_post_settlement_and_excludes_future_paths() -> None:
    market = _dataset()
    pair = _paired(market=market)
    sample = _sample(market=market, paired=pair)
    assert sample.source_scope == "synthetic_only"
    assert sample.dataset_id == market.dataset_id
    assert sample.decision_time_ns == int(market.timestamps[1].astype("int64"))
    assert sample.terminal_time_ns == int(market.timestamps[7].astype("int64"))
    assert sample.label_first_available_ns == int(
        np.datetime64("2026-01-01T07:30:00", "ns").astype("int64")
    )
    assert sample.fit_cutoff_ns > sample.label_first_available_ns
    assert sample.observation == pair.baseline_snapshot
    assert sample.label_delta == pytest.approx(pair.delta_per_initial_capital)
    assert not hasattr(sample, "baseline_result")
    assert not hasattr(sample, "candidate_result")
    assert not hasattr(sample, "market_price_path")

    # Mutating a *future* market path changes a training label, not its inputs.
    down = replace(_dataset(rise=-10), dataset_id="2" * 64)
    changed = _sample(market=down)
    assert sample.observation == changed.observation
    assert sample.label_delta * changed.label_delta < 0.0


@pytest.mark.parametrize(
    "bad_cutoff",
    ["2026-01-01T07:30:00", "2026-01-01T07:00:00"],
)
def test_label_must_mature_strictly_before_fit(bad_cutoff: str) -> None:
    with pytest.raises(ValueError, match="matur|cutoff"):
        _sample(fit_cutoff=np.datetime64(bad_cutoff, "ns"))


def test_late_settlement_receipt_blocks_preceding_fit() -> None:
    with pytest.raises(ValueError, match="matur|cutoff"):
        _sample(
            settlement_record_available_at=np.datetime64("2026-01-01T12:00:00", "ns")
        )


def test_observation_arriving_after_decision_is_not_a_causal_feature() -> None:
    with pytest.raises(ValueError, match="observation|decision"):
        _sample(
            observation_record_available_at=np.datetime64("2026-01-01T02:00:00", "ns")
        )


def test_untrusted_lineage_or_forged_pair_cannot_form_label() -> None:
    with pytest.raises(ValueError, match="execution_policy_digest"):
        _sample(expected_execution_policy_digest="b" * 64)
    with pytest.raises(ValueError, match="reference_policy_digest"):
        _sample(reference_policy_digest="not a digest")

    pair = _paired()
    forged = replace(pair, delta_per_initial_capital=999.0)
    with pytest.raises(ValueError, match="delta|ledger"):
        _sample(paired=forged)

    forged_snapshot = replace(
        pair,
        baseline_snapshot=replace(
            pair.baseline_snapshot,
            timestamp_ns=pair.baseline_snapshot.timestamp_ns + 1,
        ),
    )
    with pytest.raises(ValueError, match="snapshot|decision"):
        _sample(paired=forged_snapshot)


def test_invalid_settlement_never_becomes_fit_sample() -> None:
    market = _dataset()
    volume = market.volume.copy()
    volume[-1, 0] = 0.0
    blocked = replace(market, volume=volume)
    pair = _paired(market=blocked)
    assert not pair.eligible_for_label
    with pytest.raises(ValueError, match="settled|eligible"):
        _sample(market=blocked, paired=pair)


def test_label_batch_groups_overlapping_paths_and_blocks() -> None:
    cutoff = np.datetime64("2026-03-01T00:00:00", "ns")
    first = _sample(fit_cutoff=cutoff)
    second_market = replace(_dataset(rise=-10), dataset_id="2" * 64)
    second = _sample(market=second_market, fit_cutoff=cutoff)
    third_market = replace(
        _dataset(),
        dataset_id="3" * 64,
        timestamps=_dataset().timestamps + np.timedelta64(31, "D"),
    )
    third = _sample(
        market=third_market,
        observation_record_available_at=np.datetime64("2026-02-01T01:00:00", "ns"),
        settlement_record_available_at=np.datetime64("2026-02-01T07:30:00", "ns"),
        fit_cutoff=cutoff,
    )
    batch = build_synthetic_paired_label_batch(
        (third, first, second),
        fit_cutoff=cutoff,
        evaluation_start=np.datetime64("2026-03-02T00:00:00", "ns"),
        expected_reference_policy_digest="a" * 64,
    )
    assert batch.group_ids == (0, 0, 1)
    assert batch.samples[0].decision_time_ns <= batch.samples[1].decision_time_ns
    assert batch.samples[1].label_first_available_ns < cutoff.astype("int64")
    assert len(batch.samples) == 3

    with pytest.raises(ValueError, match="duplicate"):
        build_synthetic_paired_label_batch(
            (first, first),
            fit_cutoff=cutoff,
            evaluation_start=np.datetime64("2026-03-02", "ns"),
            expected_reference_policy_digest="a" * 64,
        )
    with pytest.raises(ValueError, match="reference_policy_digest"):
        build_synthetic_paired_label_batch(
            (first, replace(second, reference_policy_digest="b" * 64)),
            fit_cutoff=cutoff,
            evaluation_start=np.datetime64("2026-03-02", "ns"),
            expected_reference_policy_digest="a" * 64,
        )


def test_synthetic_scope_cannot_be_relabelled_as_market_evidence() -> None:
    sample = _sample()
    with pytest.raises(ValueError, match="synthetic"):
        build_synthetic_paired_label_batch(
            (replace(sample, source_scope="real_market"),),
            fit_cutoff=np.datetime64("2026-01-01T10:00:00", "ns"),
            evaluation_start=np.datetime64("2026-01-02T00:00:00", "ns"),
            expected_reference_policy_digest="a" * 64,
        )
