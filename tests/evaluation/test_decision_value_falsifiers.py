"""#1025 TD0/TD3/TD4 falsifiers: plan contract, B2 head, win-rate vs amount, joint cap."""

from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_decision_value_fit import (
    _CANDIDATES,
    _COST,
    _EVALUATE,
    _HOLD,
    _LONG,
    _RISK,
    _TRAIN_FIT,
    _fit_batch,
    _Hold,
    _market,
    _native,
)
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_fit import (
    fit_synthetic_linear_decision_value,
    fit_synthetic_shallow_decision_value,
)
from trade_rl.evaluation.decision_value_plan import DecisionValueStudyPlan
from trade_rl.evaluation.paired_portfolio_action import run_paired_portfolio_action
from trade_rl.evaluation.paired_portfolio_label import (
    build_synthetic_paired_label,
    build_synthetic_paired_label_batch,
)
from trade_rl.strategies.decision_value import LinearDecisionValueRuntimeContext
from trade_rl.strategies.decision_value_shallow import (
    ShallowDecisionValuePolicy,
    load_shallow_decision_value,
    serialize_shallow_decision_value,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
)


def _plan(**overrides: object) -> DecisionValueStudyPlan:
    base: dict[str, object] = dict(
        data_scope="synthetic_only",
        account_mode="shared_portfolio",
        symbols=("AAA", "BBB"),
        initial_capital=1_000.0,
        horizon_bars=7,
        terminal_reserve_bars=1,
        candidate_actions=_CANDIDATES,
        fee_rate=0.001,
        spread_rate=0.0002,
        max_policy_improvements=1,
        learner_roster=("linear", "shallow"),
    )
    base.update(overrides)
    return DecisionValueStudyPlan(**base)  # type: ignore[arg-type]


def test_plan_accepts_full_joint_16_action_roster() -> None:
    codes = (ACTION_HOLD, ACTION_SHORT, ACTION_FLAT, ACTION_LONG)
    roster = tuple(itertools.product(codes, repeat=2))
    assert len(set(roster)) == 16
    assert roster[0] == (ACTION_HOLD, ACTION_HOLD)
    assert _plan(candidate_actions=roster).candidate_actions == roster


@pytest.mark.parametrize(
    "overrides",
    [
        {"fee_rate": 0.0, "spread_rate": 0.0},
        {"initial_capital": 0.0},
        {"initial_capital": float("nan")},
        {"horizon_bars": 1},
        {"terminal_reserve_bars": 7},
        {"account_mode": "independent"},
        {"max_policy_improvements": 2},
        {"candidate_actions": ((ACTION_LONG, ACTION_HOLD), (ACTION_HOLD, ACTION_HOLD))},
        {"learner_roster": ("linear", "linear")},
        {"learner_roster": ("ppo",)},
        {"data_scope": "unused_future"},
    ],
)
def test_plan_rejects_missing_or_unsafe_fields(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _plan(**overrides)


def test_real_development_requires_separate_authorization() -> None:
    with pytest.raises(ValueError, match="authorization"):
        _plan(data_scope="real_development")
    with pytest.raises(ValueError, match="synthetic"):
        _plan(real_scope_authorization_digest="a" * 64)
    ok = _plan(data_scope="real_development", real_scope_authorization_digest="a" * 64)
    assert ok.data_scope == "real_development"


def test_shallow_head_shares_population_and_trades_on_signal_but_not_noise() -> None:
    batch = _fit_batch()
    linear = fit_synthetic_linear_decision_value(batch, candidate_actions=_CANDIDATES)
    shallow = fit_synthetic_shallow_decision_value(batch, candidate_actions=_CANDIDATES)
    # B1/B2 see exactly the same rows, groups, roster and cutoffs.
    for name in (
        "training_rows",
        "overlap_groups",
        "candidate_actions",
        "fit_cutoff_ns",
        "evaluation_start_ns",
        "symbols",
        "execution_policy_digest",
    ):
        assert getattr(linear, name) == getattr(shallow, name)
    totals_learned = 0.0
    totals_cash = 0.0
    for signal, move in ((0.0, 0.0), (0.8, 0.018), (-0.8, -0.018)):
        market = _market(signal=signal, move=move, day="2026-03-03", tag="b2")
        context = LinearDecisionValueRuntimeContext(
            source_scope="synthetic_only",
            feature_config_digest=market.feature_config_digest,
            normalization_digest=market.normalization_digest,
            execution_policy_digest=shallow.execution_policy_digest,
            reference_policy_digest=shallow.reference_policy_digest,
        )
        learned = _native(
            market, ShallowDecisionValuePolicy(shallow, runtime_context=context)
        )
        cash = _native(market, _Hold())
        totals_learned += learned
        totals_cash += cash
        if signal == 0.0:
            assert learned == pytest.approx(cash)  # no trading in fee-only noise
        else:
            assert learned > cash
    assert totals_learned > totals_cash


def test_shallow_artifact_roundtrip_and_tamper_rejection() -> None:
    model = fit_synthetic_shallow_decision_value(
        _fit_batch(), candidate_actions=_CANDIDATES
    )
    text, digest = serialize_shallow_decision_value(model)
    assert load_shallow_decision_value(text, expected_sha256=digest) == model
    with pytest.raises(ValueError, match="digest"):
        load_shallow_decision_value(text, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="hidden"):
        replace(
            model,
            hidden_weights=tuple(tuple(0.0 for _ in r) for r in model.hidden_weights),
        )
    import json

    payload = json.loads(text)
    payload["hidden_weights"] = [[0.0] * len(r) for r in payload["hidden_weights"]]
    forged_text = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    import hashlib

    forged_digest = hashlib.sha256(forged_text.encode()).hexdigest()
    with pytest.raises(ValueError, match="schema"):
        load_shallow_decision_value(forged_text, expected_sha256=forged_digest)


def test_amount_regression_declines_high_win_rate_but_negative_expectancy() -> None:
    # Three of four paths win, but the single loss is larger than all wins together
    # (net of costs). A binary win-rate classifier would enter; signed amounts must not.
    moves = (0.010, 0.010, 0.010, -0.080)
    samples = []
    for i, move in enumerate(moves):
        market = _market(signal=1.0, move=move, day="2026-01-01", tag=f"winrate-{i}")
        pair = run_paired_portfolio_action(
            market,
            reference_policy_factory=_Hold,
            start_index=0,
            stop_index=7,
            intervention_index=1,
            baseline_action=_HOLD,
            candidate_action=_LONG,
            target_abs_weights=(0.4, 0.4),
            initial_capital=1_000.0,
            execution_cost=_COST,
            risk=_RISK,
        )
        assert pair.delta_per_initial_capital is not None
        ledger = pair.baseline_result.ledger_evidence
        assert ledger is not None
        digest = ledger.to_mapping()["execution_policy_digest"]
        assert isinstance(digest, str)
        samples.append(
            build_synthetic_paired_label(
                market,
                pair,
                baseline_action=_HOLD,
                candidate_action=_LONG,
                target_abs_weights=(0.4, 0.4),
                reference_policy_digest="a" * 64,
                expected_execution_policy_digest=digest,
                observation_record_available_at=market.timestamps[1],
                settlement_record_available_at=market.timestamps[7]
                + np.timedelta64(30, "m"),
                fit_cutoff=_TRAIN_FIT,
            )
        )
    wins = sum(1 for s in samples if s.label_delta > 0)
    assert wins == 3 and sum(s.label_delta for s in samples) < 0
    batch = build_synthetic_paired_label_batch(
        tuple(samples),
        fit_cutoff=_TRAIN_FIT,
        evaluation_start=_EVALUATE,
        expected_reference_policy_digest="a" * 64,
    )
    roster = (_HOLD, _LONG)
    for fit in (
        fit_synthetic_linear_decision_value,
        fit_synthetic_shallow_decision_value,
    ):
        model = fit(batch, candidate_actions=roster)
        market = _market(signal=1.0, move=0.0, day="2026-03-03", tag="winrate-eval")
        context = LinearDecisionValueRuntimeContext(
            source_scope="synthetic_only",
            feature_config_digest=market.feature_config_digest,
            normalization_digest=market.normalization_digest,
            execution_policy_digest=model.execution_policy_digest,
            reference_policy_digest=model.reference_policy_digest,
        )
        policy_cls = (
            ShallowDecisionValuePolicy
            if fit is fit_synthetic_shallow_decision_value
            else __import__(
                "trade_rl.strategies.decision_value", fromlist=["x"]
            ).LinearDecisionValuePolicy
        )
        policy = policy_cls(model, runtime_context=context)  # type: ignore[arg-type]
        # Snapshot via a native replay capture of the first decision.
        seen: list[tuple[int, ...]] = []

        class _Spy:
            def decide_portfolio(self, snapshot):  # type: ignore[no-untyped-def]
                if snapshot.index == 1:
                    seen.append(policy.decide_action(snapshot))
                return policy.decide_portfolio(snapshot)

        _native(market, _Spy())
        assert seen == [_HOLD]


def _two_move_market(*, move_a: float, move_b: float, tag: str) -> MarketDataset:
    base = _market(signal=0.0, move=0.0, day="2026-01-01", tag=tag)
    a = np.array([1, 1, 1, 1 + move_a / 3, 1 + 2 * move_a / 3] + [1 + move_a] * 3)
    b = np.array([1, 1, 1, 1 + move_b / 3, 1 + 2 * move_b / 3] + [1 + move_b] * 3)
    closes = np.column_stack((100 * a, 200 * b))
    opens = np.vstack((closes[0], closes[:-1]))
    return replace(
        base,
        open=opens,
        high=np.maximum(opens, closes),
        low=np.minimum(opens, closes),
        close=closes,
    )


def test_joint_action_is_not_the_sum_of_single_symbol_effects_under_shared_cap() -> (
    None
):
    market = _two_move_market(move_a=0.03, move_b=0.03, tag="joint-cap")

    def delta(action: tuple[int, ...]) -> float:
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
        assert pair.delta_per_initial_capital is not None
        return pair.delta_per_initial_capital

    only_a = delta((ACTION_LONG, ACTION_HOLD))
    only_b = delta((ACTION_HOLD, ACTION_LONG))
    both = delta((ACTION_LONG, ACTION_LONG))
    # Gross 0.8 requested against a 0.5 cap: independent Q-values cannot be summed.
    assert only_a > 0 and only_b > 0
    assert both != pytest.approx(only_a + only_b, rel=1e-3)
    assert both < only_a + only_b
