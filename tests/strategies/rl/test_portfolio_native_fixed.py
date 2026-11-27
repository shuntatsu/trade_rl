"""Exact native fixed-clock Gym, training, and inference boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tests.strategies.rl.test_portfolio_env import _bound_clock, _risk
from tests.strategies.rl.test_portfolio_inference_bundle import (
    _dataset,
    _load,
    _publish,
)
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.evaluation.native_fixed_returns import (
    load_native_fixed_return_payload,
    native_fixed_return_payload,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.objectives import BoundObjectiveClock
from trade_rl.strategies.rl import portfolio_inference_bundle, portfolio_training
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    SharedPortfolioGymEnv,
)
from trade_rl.strategies.rl.portfolio_runtime_scenario import (
    PortfolioRuntimeExecutionScenarioV2,
)
from trade_rl.strategies.rl.portfolio_scenario_inference import (
    load_portfolio_scenario_inference_bundle,
)
from trade_rl.strategies.rl.portfolio_scenario_runtime import (
    capture_portfolio_scenario_inference_source_digest,
)
from trade_rl.strategies.rl.portfolio_training import (
    SharedPortfolioPPOConfig,
    fit_shared_portfolio_ppo,
)
from trade_rl.strategies.rl.portfolio_transform import (
    fit_portfolio_observation_transform,
)

_START = datetime(2026, 1, 5, tzinfo=UTC)


def _native_dataset(days: int = 7, *, n_bars: int = 8) -> MarketDataset:
    base = _dataset(n_bars=n_bars)
    first = np.datetime64("2026-01-05", "ns")
    times = first + np.arange(n_bars) * np.timedelta64(days, "D")
    clock = (
        NativeBarClock(NativeClockKind.UTC_WEEK)
        if days == 7
        else NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, first)
    )
    return replace(
        base,
        timestamps=times,
        available_at=np.broadcast_to(times[:, None], (n_bars, 2)).copy(),
        periods_per_year=Fraction(365, days),
        native_clock=clock,
        funding_event_count=np.zeros((n_bars, 2), dtype=np.int32),
        funding_due=np.zeros((n_bars, 2), dtype=np.bool_),
        funding_price_rate=np.zeros((n_bars, 2)),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "native-fixed-portfolio"})


def _cost() -> ExecutionCostConfig:
    return replace(
        ExecutionCostConfig.zero(),
        allow_short=False,
        processing_bar_volume_capacity=False,
        fee_rate=0.01,
    )


def _environment(
    dataset: MarketDataset,
    *,
    days: int = 7,
    start: int = 0,
    stop: int = 5,
    cost: ExecutionCostConfig | None = None,
    weights: tuple[float, float] = (0.5, 0.5),
) -> SharedPortfolioGymEnv:
    base = _bound_clock(start_index=start, stop_index=stop)
    bound = BoundObjectiveClock(
        replace(
            base.objective,
            evaluation_start=_START + timedelta(days=days * start),
            evaluation_stop_exclusive=_START + timedelta(days=days * stop),
        ),
        replace(
            base.clock,
            decision_interval_seconds=days * 86400,
            execution_interval_seconds=days * 86400,
            reward_interval_seconds=days * 86400,
            economic_horizon_seconds=days * 86400 * (stop - start),
            rollout_steps=4,
        ),
    )
    return SharedPortfolioGymEnv(
        dataset,
        start_index=start,
        stop_index=stop,
        bound=bound,
        target_abs_weights=weights,
        execution_cost=cost or _cost(),
        risk_config=_risk(),
    )


def _config() -> SharedPortfolioPPOConfig:
    return SharedPortfolioPPOConfig(
        total_timesteps=8,
        n_steps=4,
        batch_size=2,
        n_epochs=1,
        reset_seed=29,
        ppo_seed=31,
    )


def _replace_evidence(dataset: MarketDataset, **values: object) -> MarketDataset:
    return replace(dataset, identity_payload_json=None, **values).with_content_identity(
        {"fixture": "native-fixed-portfolio-mutated"}
    )


def test_native_gym_rejects_unverified_dataset_before_reset() -> None:
    dataset = replace(_native_dataset(), identity_payload_json=None)
    assert not dataset.identity_verified
    with pytest.raises(ValueError, match="verified Dataset identity"):
        _environment(dataset)


@pytest.mark.parametrize(
    "changes",
    [
        {"allow_short": True},
        {"processing_bar_volume_capacity": True},
        {"order_type": "limit"},
        {"max_leverage": 2.0},
        {"slippage_std": 0.01},
        {"collateral_haircut": 0.9},
    ],
)
def test_native_gym_rejects_unsupported_execution_before_reset(changes) -> None:
    with pytest.raises(ValueError, match="native fixed execution requires"):
        _environment(_native_dataset(), cost=replace(_cost(), **changes))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"margin_mode": "isolated"}, "isolated margin"),
        ({"max_leverage": 0.5}, "risk exposure limits"),
    ],
)
def test_native_gym_rejects_incompatible_roster_and_risk_before_reset(changes, message):
    with pytest.raises(ValueError, match=message):
        _environment(_native_dataset(), cost=replace(_cost(), **changes))


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"margin_mode": "isolated"}, "isolated margin"),
        ({"max_leverage": 0.5}, "risk exposure limits"),
    ],
)
def test_native_training_revalidates_incompatible_execution_before_import(
    tmp_path, monkeypatch, changes, message
):
    env = _environment(_native_dataset())
    transform = fit_portfolio_observation_transform(env)
    env.execution_cost = replace(env.execution_cost, **changes)
    attempt = tmp_path / "attempt"

    def forbidden(*args, **kwargs):
        pytest.fail("incompatible native execution reached optional runtime import")

    monkeypatch.setattr(portfolio_training.importlib, "import_module", forbidden)
    with pytest.raises(ValueError, match=message):
        fit_shared_portfolio_ppo(
            env, transform=transform, output_root=attempt, config=_config()
        )
    assert not attempt.exists() and env._session is None


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("late", "unavailable"),
        ("unavailable", "unavailable"),
        ("funding", "unsupported funding"),
        ("gap", "unsupported funding"),
        ("split", "unsupported funding"),
        ("dividend", "unsupported funding"),
        ("negative_carry", "unsupported funding"),
    ],
)
def test_native_gym_preflights_even_the_terminal_processing_row(mutation, message):
    dataset = _native_dataset()
    row = 5  # Excluded from fitting and actor calls but included in settlement.
    if mutation == "late":
        values = dataset.resolved_array("available_at").copy()
        values[row, 0] += np.timedelta64(1, "ns")
        information = dataset.resolved_array("information_available").copy()
        information[row, 0] = False
        dataset = _replace_evidence(
            dataset, available_at=values, information_available=information
        )
    elif mutation == "unavailable":
        values = dataset.resolved_array("information_available").copy()
        values[row, 0] = False
        dataset = _replace_evidence(dataset, information_available=values)
    elif mutation == "funding":
        count = dataset.resolved_array("funding_event_count").copy()
        due = dataset.resolved_array("funding_due").copy()
        count[row, 0] = 1
        due[row, 0] = True
        dataset = _replace_evidence(dataset, funding_event_count=count, funding_due=due)
    else:
        field, value = {
            "gap": ("tradable", False),
            "split": ("split_factor", 2.0),
            "dividend": ("dividend", 1.0),
            "negative_carry": ("cash_rate", -0.01),
        }[mutation]
        values = dataset.resolved_array(field).copy()
        values[row] = value
        dataset = _replace_evidence(dataset, **{field: values})
    with pytest.raises(ValueError, match=message):
        _environment(dataset)


def test_native_training_revalidates_replaced_dataset_before_attempt_or_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _environment(_native_dataset())
    transform = fit_portfolio_observation_transform(env)
    # A copied ID with no verified identity is not the original immutable source.
    env.dataset = replace(env.dataset, identity_payload_json=None)
    output = tmp_path / "attempt"

    def forbidden(*args, **kwargs):
        pytest.fail("unverified native Dataset reached optional runtime import")

    monkeypatch.setattr(portfolio_training.importlib, "import_module", forbidden)
    with pytest.raises(ValueError, match="verified Dataset identity"):
        fit_shared_portfolio_ppo(
            env, transform=transform, output_root=output, config=_config()
        )
    assert not output.exists() and env._session is None


@pytest.mark.parametrize("days", [3, 7])
def test_actual_native_fixed_ppo_bundle_and_native_replay(
    tmp_path: Path, days: int, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _native_dataset(days)
    env = _environment(dataset, days=days)
    transform = fit_portfolio_observation_transform(env)
    result = fit_shared_portfolio_ppo(
        env, transform=transform, output_root=tmp_path / "attempt", config=_config()
    )
    assert result.realized_timesteps == 8 and result.rollout_count == 2
    context = json.loads((result.output_root / "training_context.json").read_bytes())
    assert context["context"]["dataset_id"] == dataset.dataset_id
    assert context["financial_clock"]["decision_interval_seconds"] == days * 86400
    publication = _publish(result, tmp_path / "bundle")
    runtime = _environment(dataset, days=days, start=5, stop=7)
    policy = _load(publication, runtime)
    raw, _ = runtime.reset(seed=43)
    action = policy.predict(raw)
    assert runtime.action_space.contains(action)
    _, reward, done, truncated, info = runtime.step(action)
    assert done and not truncated and info["settlement_complete"]
    assert info["elapsed_intervals"] == 2
    replay = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=5,
        stop_index=7,
        initial_capital=1000.0,
        execution_cost=_cost(),
        risk=PreTradeRisk(_risk()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert replay.book.portfolio_value == pytest.approx(1000.0 * (1 + reward))
    assert all(q == 0 for q in replay.book.exact_quantities)
    assert replay.book.cash >= 0 and replay.returns.periods_per_year == Fraction(
        365, days
    )
    payload = native_fixed_return_payload(dataset, replay.returns, start_index=5)
    assert load_native_fixed_return_payload(dataset, payload) == replay.returns
    assert payload["elapsed_years"] == {"numerator": 2 * days, "denominator": 365}

    scenario = PortfolioRuntimeExecutionScenarioV2(
        scenario_id="base-native-fixed",
        role="base",
        scenario_plan_sha256="e" * 64,
        training_context_sha256=result.training_context_sha256,
        transform_digest=context["transform_digest"],
        checkpoint_sha256=result.final_checkpoint_sha256,
        base_execution_cost=_cost(),
        runtime_execution_cost=_cost(),
        changed_fields=(),
    )

    def load_scenario(runtime_env):
        return load_portfolio_scenario_inference_bundle(
            publication.root,
            expected_bundle_sha256=publication.bundle_sha256,
            expected_manifest_sha256=publication.training_manifest_sha256,
            expected_completion_sha256=publication.training_completion_sha256,
            expected_training_context_sha256=publication.training_context_sha256,
            expected_checkpoint_sha256=publication.checkpoint_sha256,
            expected_scenario=scenario,
            expected_scenario_sha256=scenario.digest,
            expected_scenario_inference_source_sha256=capture_portfolio_scenario_inference_source_digest(),
            runtime_env=runtime_env,
        )

    scenario_runtime = _environment(dataset, days=days, start=5, stop=7)
    scenario_policy = load_scenario(scenario_runtime)
    scenario_raw, _ = scenario_runtime.reset(seed=43)
    np.testing.assert_array_equal(scenario_policy.predict(scenario_raw), action)

    def forbidden(*args, **kwargs):
        pytest.fail("unverified native Dataset reached PPO deserialization")

    monkeypatch.setattr(
        portfolio_inference_bundle, "_load_policy_from_verified_path", forbidden
    )
    # The bundle loader is a second admission boundary: an environment may
    # have been changed since construction, even when its copied ID matches.
    runtime = _environment(dataset, days=days, start=5, stop=7)
    runtime.dataset = replace(dataset, identity_payload_json=None)
    with pytest.raises(ValueError, match="verified Dataset identity"):
        _load(publication, runtime)
    with pytest.raises(ValueError, match="verified Dataset identity"):
        load_scenario(runtime)
    assert runtime._session is None


def test_native_window_checks_only_processed_economic_rows() -> None:
    dataset = _native_dataset()
    # The first row supplies the decision mark, not a processed cash flow.
    # Rows after the reserved settlement are outside the declared window.
    counts = dataset.resolved_array("funding_event_count").copy()
    due = dataset.resolved_array("funding_due").copy()
    counts[[0, 6], 0] = 1
    due[[0, 6], 0] = True
    dataset = _replace_evidence(dataset, funding_event_count=counts, funding_due=due)
    env = _environment(dataset)
    env.reset()
    for _ in range(4):
        _, _, done, truncated, info = env.step(np.asarray([ACTION_HOLD, ACTION_HOLD]))
    assert done and not truncated and info["settlement_complete"]


@pytest.mark.parametrize("days", [3, 7])
def test_native_gym_literal_long_only_fees_and_terminal_reward(days: int) -> None:
    env = _environment(_native_dataset(days), days=days, weights=(0.2, 0.2))
    env.reset()
    rewards = []
    for i in range(4):
        codes = (ACTION_LONG, ACTION_SHORT) if i == 0 else (ACTION_HOLD, ACTION_HOLD)
        _, reward, done, truncated, info = env.step(np.asarray(codes, dtype=np.int64))
        rewards.append(reward)
        assert not truncated
        assert env._session is not None
        book = env._session.book
        if i < 3:
            assert not done
            assert book.exact_quantities == (Fraction(2), Fraction(0))
            assert book.cash == pytest.approx(798.0)
        else:
            assert done and info["settlement_complete"]
            assert book.exact_quantities == (Fraction(0), Fraction(0))
            assert book.cash == pytest.approx(996.0)
    # Two units bought and sold at 100, one percent fee on both fills.
    assert sum(rewards) == pytest.approx(-4 / 1000)
