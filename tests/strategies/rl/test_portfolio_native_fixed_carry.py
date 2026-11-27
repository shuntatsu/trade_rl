"""Generated signed fixed-grid PPO uses the declared carry execution identity."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.rl.test_portfolio_env import _bound_clock, _risk
from tests.strategies.rl.test_portfolio_inference_bundle import _load, _publish
from tests.strategies.rl.test_portfolio_native_fixed import (
    _START,
    _config,
    _native_dataset,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.contracts import VolumeUnit
from trade_rl.evaluation.portfolio_inference import (
    run_published_portfolio_inference_replay,
)
from trade_rl.simulation import ExecutionCostConfig, runtime_execution_policy_digest
from trade_rl.simulation.objectives import BoundObjectiveClock
from trade_rl.strategies.rl import portfolio_inference_bundle
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    SharedPortfolioGymEnv,
)
from trade_rl.strategies.rl.portfolio_training import fit_shared_portfolio_ppo
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioObservationRuntimeBinding,
    PortfolioObservationTransform,
    fit_portfolio_observation_transform,
)


def _dataset(days=7, *, funding=True):
    base = _native_dataset(days, n_bars=12)
    rates = np.full((12, 2), 0.01 if funding else 0.0)
    counts = (rates != 0).astype(np.int32)
    return replace(
        base,
        volume_units=(VolumeUnit.CONTRACTS, VolumeUnit.CONTRACTS),
        contract_multipliers=np.array([2.0, 4.0]),
        borrow_rate=np.full((12, 2), 0.365),
        funding_rate=rates,
        funding_event_count=counts,
        funding_due=counts > 0,
        funding_price_rate=rates * np.array([110.0, 200.0]),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "fixed-carry-ppo"})


def _cost(*, allow_short=True):
    return replace(
        ExecutionCostConfig.zero(),
        allow_short=allow_short,
        fee_rate=0.001,
        borrow_rate_multiplier=1.0,
        processing_bar_volume_capacity=False,
        partial_fill_carry=False,
    )


def _environment(dataset, *, days=7, start=0, stop=5, mode=True, cost=None):
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
        target_abs_weights=(0.25, 0.25),
        execution_cost=_cost() if cost is None else cost,
        risk_config=_risk(),
        native_fixed_linear_carry=mode,
    )


@pytest.mark.parametrize("days", [3, 7])
def test_signed_gym_reward_matches_literal_carry_oracle(days):
    env = _environment(_dataset(days), days=days)
    transform = fit_portfolio_observation_transform(env)
    assert transform.native_fixed_linear_carry
    env.reset(seed=42)
    _, reward, done, truncated, _ = env.step(np.array([ACTION_LONG, ACTION_SHORT]))
    assert not done and not truncated
    assert env._session is not None
    assert env._session.book.exact_quantities == (Fraction(5, 4), Fraction(-5, 8))
    # 250 long + 250 short, fee 0.5, funding -2.75+5, borrow 0.25*days.
    assert reward == pytest.approx((1.75 - days / 4) / 1000)
    rewards = [reward]
    while not done:
        _, reward, done, truncated, info = env.step(
            np.array([ACTION_HOLD, ACTION_HOLD])
        )
        rewards.append(reward)
    assert not truncated and info["settlement_complete"]
    expected = Fraction(1000) - 1 + 4 * Fraction(9, 4) - days
    assert env._session.book.exact_quantities == (0, 0)
    assert env._session.book.cash == pytest.approx(float(expected))
    assert sum(rewards) == pytest.approx(float(expected / 1000 - 1))
    assert env._session.execution_policy_digest == runtime_execution_policy_digest(
        env.execution_cost,
        funding_price_rate_present=True,
        native_fixed_linear_carry=True,
    )


def test_signed_transform_identity_and_mode_mismatch():
    dataset = _dataset(funding=False)
    cost = _cost(allow_short=False)
    signed = _environment(dataset, cost=cost)
    spot = _environment(dataset, cost=cost, mode=False)
    a = fit_portfolio_observation_transform(signed)
    b = fit_portfolio_observation_transform(spot)
    assert a.native_fixed_linear_carry and not b.native_fixed_linear_carry
    assert a.digest != b.digest
    assert a.to_payload()["schema"] != b.to_payload()["schema"]
    assert a.to_payload() | {"schema": b.to_payload()["schema"]} == b.to_payload()
    assert (
        PortfolioObservationTransform.from_payload(
            a.to_payload(), expected_digest=a.digest
        )
        == a
    )
    changed = a.to_payload() | {"native_interval_seconds": None}
    with pytest.raises(ValueError):
        PortfolioObservationTransform.from_payload(
            changed, expected_digest=content_digest(changed)
        )
    with pytest.raises(ValueError, match="mode|context"):
        a.validate_environment(spot)
    with pytest.raises(ValueError, match="mode|context"):
        b.validate_environment(signed)
    runtime = _environment(dataset, start=5, stop=10, cost=cost, mode=False)
    with pytest.raises(ValueError, match="mode|context"):
        PortfolioObservationRuntimeBinding.create(a, signed.bound, runtime)


@pytest.mark.parametrize(
    "mutation",
    ["spot_units", "legacy", "late_terminal", "multi_event", "borrow_unavailable"],
)
def test_signed_gym_refuses_bad_windows(mutation):
    dataset = _dataset()
    if mutation == "spot_units":
        dataset = replace(
            dataset,
            volume_units=(VolumeUnit.BASE_ASSET,) * 2,
            identity_payload_json=None,
        ).with_content_identity()
    elif mutation == "legacy":
        dataset = replace(
            dataset,
            native_clock=None,
            periods_per_year=365,
            identity_payload_json=None,
            timestamps=np.datetime64("2026-01-05", "ns")
            + np.arange(12) * np.timedelta64(1, "D"),
            available_at=None,
        ).with_content_identity()
    else:
        if mutation == "late_terminal":
            array = dataset.available_at.copy()
            array[5, 0] += np.timedelta64(1, "ns")
            information = dataset.information_available.copy()
            information[5, 0] = False
            changes = {"available_at": array, "information_available": information}
        elif mutation == "multi_event":
            array = dataset.funding_event_count.copy()
            array[5, 0] = 2
            changes = {"funding_event_count": array}
        else:
            array = dataset.borrow_available.copy()
            array[5, 0] = False
            changes = {"borrow_available": array}
        dataset = replace(
            dataset, identity_payload_json=None, **changes
        ).with_content_identity()
    with pytest.raises(ValueError):
        _environment(dataset, days=1 if mutation == "legacy" else 7)


def test_signed_reset_rechecks_objective_before_session():
    env = _environment(_dataset())
    env.bound = BoundObjectiveClock(
        replace(env.bound.objective, terminal_valuation="marked_continuation"),
        env.bound.clock,
    )
    with pytest.raises(ValueError, match="linear carry"):
        env.reset()
    assert env._session is None


@pytest.mark.parametrize("signed_economics", [False, True])
def test_actual_signed_ppo_bundle_and_paired_replay(tmp_path, signed_economics):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _dataset(funding=signed_economics)
    # One case really trains with signed economics; the other admits identical
    # long-only cost in both modes to prove the identity check independently.
    cost = _cost(allow_short=signed_economics)
    env = _environment(dataset, cost=cost)
    transform = fit_portfolio_observation_transform(env)
    trained = fit_shared_portfolio_ppo(
        env, transform=transform, output_root=tmp_path / "attempt", config=_config()
    )
    assert trained.realized_timesteps == 8
    context = json.loads((trained.output_root / "training_context.json").read_bytes())
    assert (
        context["schema"] == "shared_portfolio_fixed_linear_carry_training_context_v1"
    )
    manifest = json.loads((trained.output_root / "manifest.json").read_bytes())
    wrong = context | {"schema": "shared_portfolio_training_context_v1"}
    with pytest.raises(ValueError, match="mode|transform"):
        portfolio_inference_bundle._validated_training_context(wrong, manifest)
    bundle = _publish(trained, tmp_path / "bundle")
    if not signed_economics:
        spot = _environment(dataset, start=5, stop=10, cost=cost, mode=False)
        with pytest.raises(ValueError, match="mode|context"):
            _load(bundle, spot)
    runtime = _environment(dataset, start=5, stop=10, cost=cost)
    policy = _load(bundle, runtime)
    raw, _ = runtime.reset(seed=42)
    done = False
    rewards = []
    while not done:
        raw, reward, done, truncated, info = runtime.step(policy.predict(raw))
        rewards.append(reward)
    assert not truncated and info["settlement_complete"]
    artifact = publish_market_dataset_artifact(tmp_path / "dataset", dataset)
    _, replay = run_published_portfolio_inference_replay(
        dataset_artifact_root=artifact.root,
        expected_dataset_artifact_sha256=artifact.artifact_digest,
        bundle_root=bundle.root,
        expected_bundle_sha256=bundle.bundle_sha256,
        expected_manifest_sha256=bundle.training_manifest_sha256,
        expected_completion_sha256=bundle.training_completion_sha256,
        expected_training_context_sha256=bundle.training_context_sha256,
        expected_checkpoint_sha256=bundle.checkpoint_sha256,
        bound=runtime.bound,
        start_index=5,
        stop_index=10,
        initial_capital=1000.0,
        target_abs_weights=runtime.target_abs_weights,
        execution_cost=cost,
        risk_config=_risk(),
        market_order_profile=None,
        capture_ledger_evidence=True,
        native_fixed_linear_carry=True,
    )
    assert replay.book.portfolio_value == pytest.approx(1000 * (1 + sum(rewards)))
    assert replay.book.exact_quantities == (0, 0)


def test_signed_mode_cannot_use_month_or_unverified_dataset():
    from tests.strategies.rl.test_portfolio_native_calendar_month import (
        _bound as month_bound,
    )
    from tests.strategies.rl.test_portfolio_native_calendar_month import (
        _month_dataset,
    )

    with pytest.raises(ValueError, match="native fixed linear carry"):
        SharedPortfolioGymEnv(
            _month_dataset(),
            start_index=0,
            stop_index=5,
            bound=month_bound(),
            target_abs_weights=(0.25, 0.25),
            execution_cost=_cost(),
            risk_config=_risk(),
            native_fixed_linear_carry=True,
        )
    dataset = replace(_dataset(), identity_payload_json=None)
    with pytest.raises(ValueError, match="verified Dataset"):
        _environment(dataset)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_leverage": 2.0},
        {"order_latency_bars": 1},
        {"partial_fill_carry": True},
    ],
)
def test_signed_reset_rechecks_cost_before_session(changes):
    env = _environment(_dataset())
    env.execution_cost = replace(env.execution_cost, **changes)
    with pytest.raises(ValueError, match="native fixed linear carry"):
        env.reset()
    assert env._session is None


def test_signed_fit_rechecks_late_actor_window_before_normalization():
    env = _environment(_dataset())
    env.stop_index = 6
    with pytest.raises(ValueError, match="financial clock"):
        fit_portfolio_observation_transform(env)


def test_signed_mode_rejects_fixed_scenario_before_context_decode():
    from trade_rl.strategies.rl.portfolio_scenario_runtime import (
        create_portfolio_scenario_runtime_binding_v2,
    )

    env = _environment(_dataset())
    transform = fit_portfolio_observation_transform(env)
    runtime = _environment(env.dataset, start=5, stop=10)
    with pytest.raises(ValueError, match="linear carry scenario"):
        create_portfolio_scenario_runtime_binding_v2(
            transform,
            env.bound,
            runtime,
            training_context={},
            training_context_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            scenario=object(),
            expected_scenario_sha256="c" * 64,
            scenario_inference_source_sha256="d" * 64,
            require_unreset=True,
        )
