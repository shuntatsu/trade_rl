"""Signed UTC-month learning and inference retain one exact carry identity."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.rl.test_portfolio_inference_bundle import _load, _publish
from tests.strategies.rl.test_portfolio_native_calendar_month import (
    _bound,
    _month_dataset,
)
from tests.strategies.rl.test_portfolio_native_calendar_month import (
    _environment as _spot_environment,
)
from tests.strategies.rl.test_portfolio_native_fixed import _config, _risk
from tests.strategies.rl.test_portfolio_native_fixed_carry import _cost, _dataset
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_execution_clock import calendar_month_execution_frequency
from trade_rl.evaluation.portfolio_inference import (
    run_published_portfolio_inference_replay,
)
from trade_rl.simulation import runtime_execution_policy_digest
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


def _signed_dataset():
    base = _dataset()
    closes = np.arange("2024-01", "2025-01", dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    return replace(
        base,
        timestamps=closes,
        available_at=np.broadcast_to(closes[:, None], (12, 2)).copy(),
        native_clock=clock,
        nominal_bar_hours=None,
        periods_per_year=calendar_month_execution_frequency(clock, closes),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "generated-month-signed-ppo"})


def _env(dataset=None, start=0, stop=5, **changes):
    return SharedPortfolioGymEnv(
        _signed_dataset() if dataset is None else dataset,
        **(
            {
                "start_index": start,
                "stop_index": stop,
                "bound": _bound(start, stop),
                "target_abs_weights": (0.25, 0.25),
                "execution_cost": _cost(),
                "risk_config": _risk(),
                "native_calendar_month_linear_carry": True,
            }
            | changes
        ),
    )


def test_signed_month_gym_uses_literal_elapsed_borrow_and_terminal_rewards():
    env = _env()
    transform = fit_portfolio_observation_transform(env)
    assert transform.native_calendar_month_linear_carry
    assert not transform.native_fixed_linear_carry
    assert transform.native_interval_seconds is None
    env.reset(seed=42)
    _, reward, done, truncated, _ = env.step(np.array([ACTION_LONG, ACTION_SHORT]))
    assert not done and not truncated
    assert env._session is not None
    assert env._session.book.exact_quantities == (Fraction(5, 4), Fraction(-5, 8))
    # 250 long + 250 short: entry fee 0.5, funding -2.75+5, borrow 31/4.
    assert reward == pytest.approx(float((Fraction(7, 4) - Fraction(31, 4)) / 1000))
    rewards = [reward]
    while not done:
        _, reward, done, truncated, info = env.step(
            np.array([ACTION_HOLD, ACTION_HOLD])
        )
        rewards.append(reward)
    assert not truncated and info["settlement_complete"]
    # Only Jan, Feb, Mar and Apr have held short inventory. May is settled.
    expected = Fraction(1000) - 1 + 4 * Fraction(9, 4) - Fraction(31 + 29 + 31 + 30, 4)
    assert env._session.book.exact_quantities == (0, 0)
    assert env._session.book.cash == pytest.approx(float(expected))
    assert sum(rewards) == pytest.approx(float(expected / 1000 - 1))
    assert env._session.execution_policy_digest == runtime_execution_policy_digest(
        env.execution_cost,
        funding_price_rate_present=True,
        native_calendar_month_linear_carry=True,
    )


def test_signed_month_transform_schema_and_compatibility():
    env = _env()
    transform = fit_portfolio_observation_transform(env)
    payload = transform.to_payload()
    assert (
        payload["schema"]
        == "shared_portfolio_calendar_month_linear_carry_observation_transform_v1"
    )
    assert (
        PortfolioObservationTransform.from_payload(
            payload, expected_digest=transform.digest
        )
        == transform
    )
    spot_payload = payload | {
        "schema": "shared_portfolio_calendar_month_observation_transform_v1"
    }
    with pytest.raises(ValueError):
        PortfolioObservationTransform.from_payload(
            spot_payload, expected_digest=transform.digest
        )
    wrong = PortfolioObservationTransform.from_payload(
        spot_payload, expected_digest=content_digest(spot_payload)
    )
    assert not wrong.native_calendar_month_linear_carry
    with pytest.raises(ValueError, match="mode"):
        wrong.validate_environment(env)
    runtime = _env(env.dataset, 5, 10)
    with pytest.raises(ValueError, match="mode"):
        PortfolioObservationRuntimeBinding.create(wrong, env.bound, runtime)
    binding = PortfolioObservationRuntimeBinding.create(transform, env.bound, runtime)
    assert binding.transform_digest == transform.digest
    assert (
        PortfolioObservationRuntimeBinding.from_payload(
            binding.to_payload(), expected_digest=binding.digest
        )
        == binding
    )


@pytest.mark.parametrize(
    "mutation", ["multiple", "borrow", "late", "unverified", "spot"]
)
def test_month_signed_gym_refuses_invalid_window_before_session(mutation):
    dataset = _signed_dataset()
    changes = {"identity_payload_json": None}
    if mutation == "multiple":
        counts = dataset.funding_event_count.copy()
        counts[5, 0] = 2
        changes["funding_event_count"] = counts
    elif mutation == "borrow":
        available = dataset.borrow_available.copy()
        available[5, 0] = False
        changes["borrow_available"] = available
    elif mutation == "late":
        available = dataset.available_at.copy()
        available[5, 0] += np.timedelta64(1, "ns")
        info = dataset.information_available.copy()
        info[5, 0] = False
        changes.update(available_at=available, information_available=info)
    elif mutation == "spot":
        dataset = _month_dataset()
    if mutation != "spot":
        dataset = replace(dataset, **changes)
        if mutation != "unverified":
            dataset = dataset.with_content_identity()
    with pytest.raises(ValueError):
        _env(dataset)


@pytest.mark.parametrize(
    "changes",
    [
        {"native_calendar_month_linear_carry": 1},
        {"native_fixed_linear_carry": True},
        {"native_calendar_month_linear_carry": False},
        {"execution_cost": replace(_cost(), order_latency_bars=1)},
    ],
)
def test_month_signed_gym_rejects_incompatible_modes(changes):
    with pytest.raises(ValueError):
        _env(**changes)


def test_actual_signed_month_ppo_bundle_and_paired_replay(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _signed_dataset()
    env = _env(dataset)
    transform = fit_portfolio_observation_transform(env)
    trained = fit_shared_portfolio_ppo(
        env, transform=transform, output_root=tmp_path / "attempt", config=_config()
    )
    assert trained.realized_timesteps == 8
    context = json.loads((trained.output_root / "training_context.json").read_bytes())
    assert (
        context["schema"]
        == "shared_portfolio_calendar_month_linear_carry_training_context_v1"
    )
    manifest = json.loads((trained.output_root / "manifest.json").read_bytes())
    wrong = context | {"schema": "shared_portfolio_training_context_v1"}
    with pytest.raises(ValueError, match="mode|transform"):
        portfolio_inference_bundle._validated_training_context(wrong, manifest)
    publication = _publish(trained, tmp_path / "bundle")

    # A valid but incompatible Spot runtime must fail before PPO loading.
    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint deserializer must not run on a mismatched runtime")

    with monkeypatch.context() as m:
        m.setattr(
            portfolio_inference_bundle, "_load_policy_from_verified_path", forbidden
        )
        with pytest.raises(ValueError):
            _load(publication, _spot_environment(_month_dataset(), 5, 9))
    runtime = _env(dataset, 5, 10)
    policy = _load(publication, runtime)
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
        bundle_root=publication.root,
        expected_bundle_sha256=publication.bundle_sha256,
        expected_manifest_sha256=publication.training_manifest_sha256,
        expected_completion_sha256=publication.training_completion_sha256,
        expected_training_context_sha256=publication.training_context_sha256,
        expected_checkpoint_sha256=publication.checkpoint_sha256,
        bound=runtime.bound,
        start_index=5,
        stop_index=10,
        initial_capital=1000.0,
        target_abs_weights=runtime.target_abs_weights,
        execution_cost=_cost(),
        risk_config=_risk(),
        market_order_profile=None,
        capture_ledger_evidence=True,
        native_calendar_month_linear_carry=True,
    )
    assert replay.book.exact_quantities == (0, 0)
    assert replay.book.portfolio_value == pytest.approx(1000 * (1 + sum(rewards)))


@pytest.mark.parametrize(
    "changes",
    [
        {"native_interval_seconds": 30 * 86400},
        {"agent_stop_index": 3},
        {"market_order_profile_digest": "a" * 64},
    ],
)
def test_signed_month_rehashed_transform_refuses_incompatible_metadata(changes):
    transform = fit_portfolio_observation_transform(_env())
    payload = transform.to_payload() | changes
    with pytest.raises(ValueError):
        PortfolioObservationTransform.from_payload(
            payload, expected_digest=content_digest(payload)
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"order_latency_bars": 1},
        {"partial_fill_carry": True},
        {"max_leverage": 2.0},
    ],
)
def test_signed_month_mutated_cost_refuses_reset_and_fit(changes):
    env = _env()
    env.execution_cost = replace(env.execution_cost, **changes)
    with pytest.raises(ValueError, match="native calendar month linear carry"):
        env.reset()
    with pytest.raises(ValueError, match="native calendar month linear carry"):
        fit_portfolio_observation_transform(env)
    assert env._session is None


def test_month_signed_still_refuses_fixed_scenario():
    from trade_rl.strategies.rl.portfolio_scenario_runtime import (
        create_portfolio_scenario_runtime_binding_v2,
    )

    env = _env()
    transform = fit_portfolio_observation_transform(env)
    with pytest.raises(ValueError, match="calendar month scenario"):
        create_portfolio_scenario_runtime_binding_v2(
            transform,
            env.bound,
            _env(env.dataset, 5, 10),
            training_context={},
            training_context_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            scenario=object(),
            expected_scenario_sha256="c" * 64,
            scenario_inference_source_sha256="d" * 64,
            require_unreset=True,
        )
