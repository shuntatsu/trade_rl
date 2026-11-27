"""Synthetic caller-declared session clock, native Spot rewards and frozen PPO."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.rl.test_portfolio_env import _bound_clock, _risk
from tests.strategies.rl.test_portfolio_inference_bundle import _dataset as base_dataset
from tests.strategies.rl.test_portfolio_inference_bundle import _load, _publish
from tests.strategies.rl.test_portfolio_native_fixed import _config
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.evaluation.portfolio_inference import (
    run_published_portfolio_inference_replay,
)
from trade_rl.evaluation.series import SessionElapsedReturnSeries
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.simulation.objectives import (
    BoundObjectiveClock,
    SessionFinancialClockContract,
)
from trade_rl.strategies.rl import portfolio_inference_bundle
from trade_rl.strategies.rl.portfolio_env import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    SharedPortfolioGymEnv,
)
from trade_rl.strategies.rl.portfolio_training import fit_shared_portfolio_ppo
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioObservationRuntimeBinding,
    PortfolioObservationTransform,
    fit_portfolio_observation_transform,
)

_START = datetime(2026, 1, 2, 16, tzinfo=UTC)
_HOURS = (0, 65, 66, 67, 89, 90, 91, 92, 114, 115, 116, 181)


def _dataset(*, regular=False):
    base = base_dataset(12)
    hours = tuple(range(12)) if regular else _HOURS
    timestamps = np.datetime64("2026-01-02T16:00", "ns") + np.asarray(
        hours
    ) * np.timedelta64(1, "h")
    return replace(
        base,
        timestamps=timestamps,
        available_at=np.broadcast_to(timestamps[:, None], (12, 2)).copy(),
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        periods_per_year=1638,
        cash_rate=np.full(12, 0.365),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "session-ppo"})


def _bound(dataset, start=0, stop=5, **changes):
    boundaries = tuple(
        int(x) for x in dataset.timestamps[start : stop + 1].view(np.int64)
    )
    base = _bound_clock(start_index=start, stop_index=stop)
    start_dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
        seconds=boundaries[0] // 10**9
    )
    stop_dt = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
        seconds=boundaries[-1] // 10**9
    )
    return BoundObjectiveClock(
        replace(
            base.objective, evaluation_start=start_dt, evaluation_stop_exclusive=stop_dt
        ),
        SessionFinancialClockContract(
            **(
                {
                    "boundaries_ns": boundaries,
                    "nominal_bar_seconds": 3600,
                    "economic_horizon_seconds": (boundaries[-1] - boundaries[0])
                    // 10**9,
                    "rollout_steps": 4,
                    "gamma": 1.0,
                    "gae_lambda": 0.95,
                    "reward_schema": "equity_delta_v1",
                }
                | changes
            )
        ),
    )


def _cost(**changes):
    return replace(
        ExecutionCostConfig.zero(),
        **(
            {
                "allow_short": False,
                "processing_bar_volume_capacity": False,
                "fee_rate": 0.001,
                "partial_fill_carry": False,
            }
            | changes
        ),
    )


def _env(dataset, *, start=0, stop=5, bound=None, cost=None):
    return SharedPortfolioGymEnv(
        dataset,
        start_index=start,
        stop_index=stop,
        bound=_bound(dataset, start, stop) if bound is None else bound,
        target_abs_weights=(0.25, 0.25),
        execution_cost=_cost() if cost is None else cost,
        risk_config=_risk(),
    )


def test_session_gym_reward_and_clock_literal_oracle():
    env = _env(_dataset())
    transform = fit_portfolio_observation_transform(env)
    assert transform.session_boundaries_ns == env.bound.clock.boundaries_ns
    assert transform.native_interval_seconds is None
    assert (
        transform.to_payload()["schema"]
        == "shared_portfolio_session_observation_transform_v1"
    )
    raw, _ = env.reset(seed=42)
    assert transform.transform_observation(raw)["clock"] == pytest.approx(
        [0.0, 1.0, 0.0]
    )
    raw, reward, done, truncated, _ = env.step(np.array([ACTION_LONG, ACTION_FLAT]))
    assert not done and not truncated
    assert env._session.book.exact_quantities == (Fraction(5, 2), Fraction(0))
    gap_interest = Fraction(1000) * Fraction(64, 24000)
    processing_basis = Fraction(1000) - 250 - Fraction(1, 4)
    expected = gap_interest + processing_basis / 24000 - Fraction(1, 4)
    assert reward == pytest.approx(float(expected / 1000))
    assert transform.transform_observation(raw)["clock"] == pytest.approx(
        [0.25, 0.75, 65 / 90]
    )
    forged = {**raw, "clock": raw["clock"].copy()}
    forged["clock"][1] += 1
    with pytest.raises(ValueError, match="timestamp"):
        transform.transform_observation(forged)
    rewards = [reward]
    while not done:
        raw, reward, done, truncated, info = env.step(
            np.array([ACTION_HOLD, ACTION_HOLD])
        )
        rewards.append(reward)
    assert not truncated and info["objective_endpoint_valid"]
    assert env._session.book.exact_quantities == (0, 0)
    assert sum(rewards) == pytest.approx(env._session.book.portfolio_value / 1000 - 1)


def test_session_clock_and_transform_payloads_pin_interior_grid():
    dataset = _dataset()
    env = _env(dataset)
    clock = env.bound.clock
    assert portfolio_inference_bundle._clock_from_payload(clock.payload()) == clock
    transform = fit_portfolio_observation_transform(env)
    assert (
        PortfolioObservationTransform.from_payload(
            transform.to_payload(), expected_digest=transform.digest
        )
        == transform
    )
    boundaries = list(clock.boundaries_ns)
    boundaries[1] -= 3600 * 10**9
    other = _bound(dataset, boundaries_ns=tuple(boundaries))
    assert other.digest != env.bound.digest
    with pytest.raises(ValueError, match="boundary|boundaries"):
        _env(dataset, bound=other)
    payload = transform.to_payload()
    wrong = payload | {"session_boundaries_ns": boundaries}
    altered = PortfolioObservationTransform.from_payload(
        wrong, expected_digest=content_digest(wrong)
    )
    with pytest.raises(ValueError, match="boundary|boundaries|context"):
        altered.validate_environment(env)


@pytest.mark.parametrize(
    "changes",
    [
        {"nominal_bar_seconds": True},
        {"nominal_bar_seconds": 0},
        {"nominal_bar_seconds": 7200},
        {"economic_horizon_seconds": 1},
        {"gamma": 0.99},
        {"gae_lambda": 1.1},
        {"reward_schema": "net_log_return_v1"},
        {"boundaries_ns": (0, 0, 1)},
        {"boundaries_ns": (0, 10**9, 2 * 10**9 + 1)},
    ],
)
def test_session_clock_rejects_invalid_declaration(changes):
    with pytest.raises(ValueError):
        _bound(_dataset(), **changes)


def test_session_reset_rechecks_binding_and_economics():
    dataset = _dataset()
    env = _env(dataset)
    env.execution_cost = _cost(allow_short=True)
    with pytest.raises(ValueError, match="session"):
        env.reset()
    env.execution_cost = _cost()
    env.dataset = replace(dataset, identity_payload_json=None).with_content_identity()
    with pytest.raises(ValueError, match="original verified Dataset"):
        env.reset()


def test_session_forward_binding_and_cross_clock_refusal():
    dataset = _dataset()
    env = _env(dataset)
    transform = fit_portfolio_observation_transform(env)
    runtime = _env(dataset, start=5, stop=10)
    binding = PortfolioObservationRuntimeBinding.create(transform, env.bound, runtime)
    assert binding.session_boundaries_ns == runtime.bound.clock.boundaries_ns
    assert (
        PortfolioObservationRuntimeBinding.from_payload(
            binding.to_payload(), expected_digest=binding.digest
        )
        == binding
    )
    raw, _ = runtime.reset()
    assert transform.transform_observation(raw, runtime_binding=binding)[
        "clock"
    ] == pytest.approx([0.0, 1.0, 0.0])
    bad = raw.copy()
    bad["clock"] = raw["clock"].copy()
    bad["clock"][1] += 1
    with pytest.raises(ValueError, match="timestamp"):
        transform.transform_observation(bad, runtime_binding=binding)


def test_actual_session_ppo_bundle_and_paired_replay(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _dataset()
    env = _env(dataset)
    transform = fit_portfolio_observation_transform(env)
    trained = fit_shared_portfolio_ppo(
        env, transform=transform, output_root=tmp_path / "attempt", config=_config()
    )
    assert trained.realized_timesteps == 8
    context = json.loads((trained.output_root / "training_context.json").read_bytes())
    assert context["schema"] == "shared_portfolio_session_training_context_v1"
    manifest = json.loads((trained.output_root / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="session|transform"):
        portfolio_inference_bundle._validated_training_context(
            context | {"schema": "shared_portfolio_training_context_v1"}, manifest
        )
    bundle = _publish(trained, tmp_path / "bundle")
    runtime = _env(dataset, start=5, stop=10)
    policy = _load(bundle, runtime)
    raw, _ = runtime.reset(seed=42)
    rewards = []
    done = False
    while not done:
        raw, reward, done, truncated, info = runtime.step(policy.predict(raw))
        rewards.append(reward)
    assert not truncated and info["objective_endpoint_valid"]
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
        execution_cost=runtime.execution_cost,
        risk_config=_risk(),
        market_order_profile=None,
        capture_ledger_evidence=True,
    )
    assert isinstance(replay.returns, SessionElapsedReturnSeries)
    assert replay.returns.boundaries_ns == runtime.bound.clock.boundaries_ns
    assert replay.book.portfolio_value == pytest.approx(1000 * (1 + sum(rewards)))
    assert replay.book.exact_quantities == (0, 0)


def test_same_regular_session_dataset_does_not_authorize_cross_mode():
    dataset = _dataset(regular=True)
    env = _env(dataset)
    base = _bound_clock(start_index=0, stop_index=5)
    fixed_bound = BoundObjectiveClock(
        replace(
            base.objective,
            evaluation_start=env.bound.objective.evaluation_start,
            evaluation_stop_exclusive=env.bound.objective.evaluation_stop_exclusive,
        ),
        replace(base.clock, rollout_steps=4),
    )
    fixed = _env(dataset, bound=fixed_bound)
    session_transform = fit_portfolio_observation_transform(env)
    fixed_transform = fit_portfolio_observation_transform(fixed)
    assert session_transform.digest != fixed_transform.digest
    with pytest.raises(ValueError, match="session"):
        session_transform.validate_environment(fixed)
    with pytest.raises(ValueError, match="session"):
        fixed_transform.validate_environment(env)
    runtime = _env(dataset, start=5, stop=10)
    with pytest.raises(ValueError, match="session|training bound"):
        PortfolioObservationRuntimeBinding.create(fixed_transform, fixed.bound, runtime)


def test_forward_binding_rehashed_interior_boundary_is_not_environment_proof():
    dataset = _dataset()
    env = _env(dataset)
    transform = fit_portfolio_observation_transform(env)
    runtime = _env(dataset, start=5, stop=10)
    binding = PortfolioObservationRuntimeBinding.create(transform, env.bound, runtime)
    boundaries = list(binding.session_boundaries_ns)
    boundaries[2] += 3600 * 10**9
    payload = binding.to_payload() | {"session_boundaries_ns": boundaries}
    # The serialized object alone is a declaration; its environment join is
    # required independently even when a caller supplies a matching new digest.
    forged = PortfolioObservationRuntimeBinding.from_payload(
        payload, expected_digest=content_digest(payload)
    )
    with pytest.raises(ValueError, match="binding"):
        forged.validate_environment(transform, env.bound, runtime)
    for changes in (
        {"session_boundaries_ns": []},
        {"session_boundaries_ns": [True] * 6},
        {"native_interval_seconds": 3600},
        {"schema": "shared_portfolio_calendar_month_observation_runtime_binding_v1"},
    ):
        wrong = binding.to_payload() | changes
        with pytest.raises(ValueError):
            PortfolioObservationRuntimeBinding.from_payload(
                wrong, expected_digest=content_digest(wrong)
            )


def test_session_grid_digest_is_not_recomputed_per_prediction(monkeypatch):
    from trade_rl.simulation.objectives import clock as objective_clock
    from trade_rl.strategies.rl import portfolio_transform

    dataset = _dataset()
    env = _env(dataset)
    transform = fit_portfolio_observation_transform(env)
    runtime = _env(dataset, start=5, stop=10)
    binding = PortfolioObservationRuntimeBinding.create(transform, env.bound, runtime)
    raw, _ = runtime.reset()
    assert transform.digest == content_digest(transform.to_payload())
    assert env.bound.clock.digest == content_digest(env.bound.clock.payload())

    def no_rehash(_):
        raise AssertionError("session grid hashed during inference")

    monkeypatch.setattr(portfolio_transform, "content_digest", no_rehash)
    monkeypatch.setattr(objective_clock, "content_digest", no_rehash)
    for _ in range(16):
        assert (
            transform.transform_observation(raw, runtime_binding=binding)["clock"][0]
            == 0
        )
        assert env.bound.clock.digest


def test_session_clock_requires_same_objective_endpoints():
    env = _env(_dataset())
    with pytest.raises(ValueError, match="session clock boundary"):
        BoundObjectiveClock(
            replace(
                env.bound.objective,
                evaluation_start=env.bound.objective.evaluation_start
                + timedelta(seconds=1),
                evaluation_stop_exclusive=env.bound.objective.evaluation_stop_exclusive
                + timedelta(seconds=1),
            ),
            env.bound.clock,
        )


def test_session_learning_refuses_late_terminal_before_fit_or_actor():
    dataset = _dataset()
    times = dataset.available_at.copy()
    times[5, 0] += np.timedelta64(1, "ns")
    information = dataset.information_available.copy()
    information[5, 0] = False
    altered = replace(
        dataset,
        available_at=times,
        information_available=information,
        identity_payload_json=None,
    ).with_content_identity()
    with pytest.raises(ValueError, match="available"):
        _env(altered)


def test_session_fit_refuses_unpublishable_context_before_attempt(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from trade_rl.strategies.rl import portfolio_training

    env = _env(_dataset())
    transform = fit_portfolio_observation_transform(env)
    monkeypatch.setattr(portfolio_training, "_PORTFOLIO_METADATA_MAX_BYTES", 1024)
    root = tmp_path / "attempt"
    with pytest.raises(ValueError, match="metadata size"):
        fit_shared_portfolio_ppo(
            env, transform=transform, output_root=root, config=_config()
        )
    assert not root.exists()
    assert env._session is None
