"""Signed declared sessions bind their carry mode from Gym to inference."""

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.strategies.rl.test_portfolio_inference_bundle import _load, _publish
from tests.strategies.rl.test_portfolio_native_fixed import _config
from tests.strategies.rl.test_portfolio_session_elapsed import (
    _bound,
    _cost,
    _risk,
)
from tests.strategies.rl.test_portfolio_session_elapsed import (
    _dataset as _spot_dataset,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.data.contracts import VolumeUnit
from trade_rl.evaluation.portfolio_inference import (
    run_published_portfolio_inference_replay,
)
from trade_rl.evaluation.series import SessionElapsedReturnSeries
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


def _dataset():
    base = _spot_dataset()
    rates = np.full((12, 2), 0.01)
    return replace(
        base,
        volume_units=(VolumeUnit.CONTRACTS,) * 2,
        contract_multipliers=np.array([2.0, 4.0]),
        borrow_rate=np.full((12, 2), 0.365),
        funding_rate=rates,
        funding_price_rate=rates * np.array([110.0, 200.0]),
        funding_event_count=np.ones((12, 2), dtype=np.int32),
        funding_due=np.ones((12, 2), dtype=bool),
        cash_rate=np.zeros(12),
        identity_payload_json=None,
    ).with_content_identity({"fixture": "signed-session-ppo"})


def _env(dataset=None, start=0, stop=5, **changes):
    dataset = _dataset() if dataset is None else dataset
    return SharedPortfolioGymEnv(
        dataset,
        **(
            {
                "start_index": start,
                "stop_index": stop,
                "bound": _bound(dataset, start, stop),
                "target_abs_weights": (0.25, 0.25),
                "execution_cost": _cost(allow_short=True, borrow_rate_multiplier=1.0),
                "risk_config": _risk(),
                "native_session_linear_carry": True,
            }
            | changes
        ),
    )


def test_signed_session_gym_reward_and_transform_oracle():
    env = _env()
    transform = fit_portfolio_observation_transform(env)
    assert transform.native_session_linear_carry
    assert transform.to_payload()["schema"] == (
        "shared_portfolio_session_linear_carry_observation_transform_v1"
    )
    assert (
        PortfolioObservationTransform.from_payload(
            transform.to_payload(), expected_digest=transform.digest
        )
        == transform
    )
    env.reset(seed=42)
    _, reward, done, truncated, _ = env.step(np.array([ACTION_LONG, ACTION_SHORT]))
    assert not done and not truncated
    assert env._session.book.exact_quantities == (Fraction(5, 4), Fraction(-5, 8))
    # No inventory during the 64h weekend. Entry fee 0.5, funding +2.25,
    # borrow for just the one processing hour at 250 * .365 / 8760.
    expected = (Fraction(7, 4) - Fraction(250, 24000)) / 1000
    assert reward == pytest.approx(float(expected))
    rewards = [reward]
    while not done:
        _, reward, done, truncated, info = env.step(
            np.array([ACTION_HOLD, ACTION_HOLD])
        )
        rewards.append(reward)
    assert not truncated and info["objective_endpoint_valid"]
    assert env._session.book.exact_quantities == (0, 0)
    # Held short from first processing open through terminal processing open:
    # 1 + 1 + 1 + 22 hours; terminal interval is 1h with no gap.
    expected_cash = Fraction(1000) - 1 + 4 * Fraction(9, 4) - Fraction(250 * 25, 24000)
    assert env._session.book.cash == pytest.approx(float(expected_cash))
    assert sum(rewards) == pytest.approx(float(expected_cash / 1000 - 1))


def test_signed_session_mode_is_bound_to_transform_and_forward_clock():
    env = _env()
    transform = fit_portfolio_observation_transform(env)
    payload = transform.to_payload()
    spot_payload = payload | {
        "schema": "shared_portfolio_session_observation_transform_v1"
    }
    wrong = PortfolioObservationTransform.from_payload(
        spot_payload, expected_digest=content_digest(spot_payload)
    )
    assert not wrong.native_session_linear_carry
    with pytest.raises(ValueError, match="mode"):
        wrong.validate_environment(env)
    runtime = _env(env.dataset, 5, 10)
    with pytest.raises(ValueError, match="mode"):
        PortfolioObservationRuntimeBinding.create(wrong, env.bound, runtime)
    binding = PortfolioObservationRuntimeBinding.create(transform, env.bound, runtime)
    assert binding.transform_digest == transform.digest
    assert binding.session_boundaries_ns == runtime.bound.clock.boundaries_ns
    for changes in (
        {"session_boundaries_ns": []},
        {"agent_stop_index": 3},
        {"market_order_profile_digest": "a" * 64},
    ):
        changed = payload | changes
        with pytest.raises(ValueError):
            PortfolioObservationTransform.from_payload(
                changed, expected_digest=content_digest(changed)
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"native_session_linear_carry": 1},
        {"native_session_linear_carry": False},
        {"native_fixed_linear_carry": True},
        {"native_calendar_month_linear_carry": True},
        {"execution_cost": _cost(allow_short=True, order_latency_bars=1)},
    ],
)
def test_signed_session_refuses_wrong_mode_or_cost(changes):
    with pytest.raises(ValueError):
        _env(**changes)


def test_signed_session_reset_and_fit_refuse_mutated_cost():
    env = _env()
    env.execution_cost = _cost(allow_short=True, partial_fill_carry=True)
    with pytest.raises(ValueError, match="native session linear carry"):
        env.reset()
    with pytest.raises(ValueError, match="native session linear carry"):
        fit_portfolio_observation_transform(env)
    assert env._session is None


def test_actual_signed_session_ppo_bundle_and_paired_replay(tmp_path, monkeypatch):
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
    assert (
        context["schema"] == "shared_portfolio_session_linear_carry_training_context_v1"
    )
    manifest = json.loads((trained.output_root / "manifest.json").read_bytes())
    with pytest.raises(ValueError, match="mode|transform"):
        portfolio_inference_bundle._validated_training_context(
            context | {"schema": "shared_portfolio_session_training_context_v1"},
            manifest,
        )
    bundle = _publish(trained, tmp_path / "bundle")

    def forbidden(*args, **kwargs):
        pytest.fail("mismatched mode reached checkpoint loading")

    with monkeypatch.context() as m:
        m.setattr(
            portfolio_inference_bundle, "_load_policy_from_verified_path", forbidden
        )
        with pytest.raises(ValueError):
            _load(
                bundle,
                _env(
                    _spot_dataset(),
                    5,
                    10,
                    native_session_linear_carry=False,
                    execution_cost=_cost(),
                ),
            )
    runtime = _env(dataset, 5, 10)
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
        native_session_linear_carry=True,
    )
    assert isinstance(replay.returns, SessionElapsedReturnSeries)
    assert replay.returns.boundaries_ns == runtime.bound.clock.boundaries_ns
    assert replay.book.exact_quantities == (0, 0)
    assert replay.book.portfolio_value == pytest.approx(1000 * (1 + sum(rewards)))


def test_session_signed_refuses_interior_clock_change_before_fit():
    env = _env()
    boundaries = env.bound.clock.boundaries_ns
    env.bound = _bound(
        env.dataset,
        boundaries_ns=(boundaries[0], boundaries[1] - 10**9, *boundaries[2:]),
    )
    with pytest.raises(ValueError, match="boundary"):
        fit_portfolio_observation_transform(env)
    assert env._session is None
