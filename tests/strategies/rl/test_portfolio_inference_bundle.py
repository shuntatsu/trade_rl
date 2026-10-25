from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_training import (
    SharedPortfolioPPOConfig,
    fit_shared_portfolio_ppo,
)
from trade_rl.strategies.rl.portfolio_transform import (
    fit_portfolio_observation_transform,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _dataset() -> MarketDataset:
    close = np.full((8, 2), 100.0, dtype=np.float64)
    zeros = np.zeros_like(close)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        len(close)
    ) * np.timedelta64(1, "h")
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("AAA", "BBB"),
        timestamps=timestamps,
        features=np.arange(16, dtype=np.float32).reshape(8, 2, 1),
        global_features=np.arange(8, dtype=np.float32).reshape(8, 1),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=zeros.copy(),
        funding_due=None,
        borrow_rate=zeros.copy(),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((8, 2, 1), dtype=np.bool_),
        feature_names=("literal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        fee_rate=zeros.copy(),
        maker_fee_rate=zeros.copy(),
        taker_fee_rate=zeros.copy(),
        spread_rate=zeros.copy(),
        contract_multipliers=np.ones(2, dtype=np.float64),
    )


def _environment(
    dataset: MarketDataset,
    start_index: int,
    stop_index: int,
    *,
    target_abs_weights: tuple[float, ...] = (0.5, 0.5),
) -> SharedPortfolioGymEnv:
    stop = _START + timedelta(hours=stop_index)
    bound = BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=_START + timedelta(hours=start_index),
            evaluation_stop_exclusive=stop,
            terminal_valuation="settled",
            economics_digest="b" * 64,
            risk_digest="c" * 64,
            deployment_recipe_digest="d" * 64,
            maximum_drawdown=0.20,
        ),
        FinancialClockContract(
            decision_interval_seconds=3_600,
            execution_interval_seconds=3_600,
            reward_interval_seconds=3_600,
            economic_horizon_seconds=(stop_index - start_index) * 3_600,
            rollout_steps=4,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )
    return SharedPortfolioGymEnv(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        bound=bound,
        target_abs_weights=target_abs_weights,
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.20,
            drawdown_stop=0.20,
        ),
    )


def _train_one_rollout(root: Path) -> tuple[Any, MarketDataset, SharedPortfolioGymEnv]:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _dataset()
    training_env = _environment(dataset, 0, 5)
    transform = fit_portfolio_observation_transform(training_env)
    result = fit_shared_portfolio_ppo(
        training_env,
        transform=transform,
        output_root=root / "attempt",
        config=SharedPortfolioPPOConfig(
            total_timesteps=4,
            n_steps=4,
            batch_size=2,
            n_epochs=1,
            reset_seed=29,
            ppo_seed=31,
        ),
    )
    return result, dataset, training_env


def _publish(result: Any, destination: Path) -> Any:
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )

    return publish_portfolio_inference_bundle(
        result.output_root,
        destination,
        expected_manifest_sha256=result.manifest_sha256,
        expected_completion_sha256=result.completion_sha256,
        expected_training_context_sha256=result.training_context_sha256,
        expected_checkpoint_sha256=result.final_checkpoint_sha256,
    )


def _load(publication: Any, runtime_env: SharedPortfolioGymEnv) -> Any:
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        load_portfolio_inference_bundle,
    )

    return load_portfolio_inference_bundle(
        publication.root,
        expected_bundle_sha256=publication.bundle_sha256,
        expected_manifest_sha256=publication.training_manifest_sha256,
        expected_completion_sha256=publication.training_completion_sha256,
        expected_training_context_sha256=publication.training_context_sha256,
        expected_checkpoint_sha256=publication.checkpoint_sha256,
        runtime_env=runtime_env,
    )


def test_bundle_round_trip_is_strict_and_deterministic(tmp_path: Path) -> None:
    result, dataset, training_env = _train_one_rollout(tmp_path)
    assert result.output_root == tmp_path / "attempt"
    assert result.final_checkpoint_sha256
    assert result.manifest_sha256
    assert result.training_context_sha256
    assert result.completion_sha256

    altered_attempt = tmp_path / "altered-attempt"
    shutil.copytree(result.output_root, altered_attempt)
    completion = json.loads(
        (altered_attempt / "completion.json").read_text(encoding="utf-8")
    )
    checkpoint_path = altered_attempt / completion["final_checkpoint_path"]
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"tamper")
    altered_output = tmp_path / "tampered-checkpoint-bundle"
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )

    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        publish_portfolio_inference_bundle(
            altered_attempt,
            altered_output,
            expected_manifest_sha256=result.manifest_sha256,
            expected_completion_sha256=result.completion_sha256,
            expected_training_context_sha256=result.training_context_sha256,
            expected_checkpoint_sha256=result.final_checkpoint_sha256,
        )
    assert not altered_output.exists()

    publication = _publish(result, tmp_path / "bundle")
    runtime_env = _environment(dataset, 5, 7)
    policy = _load(publication, runtime_env)
    raw_observation, _ = runtime_env.reset(seed=41)
    original_features = raw_observation["features"].copy()
    action_one = policy.predict(raw_observation)
    action_two = policy.predict(raw_observation)

    assert publication.training_context_sha256 == result.training_context_sha256
    assert policy.runtime_binding.runtime_start_index == 5
    assert np.array_equal(action_one, action_two)
    assert action_one.shape == (2,)
    assert np.all((0 <= action_one) & (action_one < 4))
    assert np.array_equal(raw_observation["features"], original_features)
    assert not hasattr(policy, "learn")
    assert not hasattr(policy, "model")
    assert runtime_env._session is not None
    assert runtime_env._session.current_index == 5
    assert training_env._session is not None

    mismatched_env = _environment(dataset, 5, 7, target_abs_weights=(0.4, 0.6))
    with pytest.MonkeyPatch.context() as monkeypatch:
        from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

        def must_not_deserialize(*args: object, **kwargs: object) -> object:
            raise AssertionError("runtime binding rejection must precede PPO load")

        monkeypatch.setattr(
            bundle_module, "_load_policy_from_verified_path", must_not_deserialize
        )
        with pytest.raises(ValueError, match="financial or target-weight context"):
            _load(publication, mismatched_env)

    tampered_root = tmp_path / "tampered"
    shutil.copytree(publication.root, tampered_root)
    context_path = tampered_root / "training_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["transform_digest"] = "0" * 64
    context_path.write_text(json.dumps(context), encoding="utf-8")

    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    def must_not_deserialize(*args: object, **kwargs: object) -> object:
        raise AssertionError("metadata rejection must precede PPO deserialization")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            bundle_module,
            "_load_policy_from_verified_path",
            must_not_deserialize,
        )
        with pytest.raises(
            ValueError, match="training context.*digest|digest.*context"
        ):
            _load(
                publication.__class__(
                    root=tampered_root,
                    bundle_sha256=publication.bundle_sha256,
                    training_manifest_sha256=publication.training_manifest_sha256,
                    training_completion_sha256=publication.training_completion_sha256,
                    training_context_sha256=publication.training_context_sha256,
                    checkpoint_sha256=publication.checkpoint_sha256,
                ),
                runtime_env,
            )


def test_failed_training_attempt_cannot_be_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    dataset = _dataset()
    env = _environment(dataset, 0, 5)
    transform = fit_portfolio_observation_transform(env)
    attempt = tmp_path / "failed-attempt"

    def fail_reset(
        self: SharedPortfolioGymEnv,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        raise RuntimeError("literal synthetic reset failure")

    monkeypatch.setattr(SharedPortfolioGymEnv, "reset", fail_reset)
    with pytest.raises(RuntimeError, match="literal synthetic reset failure"):
        fit_shared_portfolio_ppo(
            env,
            transform=transform,
            output_root=attempt,
            config=SharedPortfolioPPOConfig(
                total_timesteps=4,
                n_steps=4,
                batch_size=2,
                n_epochs=1,
                reset_seed=29,
                ppo_seed=31,
            ),
        )

    failure = json.loads((attempt / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert not (attempt / "completion.json").exists()
    output = tmp_path / "must-not-reserve"
    from trade_rl.strategies.rl.portfolio_inference_bundle import (
        publish_portfolio_inference_bundle,
    )

    with pytest.raises((FileNotFoundError, ValueError), match="completion|completed"):
        publish_portfolio_inference_bundle(
            attempt,
            output,
            expected_manifest_sha256=hashlib.sha256(
                (attempt / "manifest.json").read_bytes()
            ).hexdigest(),
            expected_completion_sha256="1" * 64,
            expected_training_context_sha256=hashlib.sha256(
                (attempt / "training_context.json").read_bytes()
            ).hexdigest(),
            expected_checkpoint_sha256="2" * 64,
        )
    assert not output.exists()
