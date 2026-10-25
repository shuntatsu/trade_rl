from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.rl import portfolio_training as training_module
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_episode_journal import (
    PortfolioEpisodeJournalWrapper,
)
from trade_rl.strategies.rl.portfolio_training import (
    SharedPortfolioPPOConfig,
    _ActorNativeJoinWrapper,
    fit_shared_portfolio_ppo,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioGymObservationWrapper,
    fit_portfolio_observation_transform,
)

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _environment(dataset_id: str = "a" * 64) -> SharedPortfolioGymEnv:
    symbols = ("AAA", "BBB")
    close = np.full((8, len(symbols)), 100.0, dtype=np.float64)
    open_price = close.copy()
    zeros = np.zeros_like(close)
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        len(close)
    ) * np.timedelta64(1, "h")
    dataset = MarketDataset(
        dataset_id=dataset_id,
        symbols=symbols,
        timestamps=timestamps,
        features=np.arange(16, dtype=np.float32).reshape(8, 2, 1),
        global_features=np.arange(8, dtype=np.float32).reshape(8, 1),
        open=open_price,
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
    stop = _START + timedelta(hours=7)
    bound = BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=_START,
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
            economic_horizon_seconds=7 * 3_600,
            rollout_steps=2,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )
    return SharedPortfolioGymEnv(
        dataset,
        start_index=0,
        stop_index=7,
        bound=bound,
        target_abs_weights=(0.5, 0.5),
        execution_cost=ExecutionCostConfig.zero(),
        risk_config=PreTradeRiskConfig(
            max_gross=1.0,
            max_abs_weight=0.5,
            max_turnover=None,
            drawdown_start=0.20,
            drawdown_stop=0.20,
        ),
    )


def _config(**overrides: object) -> SharedPortfolioPPOConfig:
    values: dict[str, object] = {
        "total_timesteps": 8,
        "n_steps": 4,
        "batch_size": 2,
        "n_epochs": 2,
        "reset_seed": 29,
        "ppo_seed": 31,
    }
    values.update(overrides)
    return SharedPortfolioPPOConfig(**values)  # type: ignore[arg-type]


def _json_files(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]


def test_training_config_requires_exact_rollouts_and_valid_advantage_batches() -> None:
    with pytest.raises(ValueError, match="multiple of n_steps"):
        _config(total_timesteps=7)
    with pytest.raises(ValueError, match="at least 2"):
        _config(batch_size=1)
    with pytest.raises(ValueError, match="at least 2"):
        _config(n_steps=1, total_timesteps=2)
    with pytest.raises(ValueError, match="divide n_steps"):
        _config(batch_size=3)


def test_transform_mismatch_fails_before_attempt_reservation_or_environment_reset(
    tmp_path: Path,
) -> None:
    env = _environment()
    mismatched = fit_portfolio_observation_transform(_environment(dataset_id="e" * 64))
    output = tmp_path / "attempt"

    with pytest.raises(ValueError, match="transform context"):
        fit_shared_portfolio_ppo(
            env,
            transform=mismatched,
            output_root=output,
            config=_config(),
        )

    assert env._session is None
    assert not output.exists()


@pytest.mark.parametrize("sample_state", ["missing", "mismatched_action"])
def test_join_rejects_missing_or_mismatched_sample_before_native_step(
    tmp_path: Path, sample_state: str
) -> None:
    env = _environment()
    transform = fit_portfolio_observation_transform(env)
    context_digest = "f" * 64
    episode = PortfolioEpisodeJournalWrapper(
        PortfolioGymObservationWrapper(env, transform),
        root=tmp_path / "native",
        declared_context_digest=context_digest,
    )
    join = _ActorNativeJoinWrapper(
        episode,
        output_root=tmp_path,
        root=tmp_path / "joins",
        context_digest=context_digest,
    )
    join.reset(seed=29)
    policy_journal = SimpleNamespace(
        sample_count=0,
        latest_sample_reference=None,
        root=tmp_path / "policy-samples",
    )
    join.begin_rollout(1, cast(Any, policy_journal))

    if sample_state == "mismatched_action":
        policy_journal.sample_count = 1
        policy_journal.latest_sample_reference = SimpleNamespace(
            sample_id=1,
            raw_action_codes=(0, 0),
            forward_batch_size=1,
        )

    with pytest.raises(RuntimeError, match="fresh batch-one|submitted action"):
        join.step(np.asarray([1, 0], dtype=np.int64))

    assert env._session is not None
    assert env._session.current_index == 0
    assert episode.latest_step_reference is None
    assert join.joined_count == 0
    assert join.is_poisoned
    native_events = _json_files(tmp_path / "native" / "events")
    assert [event["event"] for event in native_events] == [
        "episode_started",
        "episode_reset",
    ]
    join.close()


def test_two_rollouts_bind_actual_policy_samples_to_native_episode_steps(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    env = _environment()
    transform = fit_portfolio_observation_transform(env)
    output = tmp_path / "attempt"

    result = fit_shared_portfolio_ppo(
        env,
        transform=transform,
        output_root=output,
        config=_config(),
    )

    assert result.requested_timesteps == result.realized_timesteps == 8
    assert result.rollout_count == 2
    assert result.optimizer_step_calls == 8
    assert result.epoch_count == 4
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["effective_config"]["gamma"] == 1.0
    assert manifest["effective_config"]["normalize_reward"] is False
    assert manifest["reset_seed"] == 29
    assert manifest["ppo_seed"] == 31
    assert manifest["execution_seed"] == env.execution_cost.random_seed

    expected_policy_version = hashlib.sha256(
        (output / "checkpoints" / "initial.zip").read_bytes()
    ).hexdigest()
    rollout_starts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output / "rollouts").glob("rollout-*/rollout_started.json"))
    ]
    assert len(rollout_starts) == 2
    for rollout_index, rollout in enumerate(rollout_starts, start=1):
        assert rollout["policy_version"] == expected_policy_version
        checkpoint = output / rollout["checkpoint_path"]
        assert (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            == rollout["policy_version"]
        )
        samples = _json_files(output / rollout["sample_directory"] / "samples")
        assert len(samples) == 4
        assert all(
            sample["policy_version"] == rollout["policy_version"] for sample in samples
        )
        assert all(sample["forward_batch_size"] == 1 for sample in samples)
        completed = json.loads(
            (
                output
                / "rollouts"
                / f"rollout-{rollout_index:06d}"
                / "rollout_completed.json"
            ).read_text(encoding="utf-8")
        )
        assert completed["sample_count"] == 4
        assert completed["optimizer_step_calls"] == 4
        assert completed["sb3_epoch_count"] == 2
        expected_policy_version = completed["checkpoint_sha256"]

    joins = _json_files(output / "joins" / "events")
    assert len(joins) == 8
    terminal_joins = [record for record in joins if record["terminated"]]
    assert len(terminal_joins) == 1
    for joined in joins:
        native_path = output / joined["native_event"]["path"]
        sample_path = output / joined["actor_sample"]["path"]
        native_bytes = native_path.read_bytes()
        sample_bytes = sample_path.read_bytes()
        assert (
            hashlib.sha256(native_bytes).hexdigest() == joined["native_event"]["sha256"]
        )
        assert (
            hashlib.sha256(sample_bytes).hexdigest() == joined["actor_sample"]["sha256"]
        )
        native_event = json.loads(native_bytes)
        actor_sample = json.loads(sample_bytes)
        assert (
            actor_sample["raw_action_codes"] == native_event["data"]["raw_action_codes"]
        )
        assert joined["terminated"] is native_event["data"]["terminated"]

    assert (output / "completion.json").is_file()
    native_events = _json_files(output / "native" / "events")
    assert sum(event["event"] == "step_result" for event in native_events) == 8
    assert any(
        event["event"] == "episode_interrupted" and event["data"]["reason"] == "close"
        for event in native_events
    )
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    final_checkpoint = output / completion["final_checkpoint_path"]
    assert (
        hashlib.sha256(final_checkpoint.read_bytes()).hexdigest()
        == result.final_checkpoint_sha256
    )
    loaded = sb3.PPO.load(final_checkpoint)
    assert not hasattr(loaded, "_training_coordinator")


def test_join_write_failure_preserves_unmatched_actor_and_native_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    env = _environment()
    transform = fit_portfolio_observation_transform(env)
    output = tmp_path / "partial-attempt"
    atomic_write = training_module.atomic_write_bytes

    def fail_join_write(path: str | Path, payload: bytes) -> Path:
        destination = Path(path)
        if (
            destination.parent.name == "events"
            and destination.parent.parent.name == "joins"
            and destination.name.startswith("transition-")
        ):
            raise OSError("literal actor/native join persistence failure")
        return atomic_write(destination, payload)

    monkeypatch.setattr(training_module, "atomic_write_bytes", fail_join_write)
    with pytest.raises(OSError, match="actor/native join persistence failure"):
        fit_shared_portfolio_ppo(
            env,
            transform=transform,
            output_root=output,
            config=_config(total_timesteps=4, n_steps=4),
        )

    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["join_poisoned"] is True
    assert failure["joined_transition_count"] == 0
    assert failure["realized_timesteps"] == 0
    assert failure["latest_actor_sample"] is not None
    assert failure["latest_native_event"] is not None
    assert not (output / "completion.json").exists()
    assert not tuple((output / "joins" / "events").glob("transition-*.json"))

    native_path = output / "native" / failure["latest_native_event"]["relative_path"]
    native_event = json.loads(native_path.read_text(encoding="utf-8"))
    assert native_event["event"] == "step_result"

    sample_path = (
        output
        / "rollouts"
        / "rollout-000001"
        / "policy-samples"
        / failure["latest_actor_sample"]["relative_path"]
    )
    assert sample_path.is_file()

    with pytest.raises(FileExistsError, match="already exists"):
        fresh_env = _environment()
        fit_shared_portfolio_ppo(
            fresh_env,
            transform=fit_portfolio_observation_transform(fresh_env),
            output_root=output,
            config=_config(total_timesteps=4, n_steps=4),
        )
    assert fresh_env._session is None


def test_post_rollout_checkpoint_failure_keeps_last_actor_sample_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("torch")
    sb3 = pytest.importorskip("stable_baselines3")
    env = _environment()
    transform = fit_portfolio_observation_transform(env)
    output = tmp_path / "checkpoint-failure"
    original_save = sb3.PPO.save

    def fail_after_rollout_save(
        model: Any, path: str | Path, *args: object, **kwargs: object
    ) -> None:
        if Path(path).name == "after-rollout-000001.zip":
            raise OSError("literal post-rollout checkpoint failure")
        original_save(model, path, *args, **kwargs)

    monkeypatch.setattr(sb3.PPO, "save", fail_after_rollout_save)

    with pytest.raises(OSError, match="post-rollout checkpoint failure"):
        fit_shared_portfolio_ppo(
            env,
            transform=transform,
            output_root=output,
            config=_config(total_timesteps=4, n_steps=4),
        )

    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["realized_timesteps"] == 4
    assert failure["joined_transition_count"] == 4
    assert failure["completed_rollouts"] == []
    assert failure["latest_actor_sample"]["sample_id"] == 4
    assert failure["latest_actor_sample"]["rollout_index"] == 1
    assert failure["latest_native_event"] is not None
    assert not (output / "completion.json").exists()
    assert not (output / "rollouts" / "rollout-000002").exists()

    rollout = output / "rollouts" / "rollout-000001"
    samples = _json_files(rollout / "policy-samples" / "samples")
    assert len(samples) == 4
    last = samples[-1]
    assert failure["latest_actor_sample"]["sample_id"] == last["sample_id"]
    sample_relative_path = f"samples/sample-{last['sample_id']:012d}.json"
    assert failure["latest_actor_sample"]["relative_path"] == sample_relative_path
    sample_path = (
        output
        / failure["latest_actor_sample"]["sample_directory"]
        / failure["latest_actor_sample"]["relative_path"]
    )
    assert sample_path.is_file()
    assert (
        failure["latest_actor_sample"]["sha256"]
        == hashlib.sha256(sample_path.read_bytes()).hexdigest()
    )
    assert not (rollout / "rollout_completed.json").exists()


def test_target_kl_early_stop_records_actual_optimizer_calls(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    env = _environment()
    transform = fit_portfolio_observation_transform(env)

    result = fit_shared_portfolio_ppo(
        env,
        transform=transform,
        output_root=tmp_path / "target-kl",
        config=_config(
            total_timesteps=4,
            n_steps=4,
            n_epochs=8,
            target_kl=1e-12,
            learning_rate=0.03,
            ent_coef=0.5,
        ),
    )

    nominal_maximum_calls = 8 * (4 // 2)
    assert result.realized_timesteps == 4
    assert 0 < result.optimizer_step_calls < nominal_maximum_calls
    assert result.epoch_count < 8
    completion = json.loads(
        (result.output_root / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["optimizer_step_calls"] == result.optimizer_step_calls
    assert completion["sb3_epoch_count"] == result.epoch_count
