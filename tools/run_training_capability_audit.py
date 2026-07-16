#!/usr/bin/env python3
"""Execute short real-training probes for every maintained learning backend."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.integrations.binance import binance_multitimeframe_feature_specs
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.actions import ActionSpec, AlphaContract
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.export import export_policy_actor
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import PolicyTrainingResult, ResidualTrainingConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

_ENVIRONMENT_DIGEST = "e" * 64
_ACTION_NAMES = ("tilt",)
_ACTION_SPEC_DIGEST = content_digest({"names": _ACTION_NAMES})


class AuditEnv(gym.Env[np.ndarray, np.ndarray]):
    """Tiny deterministic environment with the metadata required by the backend."""

    metadata = {"render_modes": []}
    environment_digest = _ENVIRONMENT_DIGEST
    initial_capital = 100_000.0
    decision_hours = 1.0
    action_names = _ACTION_NAMES
    action_spec_digest = _ACTION_SPEC_DIGEST
    asset_active_column = 1
    layout = ObservationLayout(
        n_symbols=1,
        n_features=1,
        action_size=1,
        n_factors=0,
        per_symbol_width=2,
        global_width=1,
    )

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        self._step = 0
        return np.asarray((0.0, 1.0, 0.0), dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._step += 1
        phase = self._step / 8.0
        observation = np.asarray(
            (math.sin(phase), 1.0, math.cos(phase)), dtype=np.float32
        )
        target = math.sin(phase * 0.5)
        reward = -float(np.square(action - target).sum())
        return observation, reward, self._step >= 8, False, {}


def _linear_widths(module: nn.Module) -> list[int]:
    return [
        int(item.out_features)
        for item in module.modules()
        if isinstance(item, nn.Linear)
    ]


def _architecture(algorithm: str, checkpoint: Path) -> dict[str, list[int]]:
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(str(checkpoint), device="cpu")
        return {
            "actor": _linear_widths(model.policy.mlp_extractor.policy_net),
            "critic": _linear_widths(model.policy.mlp_extractor.value_net),
        }
    if algorithm == "sac":
        from stable_baselines3 import SAC

        model = SAC.load(str(checkpoint), device="cpu")
        return {
            "actor": _linear_widths(model.policy.actor.latent_pi),
            "critic": _linear_widths(model.policy.critic.q_networks[0])[:-1],
        }
    if algorithm == "td3":
        from stable_baselines3 import TD3

        model = TD3.load(str(checkpoint), device="cpu")
        return {
            "actor": _linear_widths(model.policy.actor.mu)[:-1],
            "critic": _linear_widths(model.policy.critic.q_networks[0])[:-1],
        }
    if algorithm == "tqc":
        from sb3_contrib import TQC

        model = TQC.load(str(checkpoint), device="cpu")
        return {
            "actor": _linear_widths(model.policy.actor.latent_pi),
            "critic": _linear_widths(model.policy.critic.q_networks[0])[:-1],
        }
    raise ValueError(f"unsupported algorithm: {algorithm}")


def _config(algorithm: str) -> ResidualTrainingConfig:
    common: dict[str, object] = {
        "timesteps": 16,
        "gamma": 0.99,
        "seeds": (0,),
        "algorithm": algorithm,
        "batch_size": 8,
        "learning_rate": 3e-4,
        "policy_net_arch": (16, 8),
        "value_net_arch": (24, 12),
        "checkpoint_interval_steps": 8,
        "max_checkpoints": 2,
        "device": "cpu",
    }
    if algorithm == "ppo":
        common.update(
            {
                "n_steps": 8,
                "n_epochs": 1,
                "asset_set_encoder": True,
                "asset_embedding_dim": 8,
                "global_embedding_dim": 8,
            }
        )
    else:
        common.update(
            {
                "buffer_size": 64,
                "learning_starts": 0,
                "train_freq": 1,
                "gradient_steps": 1,
                "asset_set_encoder": False,
            }
        )
        if algorithm in {"sac", "tqc"}:
            common.update({"use_sde": True, "sde_sample_freq": 4})
    return ResidualTrainingConfig(**common)  # type: ignore[arg-type]


def _train_algorithm(
    root: Path, algorithm: str
) -> tuple[dict[str, object], PolicyTrainingResult]:
    config = _config(algorithm)
    output = root / algorithm / "policy.zip"
    result = StableBaselines3Backend(AuditEnv).train(
        seed=0,
        config=config,
        output_path=output,
    )
    if result.actual_timesteps < config.timesteps:
        raise RuntimeError(f"{algorithm} stopped before requested timesteps")
    if not output.is_file():
        raise RuntimeError(f"{algorithm} did not publish a policy")
    checkpoints = sorted((output.parent / "checkpoints").glob("step-*/checkpoint.json"))
    if not checkpoints:
        raise RuntimeError(f"{algorithm} did not publish checkpoints")
    observed_architecture = _architecture(algorithm, output)
    expected_architecture = {
        "actor": list(config.policy_net_arch),
        "critic": list(config.value_net_arch),
    }
    if observed_architecture != expected_architecture:
        raise RuntimeError(
            f"{algorithm} architecture mismatch: "
            f"{observed_architecture!r} != {expected_architecture!r}"
        )
    if algorithm != "ppo" and result.replay_buffer_path is None:
        raise RuntimeError(f"{algorithm} did not publish a replay buffer")
    return (
        {
            "actual_timesteps": result.actual_timesteps,
            "architecture": observed_architecture,
            "checkpoint_count": len(checkpoints),
            "parameter_count": result.parameter_count,
            "replay_buffer_digest": result.replay_buffer_digest,
            "resolved_device": result.resolved_device,
            "sde_enabled": config.use_sde,
            "status": "pass",
        },
        result,
    )


def _resume_ppo(root: Path) -> dict[str, object]:
    config = _config("ppo")
    source = root / "ppo" / "checkpoints"
    checkpoints = sorted(source.glob("step-*/checkpoint.json"))
    if not checkpoints:
        raise RuntimeError("PPO checkpoint is unavailable for resume")
    resume_root = checkpoints[0].parent
    output = root / "ppo-resume" / "policy.zip"
    result = StableBaselines3Backend(
        AuditEnv,
        resume_checkpoint_artifacts={0: resume_root},
    ).train(seed=0, config=config, output_path=output)
    if result.actual_timesteps != config.timesteps:
        raise RuntimeError("PPO resume did not reach the requested total")
    if not (output.parent / "resume.json").is_file():
        raise RuntimeError("PPO resume evidence is missing")
    return {
        "actual_timesteps": result.actual_timesteps,
        "resume_checkpoint": resume_root.as_posix(),
        "status": "pass",
    }


def _resume_replay(root: Path, source: PolicyTrainingResult) -> dict[str, object]:
    if source.replay_buffer_path is None:
        raise RuntimeError("SAC replay artifact is unavailable")
    replay_root = source.replay_buffer_path.parent
    config = _config("sac")
    output = root / "sac-replay-resume" / "policy.zip"
    result = StableBaselines3Backend(
        AuditEnv,
        resume_replay_artifact=replay_root,
    ).train(seed=0, config=config, output_path=output)
    if result.actual_timesteps < config.timesteps:
        raise RuntimeError("SAC replay resume did not complete")
    return {
        "actual_timesteps": result.actual_timesteps,
        "source_replay_digest": source.replay_buffer_digest,
        "status": "pass",
    }


def _export_ppo(root: Path) -> dict[str, object]:
    checkpoint = root / "ppo" / "policy.zip"
    manifest = export_policy_actor(
        checkpoint_path=checkpoint,
        output_dir=root / "ppo-exports",
        algorithm="ppo",
        observation_size=3,
        action_size=1,
        action_spec_digest=_ACTION_SPEC_DIGEST,
        normalizer_digest=None,
        onnx=True,
        torchscript=True,
        tolerance=1e-5,
    )
    records = {item.format: asdict(item) for item in manifest.exports}
    if records["onnx"]["status"] != "verified":
        raise RuntimeError("ONNX export was not verified")
    if not any(item["status"] == "verified" for item in records.values()):
        raise RuntimeError("no deterministic actor export was verified")
    return {
        "manifest_digest": manifest.digest,
        "records": records,
        "status": "pass",
    }


def _market_dataset(*, n_bars: int = 160, n_symbols: int = 3) -> MarketDataset:
    phase = np.arange(n_bars, dtype=np.float64)
    close_columns = []
    for index in range(n_symbols):
        returns = 0.0002 * (index - 1) + 0.001 * np.sin(phase / (11 + index))
        close_columns.append(100.0 * np.exp(np.cumsum(returns)))
    close = np.column_stack(close_columns)
    open_price = np.vstack((close[0], close[:-1]))
    features = np.stack(
        (
            np.sin(phase / 7.0),
            np.cos(phase / 13.0),
            np.tanh((phase - n_bars / 2.0) / 30.0),
        ),
        axis=1,
    )[:, None, :]
    features = np.repeat(features, n_symbols, axis=1).astype(np.float32)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=tuple(f"ASSET{index}" for index in range(n_symbols)),
        timestamps=np.datetime64("2025-01-01T01:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.stack(
            (np.sin(phase / 17.0), np.cos(phase / 29.0)), axis=1
        ).astype(np.float32),
        open=open_price,
        high=np.maximum(open_price, close) * 1.001,
        low=np.minimum(open_price, close) * 0.999,
        close=close,
        volume=np.full((n_bars, n_symbols), 1_000_000.0),
        funding_rate=np.zeros((n_bars, n_symbols), dtype=np.float64),
        tradable=np.ones((n_bars, n_symbols), dtype=np.bool_),
        feature_available=np.ones(features.shape, dtype=np.bool_),
        feature_names=("cycle_fast", "cycle_slow", "regime"),
        global_feature_names=("global_cycle", "global_risk"),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, n_symbols), 0.0001),
        spread_rate=np.full((n_bars, n_symbols), 0.0001),
        max_participation_rate=np.full((n_bars, n_symbols), 0.10),
        borrow_available=np.ones((n_bars, n_symbols), dtype=np.bool_),
    )
    return dataset.with_content_identity({"source": "training-capability-audit-v1"})


class AuditAlphaProvider:
    artifact_digest = "a" * 64

    def predict_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        phase = index / 9.0
        values = np.asarray(
            [math.sin(phase + offset) for offset in range(dataset.n_symbols)],
            dtype=np.float64,
        )
        return values / max(float(np.abs(values).sum()), 1e-12)


class AuditFactorProvider:
    artifact_digest = "f" * 64
    n_factors = 2

    def basis_at(self, dataset: MarketDataset, index: int) -> np.ndarray:
        del index
        first = np.linspace(-1.0, 1.0, dataset.n_symbols)
        second = np.cos(np.linspace(0.0, math.pi, dataset.n_symbols))
        return np.stack((first, second), axis=0)


def _residual_feature_training(root: Path) -> dict[str, object]:
    dataset = _market_dataset()
    action = ActionSpec(
        mode="residual",
        alpha_enabled=True,
        risk_tilt_enabled=True,
        n_factors=2,
    )

    def factory() -> ResidualMarketEnv:
        return ResidualMarketEnv(
            dataset,
            trend_strategy=TrendStrategy(
                TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
            ),
            alpha_provider=AuditAlphaProvider(),
            alpha_enabled=True,
            alpha_artifact_digest=AuditAlphaProvider.artifact_digest,
            alpha_contract=AlphaContract(kind="target_weight"),
            factor_basis_provider=AuditFactorProvider(),
            factor_artifact_digest=AuditFactorProvider.artifact_digest,
            factor_count=2,
            action_spec=action,
            config=ResidualMarketEnvConfig(
                episode_bars=16,
                decision_every=1,
                initial_capital=100_000.0,
                finite_horizon_observation=True,
                liquidate_on_end=True,
                execution_cost=ExecutionCostConfig.zero(),
            ),
        )

    config = ResidualTrainingConfig(
        timesteps=16,
        gamma=0.99,
        seeds=(0,),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_net_arch=(16, 8),
        value_net_arch=(24, 12),
        asset_set_encoder=False,
        device="cpu",
    )
    output = root / "residual-all-controls" / "policy.zip"
    result = StableBaselines3Backend(factory).train(
        seed=0,
        config=config,
        output_path=output,
    )
    if result.action_names != action.names:
        raise RuntimeError("residual action controls were not bound into the model")
    if result.alpha_artifact_digest != AuditAlphaProvider.artifact_digest:
        raise RuntimeError("alpha artifact identity was not consumed")
    if result.factor_artifact_digest != AuditFactorProvider.artifact_digest:
        raise RuntimeError("factor artifact identity was not consumed")
    return {
        "action_names": result.action_names,
        "action_size": result.action_size,
        "alpha_artifact_digest": result.alpha_artifact_digest,
        "factor_artifact_digest": result.factor_artifact_digest,
        "status": "pass",
    }


def _behavior_cloning_training(root: Path) -> dict[str, object]:
    dataset = _market_dataset(n_bars=120, n_symbols=2)
    action = ActionSpec(
        mode="target_weight",
        risk_tilt_enabled=False,
        target_weight_count=dataset.n_symbols,
    )

    def factory() -> ResidualMarketEnv:
        return ResidualMarketEnv(
            dataset,
            trend_strategy=TrendStrategy(
                TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
            ),
            action_spec=action,
            config=ResidualMarketEnvConfig(
                episode_bars=8,
                decision_every=1,
                initial_capital=100_000.0,
                liquidate_on_end=True,
                execution_cost=ExecutionCostConfig.zero(),
            ),
        )

    config = ResidualTrainingConfig(
        timesteps=8,
        gamma=0.99,
        seeds=(0,),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_net_arch=(16, 8),
        value_net_arch=(24, 12),
        asset_set_encoder=False,
        behavior_cloning_epochs=1,
        behavior_cloning_batch_size=16,
        behavior_cloning_validation_fraction=0.1,
        device="cpu",
    )
    output = root / "behavior-cloning" / "policy.zip"
    result = StableBaselines3Backend(factory).train(
        seed=0,
        config=config,
        output_path=output,
    )
    payload_path = output.parent / "behavior-cloning.json"
    teacher_manifest = output.parent / "teacher" / "manifest.json"
    if not payload_path.is_file() or not teacher_manifest.is_file():
        raise RuntimeError("behavior cloning artifacts are missing")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not np.isfinite(float(payload["initial_mse"])):
        raise RuntimeError("behavior cloning initial MSE is invalid")
    if not np.isfinite(float(payload["final_mse"])):
        raise RuntimeError("behavior cloning final MSE is invalid")
    return {
        "actual_timesteps": result.actual_timesteps,
        "final_mse": payload["final_mse"],
        "initial_mse": payload["initial_mse"],
        "sample_count": payload["sample_count"],
        "status": "pass",
    }


def _sequence_dataset(n_bars: int = 640) -> MarketDataset:
    specs = binance_multitimeframe_feature_specs(
        base_timeframe="15m", feature_timeframes=("1h", "4h", "1d")
    )
    phase = np.arange(n_bars, dtype=np.float64)
    returns = 0.00005 + 0.0004 * np.sin(phase / 47.0)
    close = 30_000.0 * np.exp(np.cumsum(returns))
    open_price = np.concatenate(([close[0]], close[:-1]))
    spread = 0.001 + 0.0002 * np.cos(phase / 19.0)
    features = np.stack(
        tuple(
            np.sin(phase / float(11 + index % 97))
            + 0.1 * np.cos(phase / float(7 + index % 43))
            for index in range(len(specs))
        ),
        axis=1,
    ).astype(np.float32)[:, None, :]
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2025-01-01T00:15:00", "ns")
        + np.arange(n_bars) * np.timedelta64(15, "m"),
        features=features,
        global_features=np.stack(
            (np.sin(phase / 97.0), np.cos(phase / 193.0)), axis=1
        ).astype(np.float32),
        open=open_price[:, None],
        high=(np.maximum(open_price, close) * (1.0 + spread))[:, None],
        low=(np.minimum(open_price, close) * (1.0 - spread))[:, None],
        close=close[:, None],
        volume=(1_000.0 + 100.0 * np.sin(phase / 13.0))[:, None],
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones(features.shape, dtype=np.bool_),
        feature_names=tuple(spec.name for spec in specs),
        global_feature_names=("market_cycle", "risk_cycle"),
        periods_per_year=35_040,
        fee_rate=np.full((n_bars, 1), 0.0001),
        spread_rate=np.full((n_bars, 1), 0.0001),
        max_participation_rate=np.full((n_bars, 1), 0.10),
        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
    )
    return dataset.with_content_identity({"source": "sequence-capability-audit-v1"})


def _sequence_training(root: Path) -> dict[str, object]:
    dataset = _sequence_dataset()
    action = ActionSpec(
        mode="target_weight",
        risk_tilt_enabled=False,
        target_weight_count=1,
    )

    def factory() -> ResidualMarketEnv:
        return ResidualMarketEnv(
            dataset,
            trend_strategy=TrendStrategy(
                TrendConfig(fast_lookback=4, base_lookback=8, slow_lookback=16)
            ),
            action_spec=action,
            config=ResidualMarketEnvConfig(
                episode_hours=2.0,
                decision_hours=0.25,
                episode_bars=8,
                decision_every=1,
                initial_capital=100_000.0,
                finite_horizon_observation=True,
                structured_sequence_observation=True,
                sequence_windows=(("15m", 8), ("1h", 8), ("4h", 6), ("1d", 3)),
                liquidate_on_end=True,
                execution_cost=ExecutionCostConfig.zero(),
            ),
        )

    config = ResidualTrainingConfig(
        timesteps=8,
        gamma=0.99,
        seeds=(0,),
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy="MultiInputPolicy",
        policy_net_arch=(16, 8),
        value_net_arch=(24, 12),
        sequence_encoder=True,
        sequence_d_model=32,
        sequence_attention_heads=4,
        sequence_attention_layers=1,
        sequence_dropout=0.0,
        max_policy_parameters=2_000_000,
        asset_set_encoder=False,
        device="cpu",
    )
    output = root / "structured-sequence" / "policy.zip"
    result = StableBaselines3Backend(factory).train(
        seed=0,
        config=config,
        output_path=output,
    )
    architecture_path = output.parent / "model-architecture.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    if architecture["architecture"].get("encoder") != "MultiTimeframeTCNEncoder":
        raise RuntimeError("structured sequence encoder was not instantiated")
    return {
        "actual_timesteps": result.actual_timesteps,
        "observation_schema": result.observation_schema,
        "parameter_count": result.parameter_count,
        "sequence_encoder": architecture["architecture"].get("encoder"),
        "status": "pass",
    }


def run_audit(output_root: Path) -> dict[str, object]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    algorithms: dict[str, object] = {}
    results: dict[str, PolicyTrainingResult] = {}
    for algorithm in ("ppo", "sac", "td3", "tqc"):
        record, result = _train_algorithm(output_root, algorithm)
        algorithms[algorithm] = record
        results[algorithm] = result
    report = {
        "algorithms": algorithms,
        "behavior_cloning": _behavior_cloning_training(output_root),
        "exports": _export_ppo(output_root),
        "replay_resume": _resume_replay(output_root, results["sac"]),
        "residual_controls": _residual_feature_training(output_root),
        "schema_version": "full_training_capability_audit_v1",
        "sequence": _sequence_training(output_root),
        "training_resume": _resume_ppo(output_root),
    }
    report["digest"] = content_digest(report)
    report_path = output_root / "audit-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/training-capability-audit"),
    )
    args = parser.parse_args()
    report = run_audit(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
