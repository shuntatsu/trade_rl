"""End-to-end training-run orchestration and atomic artifact publication."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    validate_training_run_directory,
    write_training_run_manifest,
)
from trade_rl.artifacts.store import ArtifactStore
from trade_rl.data.artifacts import load_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.policies import PolicyEnsembleManifest
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.actions import ActionSpec, AlphaContract
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.rewards import RewardConfig
from trade_rl.rl.training import (
    ResidualTrainingConfig,
    StableBaselines3Backend,
    train_residual_ensemble,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


def _tuple_fields(payload: dict[str, Any], *names: str) -> dict[str, Any]:
    resolved = dict(payload)
    for name in names:
        if name in resolved and isinstance(resolved[name], list):
            resolved[name] = tuple(resolved[name])
    return resolved


def _boolean(value: object, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    training: ResidualTrainingConfig
    environment: ResidualMarketEnvConfig
    risk: PreTradeRiskConfig
    reward: RewardConfig
    trend: TrendConfig
    action: ActionSpec
    alpha_contract: AlphaContract
    export_onnx: bool = False
    export_torchscript: bool = False
    export_tolerance: float = 1e-5
    git_commit: str | None = None
    schema_version: str = "training_run_config_v1"

    def __post_init__(self) -> None:
        if self.environment.resolved_reward_config() != self.reward:
            raise ValueError("environment reward configuration differs from run reward")
        if self.action.alpha_enabled:
            raise ValueError(
                "CLI training with alpha actions requires an alpha artifact adapter"
            )
        if self.action.n_factors:
            raise ValueError(
                "CLI training with factor actions requires a factor artifact adapter"
            )
        if not isinstance(self.export_onnx, bool) or not isinstance(
            self.export_torchscript, bool
        ):
            raise ValueError("export flags must be booleans")
        if not math.isfinite(self.export_tolerance) or self.export_tolerance <= 0.0:
            raise ValueError("export_tolerance must be finite and positive")
        if self.git_commit is not None and not self.git_commit:
            raise ValueError("git_commit must be non-empty when provided")
        if self.schema_version != "training_run_config_v1":
            raise ValueError("unsupported training run configuration schema")

    @classmethod
    def from_mapping(cls, raw: object) -> TrainingRunConfig:
        payload = _mapping(raw, field="training run config")
        training_data = _tuple_fields(
            _mapping(payload.get("training"), field="training"),
            "seeds",
            "policy_net_arch",
        )
        reward = RewardConfig(**_mapping(payload.get("reward"), field="reward"))
        execution = ExecutionCostConfig(
            **_mapping(payload.get("execution"), field="execution")
        )
        environment_data = _tuple_fields(
            _mapping(payload.get("environment"), field="environment"),
            "episode_hour_choices",
            "initial_state_modes",
        )
        environment_data.pop("reward_config", None)
        environment_data.pop("execution_cost", None)
        exports = _mapping(payload.get("exports"), field="exports")
        git_commit = payload.get("git_commit")
        if git_commit is not None and not isinstance(git_commit, str):
            raise ValueError("git_commit must be a string or null")
        schema_version = payload.get("schema_version", "training_run_config_v1")
        if not isinstance(schema_version, str):
            raise ValueError("schema_version must be a string")
        return cls(
            training=ResidualTrainingConfig(**training_data),
            environment=ResidualMarketEnvConfig(
                **environment_data,
                reward_config=reward,
                execution_cost=execution,
            ),
            risk=PreTradeRiskConfig(
                **_mapping(payload.get("risk"), field="risk")
            ),
            reward=reward,
            trend=TrendConfig(**_mapping(payload.get("trend"), field="trend")),
            action=ActionSpec(**_mapping(payload.get("action"), field="action")),
            alpha_contract=AlphaContract(
                **_mapping(payload.get("alpha_contract"), field="alpha_contract")
            ),
            export_onnx=_boolean(
                exports.get("onnx"), field="exports.onnx", default=False
            ),
            export_torchscript=_boolean(
                exports.get("torchscript"),
                field="exports.torchscript",
                default=False,
            ),
            export_tolerance=float(exports.get("tolerance", 1e-5)),
            git_commit=git_commit,
            schema_version=schema_version,
        )

    @classmethod
    def from_json(cls, path: Path) -> TrainingRunConfig:
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def digest_payload(self) -> dict[str, object]:
        return {
            "action": asdict(self.action),
            "alpha_contract": asdict(self.alpha_contract),
            "environment": asdict(self.environment),
            "export_onnx": self.export_onnx,
            "export_tolerance": self.export_tolerance,
            "export_torchscript": self.export_torchscript,
            "git_commit": self.git_commit,
            "risk": asdict(self.risk),
            "reward": asdict(self.reward),
            "schema_version": self.schema_version,
            "training": self.training.digest_payload(),
            "trend": asdict(self.trend),
        }


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    run_id: str
    status: str
    path: Path
    run_digest: str
    policy_digest: str
    dataset_id: str
    production_status: str = "NO-GO"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _dataset_manifest(dataset: MarketDataset, *, created_at: datetime) -> DatasetManifest:
    return DatasetManifest(
        dataset_id=dataset.dataset_id,
        symbols=dataset.symbols,
        feature_names=dataset.feature_names,
        base_timeframe=f"{dataset.bar_hours:g}h",
        bar_hours=dataset.bar_hours,
        created_at=created_at,
    )


def _environment_factory(
    dataset: MarketDataset,
    config: TrainingRunConfig,
) -> Callable[[], ResidualMarketEnv]:
    def create() -> ResidualMarketEnv:
        return ResidualMarketEnv(
            dataset,
            trend_strategy=TrendStrategy(config.trend),
            alpha_enabled=False,
            alpha_contract=config.alpha_contract,
            action_spec=config.action,
            pre_trade_risk=PreTradeRisk(config.risk),
            config=config.environment,
        )

    return create


def _artifact_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "run.json"
        )
    )


def _validate_for_store(path: Path) -> bool:
    validate_training_run_directory(path)
    return True


def _ensemble_payload(manifest: PolicyEnsembleManifest) -> dict[str, object]:
    return asdict(manifest)


def _policy_loader_payload(
    ensemble: PolicyEnsembleManifest,
    *,
    algorithm: str,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "members": tuple(
            f"members/member-{index:03d}/policy.zip"
            for index in range(ensemble.expected_members)
        ),
        "schema_version": "sb3_policy_loader_v1",
    }


def execute_training_run(
    *,
    config_path: Path,
    dataset_path: Path,
    store_root: Path,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> TrainingRunResult:
    """Train, serialize, validate, and atomically publish one ensemble run."""

    resolved_created_at = created_at or datetime.now(UTC)
    resolved_run_id = run_id or resolved_created_at.strftime("run-%Y%m%dT%H%M%SZ")
    config = TrainingRunConfig.from_json(config_path)
    dataset = load_market_dataset_artifact(dataset_path)
    store = ArtifactStore(store_root)
    stage = store.stage_run(resolved_run_id)
    try:
        training_config_digest = content_digest(config.digest_payload())
        _write_json(stage / "training-config.json", config.digest_payload())
        _write_json(
            stage / "dataset-reference.json",
            {
                "artifact_path": str(dataset_path),
                "bar_hours": dataset.bar_hours,
                "dataset_id": dataset.dataset_id,
                "feature_names": dataset.feature_names,
                "global_feature_names": dataset.global_feature_names,
                "schema_version": "dataset_reference_v1",
                "symbols": dataset.symbols,
            },
        )
        _write_json(
            stage / "environment.json",
            {
                "action": asdict(config.action),
                "alpha_contract": asdict(config.alpha_contract),
                "environment": asdict(config.environment),
                "risk": asdict(config.risk),
                "reward": asdict(config.reward),
                "schema_version": "training_environment_v1",
                "trend": asdict(config.trend),
            },
        )
        ensemble = train_residual_ensemble(
            dataset=_dataset_manifest(dataset, created_at=resolved_created_at),
            environment_dataset_id=dataset.dataset_id,
            config=config.training,
            backend=StableBaselines3Backend(_environment_factory(dataset, config)),
            output_dir=stage / "members",
            created_at=resolved_created_at,
        )
        _write_json(stage / "ensemble.json", _ensemble_payload(ensemble))
        _write_json(
            stage / "policy-loader.json",
            _policy_loader_payload(ensemble, algorithm=config.training.algorithm),
        )

        if config.export_onnx or config.export_torchscript:
            from trade_rl.rl.export import export_ensemble_members

            export_ensemble_members(
                root=stage,
                ensemble=ensemble,
                algorithm=config.training.algorithm,
                onnx=config.export_onnx,
                torchscript=config.export_torchscript,
                tolerance=config.export_tolerance,
            )

        run_manifest = TrainingRunManifest.build(
            root=stage,
            run_id=resolved_run_id,
            dataset_id=dataset.dataset_id,
            environment_digest=ensemble.environment_digest,
            ensemble_digest=ensemble.digest,
            training_config_digest=training_config_digest,
            artifact_paths=_artifact_paths(stage),
            created_at=resolved_created_at,
        )
        write_training_run_manifest(stage, run_manifest)
        validate_training_run_directory(stage)
        published = store.publish_run(resolved_run_id, validate=_validate_for_store)
        return TrainingRunResult(
            run_id=resolved_run_id,
            status="published",
            path=published,
            run_digest=run_manifest.digest,
            policy_digest=ensemble.digest,
            dataset_id=dataset.dataset_id,
        )
    except Exception:
        if stage.is_dir():
            store.mark_failed(resolved_run_id)
        raise
