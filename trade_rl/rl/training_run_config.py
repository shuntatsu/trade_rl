"""Authored training-run configuration contract and stable identity."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

from trade_rl.artifacts.signals import SignalKind, load_signal_artifact
from trade_rl.domain.config_fields import (
    require_dataclass_fields,
    require_exact_fields,
)
from trade_rl.risk.emergency import EmergencyRiskConfig
from trade_rl.risk.portfolio import PortfolioRiskConfig
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.rl.actions import ActionSpec, AlphaContract
from trade_rl.rl.checkpointing import load_checkpoint_manifest
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.rewards import RewardConfig
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig


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


TRAINING_RUN_CONFIG_SCHEMA = "training_run_config_v4"
_REQUIRED_V4_TRAINING_FIELDS = frozenset(
    {
        "policy_actor_head",
        "hierarchical_gate_temperature",
        "behavior_cloning_gate_loss_weight",
        "behavior_cloning_target_loss_weight",
        "behavior_cloning_composed_loss_weight",
        "behavior_cloning_gate_change_threshold",
        "behavior_cloning_max_positive_class_weight",
        "behavior_cloning_min_gate_precision",
        "behavior_cloning_min_gate_recall",
        "behavior_cloning_max_active_target_rmse",
        "behavior_cloning_min_activity_ratio",
        "behavior_cloning_max_activity_ratio",
        "behavior_cloning_min_causal_holdout_trades",
        "behavior_cloning_max_causal_holdout_regret",
        "behavior_cloning_causal_holdout_bootstrap_resamples",
        "behavior_cloning_causal_holdout_confidence_level",
    }
)
_REQUIRED_V4_EXECUTION_FIELDS = frozenset(
    item.name for item in dataclass_fields(ExecutionCostConfig) if item.init
)
_LEGACY_TRAINING_RUN_CONFIG_SCHEMAS = frozenset(
    {"training_run_config_v1", "training_run_config_v2", "training_run_config_v3"}
)


def _require_v4_training_fields(payload: dict[str, Any]) -> None:
    missing = sorted(_REQUIRED_V4_TRAINING_FIELDS - set(payload))
    if missing:
        raise ValueError(
            "training_run_config_v4 training is missing required field(s): "
            + ", ".join(missing)
        )


def _signal_artifact_digest(path: Path | None, *, kind: SignalKind) -> str | None:
    if path is None:
        return None
    manifest, _ = load_signal_artifact(path, expected_kind=kind)
    return manifest.artifact_digest


@dataclass(frozen=True, slots=True)
class TrainingRunConfig:
    training: ResidualTrainingConfig
    environment: ResidualMarketEnvConfig
    risk: PreTradeRiskConfig
    reward: RewardConfig
    trend: TrendConfig
    action: ActionSpec
    alpha_contract: AlphaContract
    portfolio_risk: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)
    alpha_artifact: Path | None = None
    factor_artifact: Path | None = None
    resume_checkpoints: tuple[tuple[int, Path], ...] = ()
    transfer_checkpoints: tuple[tuple[int, Path], ...] = ()
    export_onnx: bool = False
    export_torchscript: bool = False
    export_structured_torchscript: bool = False
    export_tolerance: float = 1e-5
    git_commit: str | None = None
    schema_version: str = TRAINING_RUN_CONFIG_SCHEMA
    git_dirty: bool | None = None

    def __post_init__(self) -> None:
        if not self.environment.require_full_reward_preroll:
            raise ValueError(
                "training requires require_full_reward_preroll=true; "
                "configuration is not rewritten implicitly"
            )
        if self.environment.resolved_reward_config() != self.reward:
            raise ValueError("environment reward configuration differs from run reward")
        resume_seeds = tuple(seed for seed, _ in self.resume_checkpoints)
        if len(set(resume_seeds)) != len(resume_seeds):
            raise ValueError("resume checkpoint seeds must be unique")
        if any(seed not in self.training.seeds for seed in resume_seeds):
            raise ValueError("resume checkpoint seed is outside training seeds")
        transfer_seeds = tuple(seed for seed, _ in self.transfer_checkpoints)
        if len(set(transfer_seeds)) != len(transfer_seeds):
            raise ValueError("transfer checkpoint seeds must be unique")
        if any(seed not in self.training.seeds for seed in transfer_seeds):
            raise ValueError("transfer checkpoint seed is outside training seeds")
        if set(resume_seeds) & set(transfer_seeds):
            raise ValueError(
                "checkpoint resume and transfer cannot target the same seed"
            )
        if self.action.alpha_enabled != (self.alpha_artifact is not None):
            raise ValueError("alpha action requires exactly one alpha artifact")
        if (self.action.n_factors > 0) != (self.factor_artifact is not None):
            raise ValueError("factor actions require exactly one factor artifact")
        if any(
            not isinstance(value, bool)
            for value in (
                self.export_onnx,
                self.export_torchscript,
                self.export_structured_torchscript,
            )
        ):
            raise ValueError("export flags must be booleans")
        if not math.isfinite(self.export_tolerance) or self.export_tolerance <= 0.0:
            raise ValueError("export_tolerance must be finite and positive")
        if self.training.observation_encoder == "hierarchical_sequence_v2" and (
            self.export_onnx or self.export_torchscript
        ):
            raise ValueError(
                "structured sequence policies do not support flat ONNX/TorchScript export"
            )
        if self.export_structured_torchscript and (
            self.training.observation_encoder != "hierarchical_sequence_v2"
        ):
            raise ValueError(
                "structured TorchScript export requires hierarchical_sequence_v2"
            )
        if self.git_commit is not None and not self.git_commit:
            raise ValueError("git_commit must be non-empty when provided")
        if self.git_dirty is not None and not isinstance(self.git_dirty, bool):
            raise ValueError("git_dirty must be a boolean or null")
        if self.schema_version in _LEGACY_TRAINING_RUN_CONFIG_SCHEMAS:
            raise ValueError(
                f"migrate {self.schema_version} to {TRAINING_RUN_CONFIG_SCHEMA}"
            )
        if self.schema_version != TRAINING_RUN_CONFIG_SCHEMA:
            raise ValueError(
                f"unsupported training run configuration schema; expected "
                f"{TRAINING_RUN_CONFIG_SCHEMA}"
            )

    @classmethod
    def from_mapping(cls, raw: object) -> TrainingRunConfig:
        payload = require_exact_fields(
            _mapping(raw, field="training run config"),
            required={
                "schema_version",
                "training",
                "environment",
                "risk",
                "reward",
                "trend",
                "action",
                "execution",
            },
            optional={
                "portfolio_risk",
                "alpha_contract",
                "alpha_artifact",
                "factor_artifact",
                "resume_checkpoints",
                "transfer_checkpoints",
                "exports",
                "git_commit",
                "git_dirty",
            },
            field="training run config",
        )
        schema_version = payload["schema_version"]
        if not isinstance(schema_version, str):
            raise ValueError("schema_version must be a string")
        if schema_version in _LEGACY_TRAINING_RUN_CONFIG_SCHEMAS:
            raise ValueError(
                f"migrate {schema_version} to {TRAINING_RUN_CONFIG_SCHEMA}"
            )
        if schema_version != TRAINING_RUN_CONFIG_SCHEMA:
            raise ValueError(
                f"unsupported training run configuration schema; expected "
                f"{TRAINING_RUN_CONFIG_SCHEMA}"
            )

        training_mapping = _mapping(payload["training"], field="training")
        _require_v4_training_fields(training_mapping)
        training_data = _tuple_fields(
            require_dataclass_fields(
                training_mapping,
                ResidualTrainingConfig,
                field="training",
            ),
            "seeds",
            "policy_net_arch",
            "value_net_arch",
            "cost_continuous_hidden_dims",
            "cost_event_hidden_dims",
            "lagrangian_budgets",
            "lagrangian_dual_learning_rates",
            "lagrangian_ema_betas",
            "lagrangian_initial_multipliers",
            "lagrangian_max_multipliers",
            "lagrangian_warmup_rollouts",
            "lagrangian_update_interval_rollouts",
            "lagrangian_minimum_completed_episodes",
        )
        reward_data = require_dataclass_fields(
            _mapping(payload["reward"], field="reward"),
            RewardConfig,
            field="reward",
        )
        reward = RewardConfig(**reward_data)
        execution_data = _tuple_fields(
            require_exact_fields(
                _mapping(payload["execution"], field="execution"),
                required=_REQUIRED_V4_EXECUTION_FIELDS,
                optional=set(),
                field="execution",
            ),
            "trigger_volume_fractions",
        )
        execution = ExecutionCostConfig(**execution_data)
        environment_mapping = _mapping(payload["environment"], field="environment")
        environment_data = _tuple_fields(
            require_dataclass_fields(
                environment_mapping,
                ResidualMarketEnvConfig,
                field="environment",
                excluded={"reward_config", "reward", "execution_cost"},
            ),
            "episode_hour_choices",
            "initial_state_modes",
            "sequence_windows",
        )
        if "require_full_reward_preroll" not in environment_mapping:
            raise ValueError(
                "environment has missing required fields: require_full_reward_preroll"
            )
        emergency_risk_data = require_dataclass_fields(
            _mapping(
                environment_data.pop("emergency_risk", {}),
                field="emergency_risk",
            ),
            EmergencyRiskConfig,
            field="emergency_risk",
        )
        emergency_risk = EmergencyRiskConfig(**emergency_risk_data)
        exports = require_exact_fields(
            _mapping(payload.get("exports"), field="exports"),
            required=set(),
            optional={
                "onnx",
                "structured_torchscript",
                "torchscript",
                "tolerance",
            },
            field="exports",
        )
        git_commit = payload.get("git_commit")
        if git_commit is not None and not isinstance(git_commit, str):
            raise ValueError("git_commit must be a string or null")
        git_dirty = payload.get("git_dirty")
        if git_dirty is not None and not isinstance(git_dirty, bool):
            raise ValueError("git_dirty must be a boolean or null")
        raw_alpha_artifact = payload.get("alpha_artifact")
        raw_factor_artifact = payload.get("factor_artifact")
        raw_resume_checkpoints = payload.get("resume_checkpoints", {})
        if not isinstance(raw_resume_checkpoints, dict):
            raise ValueError("resume_checkpoints must be a JSON object")
        resume_checkpoints: list[tuple[int, Path]] = []
        for raw_seed, raw_path in raw_resume_checkpoints.items():
            if not isinstance(raw_seed, str) or not raw_seed.isdigit():
                raise ValueError("resume checkpoint seed keys must be integers")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("resume checkpoint paths must be non-empty strings")
            resume_checkpoints.append((int(raw_seed), Path(raw_path)))
        raw_transfer_checkpoints = payload.get("transfer_checkpoints", {})
        if not isinstance(raw_transfer_checkpoints, dict):
            raise ValueError("transfer_checkpoints must be a JSON object")
        transfer_checkpoints: list[tuple[int, Path]] = []
        for raw_seed, raw_path in raw_transfer_checkpoints.items():
            if not isinstance(raw_seed, str) or not raw_seed.isdigit():
                raise ValueError("transfer checkpoint seed keys must be integers")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("transfer checkpoint paths must be non-empty strings")
            transfer_checkpoints.append((int(raw_seed), Path(raw_path)))
        if raw_alpha_artifact is not None and not isinstance(raw_alpha_artifact, str):
            raise ValueError("alpha_artifact must be a path string or null")
        if raw_factor_artifact is not None and not isinstance(raw_factor_artifact, str):
            raise ValueError("factor_artifact must be a path string or null")

        risk_data = require_dataclass_fields(
            _mapping(payload["risk"], field="risk"),
            PreTradeRiskConfig,
            field="risk",
        )
        portfolio_risk_data = require_dataclass_fields(
            _mapping(payload.get("portfolio_risk"), field="portfolio_risk"),
            PortfolioRiskConfig,
            field="portfolio_risk",
        )
        trend_data = require_dataclass_fields(
            _mapping(payload["trend"], field="trend"),
            TrendConfig,
            field="trend",
        )
        action_data = require_dataclass_fields(
            _mapping(payload["action"], field="action"),
            ActionSpec,
            field="action",
        )
        alpha_contract_data = require_dataclass_fields(
            _mapping(payload.get("alpha_contract"), field="alpha_contract"),
            AlphaContract,
            field="alpha_contract",
        )
        return cls(
            training=ResidualTrainingConfig(**training_data),
            environment=ResidualMarketEnvConfig(
                **environment_data,
                reward_config=reward,
                emergency_risk=emergency_risk,
                execution_cost=execution,
            ),
            risk=PreTradeRiskConfig(**risk_data),
            reward=reward,
            portfolio_risk=PortfolioRiskConfig(**portfolio_risk_data),
            trend=TrendConfig(**trend_data),
            action=ActionSpec(**action_data),
            alpha_contract=AlphaContract(**alpha_contract_data),
            alpha_artifact=(
                None if raw_alpha_artifact is None else Path(raw_alpha_artifact)
            ),
            factor_artifact=(
                None if raw_factor_artifact is None else Path(raw_factor_artifact)
            ),
            resume_checkpoints=tuple(sorted(resume_checkpoints)),
            transfer_checkpoints=tuple(sorted(transfer_checkpoints)),
            export_onnx=_boolean(
                exports.get("onnx"), field="exports.onnx", default=False
            ),
            export_torchscript=_boolean(
                exports.get("torchscript"),
                field="exports.torchscript",
                default=False,
            ),
            export_structured_torchscript=_boolean(
                exports.get("structured_torchscript"),
                field="exports.structured_torchscript",
                default=False,
            ),
            export_tolerance=float(exports.get("tolerance", 1e-5)),
            git_commit=git_commit,
            schema_version=schema_version,
            git_dirty=git_dirty,
        )

    @property
    def alpha_artifact_digest(self) -> str | None:
        """Return the validated alpha artifact digest, when configured."""

        return _signal_artifact_digest(self.alpha_artifact, kind="alpha")

    @property
    def factor_artifact_digest(self) -> str | None:
        """Return the validated factor artifact digest, when configured."""

        return _signal_artifact_digest(self.factor_artifact, kind="factor")

    def resolve_artifact_paths(self, base: Path) -> TrainingRunConfig:
        """Resolve relative signal artifacts against the owning config directory."""

        def resolved(value: Path | None) -> Path | None:
            if value is None or value.is_absolute():
                return value
            return base / value

        return replace(
            self,
            alpha_artifact=resolved(self.alpha_artifact),
            factor_artifact=resolved(self.factor_artifact),
            resume_checkpoints=tuple(
                (seed, resolved(path) or path) for seed, path in self.resume_checkpoints
            ),
            transfer_checkpoints=tuple(
                (seed, resolved(path) or path)
                for seed, path in self.transfer_checkpoints
            ),
        )

    @classmethod
    def from_json(cls, path: Path) -> TrainingRunConfig:
        config = cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        return config.resolve_artifact_paths(path.parent)

    def _recipe_identity_payload(
        self,
        *,
        resume_checkpoint_digests: dict[str, str],
        transfer_checkpoint_digests: dict[str, str],
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "action": asdict(self.action),
            "alpha_contract": asdict(self.alpha_contract),
            "alpha_artifact_digest": self.alpha_artifact_digest,
            "environment": asdict(self.environment),
            "factor_artifact_digest": self.factor_artifact_digest,
            "portfolio_risk": asdict(self.portfolio_risk),
            "risk": asdict(self.risk),
            "reward": asdict(self.reward),
            "resume_checkpoint_digests": resume_checkpoint_digests,
            "schema_version": self.schema_version,
            "training": self.training.digest_payload(),
            "trend": asdict(self.trend),
        }
        if transfer_checkpoint_digests:
            payload["transfer_checkpoint_digests"] = transfer_checkpoint_digests
        return payload

    def candidate_digest_payload(self) -> dict[str, object]:
        """Return the stable learning recipe identity, excluding checkpoint transport."""

        return self._recipe_identity_payload(
            resume_checkpoint_digests={},
            transfer_checkpoint_digests={},
        )

    def digest_payload(self) -> dict[str, object]:
        return self._recipe_identity_payload(
            resume_checkpoint_digests={
                str(seed): load_checkpoint_manifest(
                    path / "checkpoint.json" if path.is_dir() else path
                ).digest
                for seed, path in self.resume_checkpoints
            },
            transfer_checkpoint_digests={
                str(seed): load_checkpoint_manifest(
                    path / "checkpoint.json" if path.is_dir() else path
                ).digest
                for seed, path in self.transfer_checkpoints
            },
        )


__all__ = ["TRAINING_RUN_CONFIG_SCHEMA", "TrainingRunConfig"]
