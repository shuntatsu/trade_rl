"""End-to-end training-run orchestration and atomic artifact publication."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.provenance import capture_runtime_provenance
from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    validate_training_run_directory,
    write_training_run_manifest,
)
from trade_rl.artifacts.signals import SignalKind, load_signal_artifact
from trade_rl.artifacts.store import ArtifactStore
from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.data.metadata_promotion import (
    METADATA_PROMOTION_FILE_NAME,
    metadata_promotion_from_dataset,
    write_metadata_promotion_evidence,
)
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.domain.policies import PolicyEnsembleManifest
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.integrations.signal_artifacts import (
    load_alpha_artifact,
    load_factor_artifact,
)
from trade_rl.release.asymmetric import (
    PublicVerificationKey,
    load_public_verification_keys,
)
from trade_rl.risk.emergency import EmergencyRiskConfig
from trade_rl.risk.portfolio import (
    PortfolioRiskConfig,
    PortfolioRiskModel,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.actions import ActionSpec, AlphaContract
from trade_rl.rl.checkpointing import load_checkpoint_manifest
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import observation_passthrough_indices
from trade_rl.rl.rewards import RewardConfig
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequenceWindowSpec,
)
from trade_rl.rl.training import ResidualTrainingConfig, train_residual_ensemble
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy
from trade_rl.workflows.selection_authorization import (
    SelectionAuthorization,
    SelectionProposal,
    load_selection_authorization,
    load_selection_proposal,
)


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
    export_onnx: bool = False
    export_torchscript: bool = False
    export_tolerance: float = 1e-5
    git_commit: str | None = None
    schema_version: str = "training_run_config_v1"
    git_dirty: bool | None = None

    def __post_init__(self) -> None:
        if not self.environment.require_full_reward_preroll:
            object.__setattr__(
                self,
                "environment",
                replace(self.environment, require_full_reward_preroll=True),
            )
        if self.environment.resolved_reward_config() != self.reward:
            raise ValueError("environment reward configuration differs from run reward")
        resume_seeds = tuple(seed for seed, _ in self.resume_checkpoints)
        if len(set(resume_seeds)) != len(resume_seeds):
            raise ValueError("resume checkpoint seeds must be unique")
        if any(seed not in self.training.seeds for seed in resume_seeds):
            raise ValueError("resume checkpoint seed is outside training seeds")
        if self.action.alpha_enabled != (self.alpha_artifact is not None):
            raise ValueError("alpha action requires exactly one alpha artifact")
        if (self.action.n_factors > 0) != (self.factor_artifact is not None):
            raise ValueError("factor actions require exactly one factor artifact")
        if not isinstance(self.export_onnx, bool) or not isinstance(
            self.export_torchscript, bool
        ):
            raise ValueError("export flags must be booleans")
        if not math.isfinite(self.export_tolerance) or self.export_tolerance <= 0.0:
            raise ValueError("export_tolerance must be finite and positive")
        if self.training.sequence_encoder and (
            self.export_onnx or self.export_torchscript
        ):
            raise ValueError(
                "structured sequence policies do not support flat ONNX/TorchScript export"
            )
        if self.git_commit is not None and not self.git_commit:
            raise ValueError("git_commit must be non-empty when provided")
        if self.git_dirty is not None and not isinstance(self.git_dirty, bool):
            raise ValueError("git_dirty must be a boolean or null")
        if self.schema_version != "training_run_config_v1":
            raise ValueError("unsupported training run configuration schema")

    @classmethod
    def from_mapping(cls, raw: object) -> TrainingRunConfig:
        payload = _mapping(raw, field="training run config")
        training_data = _tuple_fields(
            _mapping(payload.get("training"), field="training"),
            "seeds",
            "policy_net_arch",
            "value_net_arch",
        )
        reward = RewardConfig(**_mapping(payload.get("reward"), field="reward"))
        execution = ExecutionCostConfig(
            **_mapping(payload.get("execution"), field="execution")
        )
        environment_data = _tuple_fields(
            _mapping(payload.get("environment"), field="environment"),
            "episode_hour_choices",
            "initial_state_modes",
            "sequence_windows",
        )
        environment_data.pop("reward_config", None)
        environment_data.pop("execution_cost", None)
        emergency_risk = EmergencyRiskConfig(
            **_mapping(
                environment_data.pop("emergency_risk", {}),
                field="emergency_risk",
            )
        )
        exports = _mapping(payload.get("exports"), field="exports")
        git_commit = payload.get("git_commit")
        if git_commit is not None and not isinstance(git_commit, str):
            raise ValueError("git_commit must be a string or null")
        git_dirty = payload.get("git_dirty")
        if git_dirty is not None and not isinstance(git_dirty, bool):
            raise ValueError("git_dirty must be a boolean or null")
        schema_version = payload.get("schema_version", "training_run_config_v1")
        if not isinstance(schema_version, str):
            raise ValueError("schema_version must be a string")
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
        if raw_alpha_artifact is not None and not isinstance(raw_alpha_artifact, str):
            raise ValueError("alpha_artifact must be a path string or null")
        if raw_factor_artifact is not None and not isinstance(raw_factor_artifact, str):
            raise ValueError("factor_artifact must be a path string or null")
        return cls(
            training=ResidualTrainingConfig(**training_data),
            environment=ResidualMarketEnvConfig(
                **environment_data,
                reward_config=reward,
                emergency_risk=emergency_risk,
                execution_cost=execution,
            ),
            risk=PreTradeRiskConfig(**_mapping(payload.get("risk"), field="risk")),
            reward=reward,
            portfolio_risk=PortfolioRiskConfig(
                **_mapping(payload.get("portfolio_risk"), field="portfolio_risk")
            ),
            trend=TrendConfig(**_mapping(payload.get("trend"), field="trend")),
            action=ActionSpec(**_mapping(payload.get("action"), field="action")),
            alpha_contract=AlphaContract(
                **_mapping(payload.get("alpha_contract"), field="alpha_contract")
            ),
            alpha_artifact=(
                None if raw_alpha_artifact is None else Path(raw_alpha_artifact)
            ),
            factor_artifact=(
                None if raw_factor_artifact is None else Path(raw_factor_artifact)
            ),
            resume_checkpoints=tuple(sorted(resume_checkpoints)),
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
            git_dirty=git_dirty,
        )

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
        )

    @classmethod
    def from_json(cls, path: Path) -> TrainingRunConfig:
        config = cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        return config.resolve_artifact_paths(path.parent)

    def _identity_payload(
        self, *, resume_checkpoint_digests: dict[str, str]
    ) -> dict[str, object]:
        return {
            "action": asdict(self.action),
            "alpha_contract": asdict(self.alpha_contract),
            "alpha_artifact_digest": _signal_artifact_digest(
                self.alpha_artifact, kind="alpha"
            ),
            "environment": asdict(self.environment),
            "factor_artifact_digest": _signal_artifact_digest(
                self.factor_artifact, kind="factor"
            ),
            "export_onnx": self.export_onnx,
            "export_tolerance": self.export_tolerance,
            "export_torchscript": self.export_torchscript,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "portfolio_risk": asdict(self.portfolio_risk),
            "risk": asdict(self.risk),
            "reward": asdict(self.reward),
            "resume_checkpoint_digests": resume_checkpoint_digests,
            "schema_version": self.schema_version,
            "training": self.training.digest_payload(),
            "trend": asdict(self.trend),
        }

    def candidate_digest_payload(self) -> dict[str, object]:
        """Return the stable learning recipe identity, excluding resume transport."""

        return self._identity_payload(resume_checkpoint_digests={})

    def digest_payload(self) -> dict[str, object]:
        return self._identity_payload(
            resume_checkpoint_digests={
                str(seed): load_checkpoint_manifest(
                    path / "checkpoint.json" if path.is_dir() else path
                ).digest
                for seed, path in self.resume_checkpoints
            }
        )


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    run_id: str
    status: str
    path: Path
    run_digest: str
    policy_digest: str
    dataset_id: str
    run_kind: str = "research_exploratory"
    selection_authorization_digest: str | None = None
    selection_proposal_digest: str | None = None
    production_status: str = "NO-GO"


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def _dataset_manifest(
    dataset: MarketDataset, *, created_at: datetime
) -> DatasetManifest:
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
    *,
    normalizer: ObservationNormalizer | None = None,
    sequence_normalizer: SequenceFeatureNormalizer | None = None,
) -> Callable[[], ResidualMarketEnv]:
    alpha_provider = (
        None
        if config.alpha_artifact is None
        else load_alpha_artifact(
            config.alpha_artifact,
            dataset_id=dataset.dataset_id,
            expected_symbols=dataset.symbols,
        )
    )
    factor_provider = (
        None
        if config.factor_artifact is None
        else load_factor_artifact(
            config.factor_artifact,
            dataset_id=dataset.dataset_id,
            expected_names=tuple(
                f"factor_{index}" for index in range(config.action.n_factors)
            ),
            expected_symbols=dataset.n_symbols,
        )
    )

    def create() -> ResidualMarketEnv:
        return ResidualMarketEnv(
            dataset,
            trend_strategy=TrendStrategy(config.trend),
            alpha_provider=alpha_provider,
            alpha_enabled=config.action.alpha_enabled,
            alpha_artifact_digest=(
                None if alpha_provider is None else alpha_provider.artifact_digest
            ),
            alpha_contract=config.alpha_contract,
            factor_basis_provider=factor_provider,
            factor_artifact_digest=(
                None if factor_provider is None else factor_provider.artifact_digest
            ),
            factor_count=config.action.n_factors,
            action_spec=config.action,
            pre_trade_risk=PreTradeRisk(config.risk),
            portfolio_risk=PortfolioRiskModel(config.portfolio_risk),
            normalizer=normalizer,
            sequence_normalizer=sequence_normalizer,
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


def _feature_alignment_payload(
    feature_names: tuple[str, ...],
) -> dict[str, str]:
    return {
        name: "unshifted_decision_time"
        for name in feature_names
        if "__ichimoku_" in name or name.startswith("ichimoku_")
    }


def _policy_loader_payload(
    ensemble: PolicyEnsembleManifest,
    *,
    algorithm: str,
    structured_sequence: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "algorithm": algorithm,
        "members": tuple(
            f"members/member-{index:03d}/policy.zip"
            for index in range(ensemble.expected_members)
        ),
        "schema_version": (
            "sb3_policy_loader_v2" if structured_sequence else "sb3_policy_loader_v1"
        ),
    }
    if structured_sequence:
        payload.update(
            {
                "dataset_reference": "dataset-reference.json",
                "environment": "environment.json",
                "normalizer": "normalizer.json",
                "observation_mode": "structured_sequence",
                "sequence_normalizer": "sequence-normalizer.json",
            }
        )
    return payload


def _serving_support_payload(config: TrainingRunConfig) -> dict[str, object]:
    if config.training.sequence_encoder:
        return {
            "loader_schema": "sb3_policy_loader_v2",
            "observation_mode": "structured_sequence",
            "runtime": "native_sb3_structured_sequence_v1",
            "schema_version": "serving_support_v2",
            "status": "supported",
        }
    return {
        "loader_schema": "sb3_policy_loader_v1",
        "observation_mode": "flat",
        "runtime": "flat_vector_v1",
        "schema_version": "serving_support_v2",
        "status": "supported",
    }


def _normalizer_payload(normalizer: ObservationNormalizer) -> dict[str, object]:
    """Return the canonical serving-compatible normalizer sidecar."""

    return {"digest": normalizer.digest, **normalizer.digest_payload()}


def _sequence_normalizer_payload(
    normalizer: SequenceFeatureNormalizer,
) -> dict[str, object]:
    sample_count = normalizer.sample_count
    if sample_count is None:
        raise RuntimeError("sequence normalizer sample counts are unavailable")
    return {
        "center": {
            key: tuple(float(value) for value in normalizer.center[key])
            for key in normalizer.feature_names
        },
        "clip": normalizer.clip,
        "dataset_id": normalizer.dataset_id,
        "digest": normalizer.digest,
        "feature_names": dict(normalizer.feature_names),
        "scale": {
            key: tuple(float(value) for value in normalizer.scale[key])
            for key in normalizer.feature_names
        },
        "sample_count": {
            key: tuple(int(value) for value in sample_count[key])
            for key in normalizer.feature_names
        },
        "minimum_samples_per_channel": normalizer.minimum_samples_per_channel,
        "schema_version": normalizer.schema_version,
        "sequence_schema_digest": normalizer.sequence_schema_digest,
        "source_dataset_id": normalizer.source_dataset_id,
        "train_range": [normalizer.train_start, normalizer.train_end],
    }


def _fit_full_normalizers(
    dataset: MarketDataset,
    config: TrainingRunConfig,
) -> tuple[ObservationNormalizer, SequenceFeatureNormalizer | None]:
    flat_config = (
        replace(
            config,
            environment=replace(
                config.environment,
                structured_sequence_observation=False,
                sequence_windows=(),
            ),
        )
        if config.environment.structured_sequence_observation
        else config
    )
    env = _environment_factory(dataset, flat_config)()
    observations: list[np.ndarray] = []
    try:
        start = env.minimum_start_index
        episode_bars = dataset.n_bars - 1 - start
        if episode_bars <= 0:
            raise ValueError("dataset is too short to fit the full-run normalizer")
        observation, _ = env.reset(
            seed=0,
            options={
                "episode_bars": episode_bars,
                "initial_state_mode": "cash",
                "start_idx": start,
            },
        )
        terminated = False
        truncated = False
        while not terminated and not truncated:
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            observation, _, terminated, truncated, _ = env.step(
                np.zeros(config.action.size, dtype=np.float32)
            )
        matrix = np.stack(observations, axis=0)
        passthrough = observation_passthrough_indices(
            dataset,
            action_size=config.action.size,
            n_factors=config.action.n_factors,
            finite_horizon=config.environment.finite_horizon_observation,
        )
        normalizer = ObservationNormalizer.fit(
            matrix,
            train_start=0,
            train_end=matrix.shape[0],
            passthrough_indices=passthrough,
            dataset_id=dataset.dataset_id,
            source_dataset_id=dataset.dataset_id,
            absolute_train_start=start,
            absolute_train_end=dataset.n_bars,
            observation_schema_digest=env.observation_builder.schema_digest(dataset),
            action_spec_digest=env.action_spec_digest,
            alpha_artifact_digest=env.alpha_artifact_digest,
            factor_artifact_digest=env.factor_artifact_digest,
            candidate_config_digest=content_digest(config.candidate_digest_payload()),
        )
    finally:
        env.close()

    if not config.environment.structured_sequence_observation:
        return normalizer, None
    builder = SequenceObservationBuilder(
        windows=tuple(
            SequenceWindowSpec(timeframe, length)
            for timeframe, length in config.environment.resolved_sequence_windows
        )
    )
    sequence_normalizer = SequenceFeatureNormalizer.fit(
        dataset,
        builder,
        train_start=max(start, builder.minimum_index(dataset)),
        train_end=dataset.n_bars,
        source_dataset_id=dataset.dataset_id,
    )
    return normalizer, sequence_normalizer


def _dataset_artifact_digest(root: Path) -> str:
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = payload.get("artifact_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("dataset artifact digest is missing or invalid")
    return digest


def normalize_training_run_config(config: TrainingRunConfig) -> TrainingRunConfig:
    """Bind full training to the same liquidation-at-close terminal contract as OOS."""

    if config.environment.liquidate_on_end:
        return config
    return replace(
        config,
        environment=replace(config.environment, liquidate_on_end=True),
    )


def _lockfile_digest() -> str:
    path = Path(__file__).resolve().parents[2] / "uv.lock"
    if not path.is_file():
        raise ValueError("selected final training requires uv.lock")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_authorization(
    *,
    proposal_path: Path | None,
    authorization_path: Path | None,
    public_keys_path: Path | None,
    required: bool,
    dataset: MarketDataset,
    config: TrainingRunConfig,
    trusted_at: datetime,
) -> tuple[SelectionProposal | None, SelectionAuthorization | None]:
    supplied = (proposal_path, authorization_path, public_keys_path)
    if all(value is None for value in supplied):
        if required:
            raise ValueError(
                "selected final training requires proposal, authorization, and public keys"
            )
        return None, None
    if any(value is None for value in supplied):
        raise ValueError(
            "selection proposal, authorization, and public keys must be supplied together"
        )
    assert proposal_path is not None
    assert authorization_path is not None
    assert public_keys_path is not None
    if config.resume_checkpoints:
        raise ValueError("selected final training forbids resume checkpoints")
    proposal = load_selection_proposal(proposal_path)
    authorization = load_selection_authorization(authorization_path)
    trusted_keys: dict[str, PublicVerificationKey] = load_public_verification_keys(
        public_keys_path
    )
    authorization.verify(
        proposal,
        trusted_keys=trusted_keys,
        trusted_at=trusted_at,
    )
    if proposal.dataset_id != dataset.dataset_id:
        raise ValueError("selection proposal dataset identity mismatch")
    if proposal.candidate_config_digest != content_digest(
        config.candidate_digest_payload()
    ):
        raise ValueError("selection proposal candidate identity mismatch")
    if proposal.seeds != config.training.seeds:
        raise ValueError("selection proposal seed set mismatch")
    if proposal.resume_checkpoint_digests:
        raise ValueError(
            "selected final proposal must not authorize resume checkpoints"
        )
    if config.git_commit is None or proposal.git_commit != config.git_commit:
        raise ValueError("selection proposal git commit mismatch")
    if proposal.dependency_digest != _lockfile_digest():
        raise ValueError("selection proposal dependency digest mismatch")
    return proposal, authorization


def execute_training_run(
    *,
    config_path: Path,
    dataset_path: Path,
    store_root: Path,
    run_id: str | None = None,
    created_at: datetime | None = None,
    selection_proposal_path: Path | None = None,
    selection_authorization_path: Path | None = None,
    selection_public_keys_path: Path | None = None,
    require_selection_authorization: bool = False,
) -> TrainingRunResult:
    """Train, serialize, validate, and atomically publish one ensemble run."""

    resolved_created_at = created_at or datetime.now(UTC)
    resolved_run_id = run_id or resolved_created_at.strftime("run-%Y%m%dT%H%M%SZ")
    config = normalize_training_run_config(TrainingRunConfig.from_json(config_path))
    dataset = load_market_dataset_artifact(dataset_path)
    proposal, authorization = _selection_authorization(
        proposal_path=selection_proposal_path,
        authorization_path=selection_authorization_path,
        public_keys_path=selection_public_keys_path,
        required=require_selection_authorization,
        dataset=dataset,
        config=config,
        trusted_at=resolved_created_at,
    )
    run_kind = (
        "research_selected_final"
        if authorization is not None
        else "research_exploratory"
    )
    metadata_promotion = metadata_promotion_from_dataset(dataset)
    if run_kind == "research_selected_final":
        metadata_promotion.require_promotable()
    normalizer, sequence_normalizer = _fit_full_normalizers(dataset, config)
    store = ArtifactStore(store_root)
    stage = store.stage_run(resolved_run_id)
    try:
        training_config_digest = content_digest(config.digest_payload())
        provenance = capture_runtime_provenance(
            Path(__file__).resolve().parents[2],
            git_commit=config.git_commit,
            git_dirty=config.git_dirty,
            deterministic_seed_config={
                "seeds": config.training.seeds,
                "training_config_digest": training_config_digest,
            },
        )
        _write_json(stage / "provenance.json", asdict(provenance))
        write_metadata_promotion_evidence(
            stage / METADATA_PROMOTION_FILE_NAME,
            metadata_promotion,
        )
        dataset_artifact_digest = _dataset_artifact_digest(dataset_path)
        _write_json(stage / "training-config.json", config.digest_payload())
        _write_json(
            stage / "training-purpose.json",
            {
                "run_kind": run_kind,
                "schema_version": "training_purpose_v1",
                "selection_authorization_digest": (
                    None
                    if authorization is None
                    else authorization.authorization_digest
                ),
                "selection_proposal_digest": (
                    None if proposal is None else proposal.digest
                ),
            },
        )
        if proposal is not None and authorization is not None:
            _write_json(stage / "selection-proposal.json", proposal.to_mapping())
            _write_json(
                stage / "selection-authorization.json",
                authorization.to_mapping(),
            )
        _write_json(
            stage / "dataset-reference.json",
            {
                "artifact_digest": dataset_artifact_digest,
                "bar_hours": dataset.bar_hours,
                "dataset_id": dataset.dataset_id,
                "feature_config_digest": dataset.feature_config_digest,
                "feature_names": dataset.feature_names,
                "feature_alignments": _feature_alignment_payload(dataset.feature_names),
                "global_feature_names": dataset.global_feature_names,
                "schema_version": "dataset_reference_v4",
                "symbols": dataset.symbols,
            },
        )
        _write_json(stage / "normalizer.json", _normalizer_payload(normalizer))
        if sequence_normalizer is not None:
            _write_json(
                stage / "sequence-normalizer.json",
                _sequence_normalizer_payload(sequence_normalizer),
            )
        _write_json(stage / "serving-support.json", _serving_support_payload(config))
        _write_json(
            stage / "environment.json",
            {
                "action": asdict(config.action),
                "alpha_contract": asdict(config.alpha_contract),
                "alpha_artifact_digest": _signal_artifact_digest(
                    config.alpha_artifact, kind="alpha"
                ),
                "environment": asdict(config.environment),
                "factor_artifact_digest": _signal_artifact_digest(
                    config.factor_artifact, kind="factor"
                ),
                "risk": asdict(config.risk),
                "reward": asdict(config.reward),
                "schema_version": "training_environment_v2",
                "terminal_accounting_mode": config.environment.terminal_accounting_mode,
                "trend": asdict(config.trend),
            },
        )
        ensemble = train_residual_ensemble(
            dataset=_dataset_manifest(dataset, created_at=resolved_created_at),
            environment_dataset_id=dataset.dataset_id,
            config=config.training,
            backend=StableBaselines3Backend(
                _environment_factory(
                    dataset,
                    config,
                    normalizer=normalizer,
                    sequence_normalizer=sequence_normalizer,
                ),
                resume_checkpoint_artifacts=dict(config.resume_checkpoints),
            ),
            output_dir=stage / "members",
            created_at=resolved_created_at,
        )
        _write_json(stage / "ensemble.json", _ensemble_payload(ensemble))
        _write_json(
            stage / "policy-loader.json",
            _policy_loader_payload(
                ensemble,
                algorithm=config.training.algorithm,
                structured_sequence=config.training.sequence_encoder,
            ),
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
            provenance_digest=provenance.digest,
            artifact_paths=_artifact_paths(stage),
            created_at=resolved_created_at,
            completed_at=datetime.now(UTC),
            run_kind=run_kind,
            selection_proposal_digest=(None if proposal is None else proposal.digest),
            selection_authorization_digest=(
                None if authorization is None else authorization.authorization_digest
            ),
            walk_forward_run_digest=(
                None if proposal is None else proposal.walk_forward_run_digest
            ),
            gate_evidence_digest=(
                None if proposal is None else proposal.gate_evidence_digest
            ),
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
            run_kind=run_kind,
            selection_authorization_digest=(
                None if authorization is None else authorization.authorization_digest
            ),
            selection_proposal_digest=(None if proposal is None else proposal.digest),
        )
    except Exception:
        if stage.is_dir():
            store.mark_failed(resolved_run_id)
        raise
