"""Concrete real-market nested walk-forward training and evaluation adapters."""

from __future__ import annotations

import math
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.provenance import capture_runtime_provenance
from trade_rl.artifacts.run_manifest import (
    WalkForwardRunManifest,
    validate_walk_forward_run_directory,
    write_walk_forward_run_manifest,
)
from trade_rl.artifacts.store import ArtifactStore
from trade_rl.catalog.contracts import ArtifactKind
from trade_rl.catalog.postgres import PostgresArtifactCatalog
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
from trade_rl.catalog.sealed_test import PostgresSealedTestLedger
from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.market import MarketDataset
from trade_rl.domain.checkpoints import PolicyCheckpoint, PolicyCheckpointLoader
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
from trade_rl.integrations.checkpoints import StableBaselines3CheckpointLoader
from trade_rl.integrations.sb3_ensemble import predict_deterministic_mean_action
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.risk.portfolio import PortfolioRiskModel
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.checkpointing import checkpoint_manifests
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import observation_passthrough_indices
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequenceWindowSpec,
)
from trade_rl.rl.training import train_residual_ensemble
from trade_rl.serving.normalizer import (
    load_observation_normalizer,
    write_observation_normalizer,
)
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionRuleStress
from trade_rl.strategies.trend import TrendStrategy
from trade_rl.workflows.fold_runner import (
    BASELINE_CONFIGURATION,
    CandidateConfiguration,
    CandidateEvaluation,
    CandidateEvaluationRequest,
    CandidateTrainer,
    CandidateTrainingRequest,
    CheckpointPolicyEvaluation,
    ConcreteFoldRunner,
    EvaluationPhase,
    FoldExecutionConfig,
    PolicyTrainingArtifact,
    select_seed_checkpoint_finalists,
)
from trade_rl.workflows.market_walk_forward_config import (
    ExecutionSensitivityConfig,
)
from trade_rl.workflows.market_walk_forward_config import (
    MarketWalkForwardConfig as MarketWalkForwardConfig,
)
from trade_rl.workflows.market_walk_forward_config import (
    NamedCandidateRun as NamedCandidateRun,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    require_mark_to_market_training_environment,
)
from trade_rl.workflows.walk_forward import (
    WalkForwardExecutionResult,
    execute_walk_forward,
)
from trade_rl.workflows.walk_forward_evaluation import (
    RangeEvaluation,
    bind_signal_providers_to_view,
    build_market_environment,
    evaluate_range_evidence,
    minimum_environment_start,
    resolve_signal_digest,
)

_NORMALIZER_ENVIRONMENT_FACTORY: Any | None = None


def _collect_normalizer_chunk(bounds: tuple[int, int, int]) -> np.ndarray:
    """Collect one cash/zero-action interval in an isolated environment."""

    start, stop, action_size = bounds
    factory = _NORMALIZER_ENVIRONMENT_FACTORY
    if factory is None:
        raise RuntimeError("normalizer worker environment factory is unavailable")
    environment = factory()
    observations: list[np.ndarray] = []
    try:
        observation, _ = environment.reset(
            seed=0,
            options={
                "episode_bars": stop - start,
                "initial_state_mode": "cash",
                "start_idx": start,
            },
        )
        terminated = False
        truncated = False
        while not terminated and not truncated:
            observations.append(
                np.asarray(observation, dtype=np.float32).copy(order="C")
            )
            observation, _, terminated, truncated, _ = environment.step(
                np.zeros(action_size, dtype=np.float32)
            )
    finally:
        environment.close()
    if len(observations) != stop - start:
        raise RuntimeError("normalizer worker observation count mismatch")
    return np.stack(observations, axis=0)


def _normalizer_worker_count(episode_bars: int) -> int:
    raw = os.environ.get("TRADE_RL_PREPROCESS_WORKERS", "8").strip()
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ValueError("TRADE_RL_PREPROCESS_WORKERS must be an integer") from exc
    if configured <= 0:
        raise ValueError("TRADE_RL_PREPROCESS_WORKERS must be positive")
    return min(configured, episode_bars)


def _normalizer_partitions(
    start: int,
    stop: int,
    worker_count: int,
    action_size: int,
) -> tuple[tuple[int, int, int], ...]:
    boundaries = np.linspace(start, stop, worker_count + 1, dtype=np.int64)
    return tuple(
        (int(left), int(right), action_size)
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        if right > left
    )


def _collect_normalizer_matrix(
    environment_factory: Any,
    *,
    start: int,
    episode_bars: int,
    action_size: int,
    finite_horizon: bool,
) -> np.ndarray:
    global _NORMALIZER_ENVIRONMENT_FACTORY

    worker_count = _normalizer_worker_count(episode_bars)
    parallel = (
        worker_count > 1
        and "fork" in multiprocessing.get_all_start_methods()
        and not finite_horizon
    )
    _NORMALIZER_ENVIRONMENT_FACTORY = environment_factory
    try:
        if not parallel:
            return _collect_normalizer_chunk((start, start + episode_bars, action_size))
        partitions = _normalizer_partitions(
            start,
            start + episode_bars,
            worker_count,
            action_size,
        )
        with ProcessPoolExecutor(
            max_workers=len(partitions),
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            chunks = tuple(executor.map(_collect_normalizer_chunk, partitions))
        return np.concatenate(chunks, axis=0)
    finally:
        _NORMALIZER_ENVIRONMENT_FACTORY = None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class WalkForwardRunResult:
    run_id: str
    status: str
    path: Path
    run_digest: str
    evaluation_digest: str
    dataset_id: str
    production_status: str = "NO-GO"


@dataclass(frozen=True, slots=True)
class _PolicyRecord:
    path: Path
    algorithm: str
    normalizer: ObservationNormalizer
    sequence_normalizer: SequenceFeatureNormalizer | None
    run: TrainingRunConfig
    members: tuple[tuple[str, Path], ...] = ()


class _DeterministicMeanPolicy:
    """Evaluate the exact deterministic mean ensemble used by serving."""

    def __init__(self, models: tuple[Any, ...]) -> None:
        if not models:
            raise ValueError("deployable ensemble requires at least one model")
        self.models = models

    def predict(
        self, observation: object, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        if deterministic is not True:
            raise ValueError("deployable ensemble evaluation must be deterministic")
        action = predict_deterministic_mean_action(
            self.models,
            observation,
            context="deployable ensemble",
        )
        return action, None


def _experiment_plan_digest(
    config: MarketWalkForwardConfig,
    *,
    dataset_id: str,
) -> str:
    """Bind sealed access to the full workflow, candidates, and selector."""

    return content_digest(
        {
            "dataset_id": dataset_id,
            "market_walk_forward_config": config.digest_payload(),
            "schema_version": "market_walk_forward_experiment_plan_v1",
            "selection_protocol": (
                "checkpoint_top_k_per_seed_then_configuration_selection_v1"
            ),
        }
    )


def _sequence_history_bars(
    dataset: MarketDataset,
    run: TrainingRunConfig,
) -> int:
    if not run.environment.structured_sequence_observation:
        return 0
    builder = SequenceObservationBuilder(
        windows=tuple(
            SequenceWindowSpec(timeframe, length)
            for timeframe, length in run.environment.resolved_sequence_windows
        )
    )
    return builder.minimum_index(dataset)


def _training_view_bounds(
    dataset: MarketDataset,
    train_range: IndexRange,
    run: TrainingRunConfig,
) -> tuple[int, int]:
    signal_minimum = TrendStrategy(run.trend).minimum_history_for(dataset)
    reward_minimum = signal_minimum
    if run.reward.baseline_underperformance_weight > 0.0:
        from trade_rl.rl.episode import minimum_reward_start_index

        reward_minimum = minimum_reward_start_index(
            dataset,
            signal_minimum=signal_minimum,
            window_hours=run.reward.baseline_window_hours,
        )
    required_history = max(
        reward_minimum,
        _sequence_history_bars(dataset, run),
    )
    if train_range.start == 0:
        # The first fold cannot borrow history from before the sealed dataset.
        # Its causal warm-up is excluded from fitted observations and recorded
        # as the effective absolute training start below.
        return 0, train_range.stop
    if train_range.start < required_history:
        raise ValueError(
            "training range lacks causal signal, reward, or sequence history"
        )
    return train_range.start - required_history, train_range.stop


def _training_view(
    dataset: MarketDataset,
    train_range: IndexRange,
    run: TrainingRunConfig,
) -> MarketDataset:
    start, stop = _training_view_bounds(dataset, train_range, run)
    return MarketDatasetView(dataset, start, stop).materialize()


def _fit_normalizer(
    dataset: MarketDataset,
    train_range: IndexRange,
    run: TrainingRunConfig,
) -> ObservationNormalizer:
    training_dataset = _training_view(dataset, train_range, run)
    view_start, view_stop = _training_view_bounds(dataset, train_range, run)
    alpha_provider, factor_provider = bind_signal_providers_to_view(
        dataset,
        training_dataset,
        run,
        start=view_start,
        stop=view_stop,
        evaluation_start=train_range.start,
    )
    start = minimum_environment_start(
        training_dataset,
        run,
        alpha_provider=alpha_provider,
        factor_provider=factor_provider,
    )
    episode_bars = training_dataset.n_bars - 1 - start
    if episode_bars <= 0:
        raise ValueError("training range is too short to fit an observation normalizer")
    normalizer_run = (
        replace(
            run,
            environment=replace(
                run.environment,
                structured_sequence_observation=False,
                sequence_windows=(),
            ),
        )
        if run.environment.structured_sequence_observation
        else run
    )

    def environment_factory() -> Any:
        return build_market_environment(
            training_dataset,
            normalizer_run,
            normalizer=None,
            sequence_normalizer=None,
            episode_bars=episode_bars,
            liquidate_on_end=False,
            alpha_provider=alpha_provider,
            factor_provider=factor_provider,
        )

    env = environment_factory()
    try:
        matrix = _collect_normalizer_matrix(
            environment_factory,
            start=start,
            episode_bars=episode_bars,
            action_size=run.action.size,
            finite_horizon=normalizer_run.environment.finite_horizon_observation,
        )
        passthrough = observation_passthrough_indices(
            training_dataset,
            action_size=run.action.size,
            n_factors=run.action.n_factors,
            finite_horizon=run.environment.finite_horizon_observation,
        )
        return ObservationNormalizer.fit(
            matrix,
            train_start=0,
            train_end=matrix.shape[0],
            passthrough_indices=passthrough,
            dataset_id=training_dataset.dataset_id,
            source_dataset_id=dataset.dataset_id,
            absolute_train_start=view_start + start,
            absolute_train_end=train_range.stop,
            observation_schema_digest=env.observation_builder.schema_digest(
                training_dataset
            ),
            action_spec_digest=env.action_spec_digest,
            alpha_artifact_digest=(
                None if alpha_provider is None else alpha_provider.artifact_digest
            ),
            factor_artifact_digest=(
                None if factor_provider is None else factor_provider.artifact_digest
            ),
            candidate_config_digest=content_digest(run.digest_payload()),
        )
    finally:
        env.close()


def _fit_sequence_normalizer(
    dataset: MarketDataset,
    train_range: IndexRange,
    run: TrainingRunConfig,
) -> SequenceFeatureNormalizer | None:
    if not run.environment.structured_sequence_observation:
        return None
    training_dataset = _training_view(dataset, train_range, run)
    view_start, _ = _training_view_bounds(dataset, train_range, run)
    builder = SequenceObservationBuilder(
        windows=tuple(
            SequenceWindowSpec(timeframe, length)
            for timeframe, length in run.environment.resolved_sequence_windows
        )
    )
    effective_start = max(
        builder.minimum_index(training_dataset),
        train_range.start - view_start,
    )
    effective_end = train_range.stop - view_start
    return SequenceFeatureNormalizer.fit(
        training_dataset,
        builder,
        train_start=effective_start,
        train_end=effective_end,
        source_dataset_id=dataset.dataset_id,
    )


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


def _normalizer_payload(
    normalizer: ObservationNormalizer,
    *,
    dataset_id: str,
    train_range: IndexRange,
) -> dict[str, object]:
    return {
        "absolute_train_range": [
            normalizer.absolute_train_start,
            normalizer.absolute_train_end,
        ],
        "clip": normalizer.clip,
        "dataset_id": dataset_id,
        "digest": normalizer.digest,
        "epsilon": normalizer.epsilon,
        "mean": tuple(float(value) for value in normalizer.mean),
        "observation_schema": normalizer.observation_schema,
        "passthrough_indices": normalizer.passthrough_indices,
        "scale": tuple(float(value) for value in normalizer.scale),
        "schema_version": normalizer.schema_version,
    }


def _maintained_training_environment(
    config: ResidualMarketEnvConfig,
    *,
    episode_bars: int,
) -> ResidualMarketEnvConfig:
    """Bind fold-local bounds without changing training terminal semantics."""

    maintained = require_mark_to_market_training_environment(config)
    return replace(
        maintained,
        episode_bars=episode_bars,
        episode_hour_choices=(),
        fail_on_incomplete_emergency_liquidation=False,
        require_full_reward_preroll=True,
    )


class MarketCandidateTrainer(CandidateTrainer):
    """Train only on one fold's train view and select a seed on checkpoint data."""

    def __init__(
        self,
        *,
        dataset: MarketDataset,
        candidates: dict[str, TrainingRunConfig],
        root: Path,
        created_at: datetime,
        registry: dict[str, _PolicyRecord],
        checkpoint_finalists_per_seed: int = 1,
        checkpoint_loader: PolicyCheckpointLoader | None = None,
    ) -> None:
        self.dataset = dataset
        self.candidates = candidates
        self.root = root
        self.created_at = created_at
        self.registry = registry
        self.checkpoint_finalists_per_seed = checkpoint_finalists_per_seed
        self.checkpoint_loader = checkpoint_loader or StableBaselines3CheckpointLoader()
        raw_normalizer_cache = os.environ.get(
            "TRADE_RL_NORMALIZER_CACHE_ROOT", ""
        ).strip()
        self.normalizer_cache_root = (
            None if not raw_normalizer_cache else Path(raw_normalizer_cache).resolve()
        )
        self.normalizer_artifact_index = (
            None
            if self.normalizer_cache_root is None
            else ReusableArtifactIndex.from_environment(
                storage_root=self.normalizer_cache_root
            )
        )
        self._normalizers: dict[tuple[int, str], ObservationNormalizer] = {}
        self._sequence_normalizers: dict[
            tuple[int, str], SequenceFeatureNormalizer | None
        ] = {}

    def _normalizer(
        self,
        request: CandidateTrainingRequest,
        run: TrainingRunConfig,
    ) -> ObservationNormalizer:
        key = (request.fold_index, request.configuration.name)
        existing = self._normalizers.get(key)
        if existing is not None:
            return existing
        cache_identity = {
            "candidate_config_digest": content_digest(run.digest_payload()),
            "dataset_id": self.dataset.dataset_id,
            "schema_version": "walk_forward_normalizer_cache_identity_v1",
            "train_range": [request.train.start, request.train.stop],
        }
        cache_path: Path | None = None
        normalizer: ObservationNormalizer | None = None
        if self.normalizer_cache_root is not None:
            cache_path = self.normalizer_cache_root / content_digest(cache_identity)
            if self.normalizer_artifact_index is not None:
                indexed = self.normalizer_artifact_index.resolve(
                    ArtifactKind.NORMALIZER,
                    cache_identity,
                )
                if indexed is not None:
                    cache_path = indexed
            if cache_path.is_dir():
                normalizer = load_observation_normalizer(cache_path)
                if normalizer.source_dataset_id != self.dataset.dataset_id:
                    raise ValueError("cached normalizer dataset identity mismatch")
                if normalizer.absolute_train_end != request.train.stop:
                    raise ValueError("cached normalizer train range mismatch")
        if normalizer is None:
            normalizer = _fit_normalizer(self.dataset, request.train, run)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(
                        prefix=f".{cache_path.name}.", dir=str(cache_path.parent)
                    )
                )
                try:
                    write_observation_normalizer(temporary, normalizer)
                    try:
                        temporary.replace(cache_path)
                    except FileExistsError:
                        pass
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
        if cache_path is not None and self.normalizer_artifact_index is not None:
            self.normalizer_artifact_index.register_directory(
                artifact_digest=normalizer.digest,
                artifact_kind=ArtifactKind.NORMALIZER,
                schema_version=normalizer.schema_version,
                dataset_id=self.dataset.dataset_id,
                cache_key=cache_identity,
                metadata={
                    "feature_count": normalizer.size,
                    "train_start": request.train.start,
                    "train_stop": request.train.stop,
                },
                location=cache_path,
            )
        self._normalizers[key] = normalizer
        _write_json(
            self.root / f"fold-{request.fold_index:03d}" / "normalizer.json",
            _normalizer_payload(
                normalizer,
                dataset_id=self.dataset.dataset_id,
                train_range=request.train,
            ),
        )
        return normalizer

    def _sequence_normalizer(
        self,
        request: CandidateTrainingRequest,
        run: TrainingRunConfig,
    ) -> SequenceFeatureNormalizer | None:
        key = (request.fold_index, request.configuration.name)
        if key in self._sequence_normalizers:
            return self._sequence_normalizers[key]
        normalizer = _fit_sequence_normalizer(self.dataset, request.train, run)
        self._sequence_normalizers[key] = normalizer
        if normalizer is not None:
            _write_json(
                self.root
                / f"fold-{request.fold_index:03d}"
                / f"sequence-normalizer-{request.configuration.name}.json",
                _sequence_normalizer_payload(normalizer),
            )
        return normalizer

    def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact:
        if request.dataset_id != self.dataset.dataset_id:
            raise ValueError("candidate training dataset identity mismatch")
        run = self.candidates[request.configuration.name]
        normalizer = self._normalizer(request, run)
        sequence_normalizer = self._sequence_normalizer(request, run)
        training_dataset = _training_view(self.dataset, request.train, run)
        view_start, view_stop = _training_view_bounds(self.dataset, request.train, run)
        alpha_provider, factor_provider = bind_signal_providers_to_view(
            self.dataset,
            training_dataset,
            run,
            start=view_start,
            stop=view_stop,
            evaluation_start=request.train.start,
        )
        minimum_start = minimum_environment_start(
            training_dataset,
            run,
            alpha_provider=alpha_provider,
            factor_provider=factor_provider,
        )
        maximum_episode = training_dataset.n_bars - 1 - minimum_start
        if maximum_episode <= 0:
            raise ValueError("fold training range is too short")
        configured_episode = run.environment.resolve_nominal_episode_bars(
            training_dataset
        )
        episode_bars = min(configured_episode, maximum_episode)
        training_environment = _maintained_training_environment(
            run.environment,
            episode_bars=episode_bars,
        )

        def factory() -> ResidualMarketEnv:
            return ResidualMarketEnv(
                training_dataset,
                trend_strategy=TrendStrategy(run.trend),
                alpha_provider=alpha_provider,
                alpha_enabled=run.action.alpha_enabled,
                alpha_artifact_digest=(
                    None if alpha_provider is None else alpha_provider.artifact_digest
                ),
                alpha_contract=run.alpha_contract,
                factor_basis_provider=factor_provider,
                factor_artifact_digest=(
                    None if factor_provider is None else factor_provider.artifact_digest
                ),
                factor_count=run.action.n_factors,
                action_spec=run.action,
                pre_trade_risk=PreTradeRisk(run.risk),
                portfolio_risk=PortfolioRiskModel(run.portfolio_risk),
                normalizer=normalizer,
                sequence_normalizer=sequence_normalizer,
                config=training_environment,
            )

        candidate_root = (
            self.root
            / f"fold-{request.fold_index:03d}"
            / "candidates"
            / request.configuration.name
        )
        ensemble = train_residual_ensemble(
            dataset=DatasetManifest(
                dataset_id=training_dataset.dataset_id,
                symbols=training_dataset.symbols,
                feature_names=training_dataset.feature_names,
                base_timeframe=f"{training_dataset.bar_hours:g}h",
                bar_hours=training_dataset.bar_hours,
                created_at=self.created_at,
            ),
            environment_dataset_id=training_dataset.dataset_id,
            config=run.training,
            backend=StableBaselines3Backend(
                factory,
                structured_export_enabled=run.export_structured_torchscript,
                structured_export_tolerance=run.export_tolerance,
            ),
            output_dir=candidate_root / "members",
            created_at=self.created_at,
        )
        _write_json(candidate_root / "ensemble.json", asdict(ensemble))
        _write_json(candidate_root / "training-config.json", run.digest_payload())
        if run.export_structured_torchscript:
            from trade_rl.serving.policy_loader import (
                write_structured_policy_loader_manifest,
            )

            write_structured_policy_loader_manifest(
                candidate_root,
                expected_members=ensemble.expected_members,
            )

        scored: list[CheckpointPolicyEvaluation] = []
        for index, member in enumerate(ensemble.members):
            member_root = candidate_root / "members" / f"member-{index:03d}"
            candidates = [
                (member.seed, member.checkpoint_digest, member_root / "policy.zip"),
                *(
                    (checkpoint.seed, checkpoint.policy_digest, checkpoint.policy_path)
                    for checkpoint in checkpoint_manifests(member_root / "checkpoints")
                ),
            ]
            if any(seed != member.seed for seed, _, _ in candidates):
                raise ValueError("checkpoint seed does not match ensemble member seed")
            for seed, policy_digest, path in candidates:
                record = _PolicyRecord(
                    path=path,
                    algorithm=run.training.algorithm,
                    normalizer=normalizer,
                    sequence_normalizer=sequence_normalizer,
                    run=run,
                )
                self.registry[policy_digest] = record
                model = self.checkpoint_loader.load(
                    PolicyCheckpoint(path=record.path, algorithm=record.algorithm)
                )
                evidence = evaluate_range_evidence(
                    dataset=self.dataset,
                    evaluation_range=request.checkpoint_validation,
                    run=run,
                    normalizer=normalizer,
                    sequence_normalizer=sequence_normalizer,
                    model=model,
                    baseline=False,
                )
                score = sum(math.log1p(value) for value in evidence.returns.values)
                evaluation_digest = content_digest(
                    {
                        "dataset_id": request.dataset_id,
                        "diagnostics": evidence.diagnostics.digest_payload(),
                        "fold_index": request.fold_index,
                        "phase": "checkpoint_validation",
                        "policy_digest": policy_digest,
                        "range": (
                            request.checkpoint_validation.start,
                            request.checkpoint_validation.stop,
                        ),
                        "returns": evidence.returns.values,
                        "score": score,
                        "seed": seed,
                    }
                )
                scored.append(
                    CheckpointPolicyEvaluation(
                        seed=seed,
                        policy_digest=policy_digest,
                        score=score,
                        evaluation_digest=evaluation_digest,
                    )
                )
        finalists = select_seed_checkpoint_finalists(
            checkpoint_evaluations=tuple(scored),
            finalists_per_seed=self.checkpoint_finalists_per_seed,
        )
        _write_json(
            candidate_root / "checkpoint-selection.json",
            {
                "candidates": tuple(
                    {
                        "evaluation_digest": item.evaluation_digest,
                        "policy_digest": item.policy_digest,
                        "score": item.score,
                        "seed": item.seed,
                    }
                    for item in sorted(
                        scored,
                        key=lambda item: (item.seed, item.policy_digest),
                    )
                ),
                "checkpoint_range": [
                    request.checkpoint_validation.start,
                    request.checkpoint_validation.stop,
                ],
                "schema_version": "checkpoint_selection_v2_seed_aware",
                "seed_finalists": tuple(
                    {
                        "checkpoint_evaluation_digest": (
                            finalist.checkpoint_evaluation_digest
                        ),
                        "checkpoint_score": finalist.checkpoint_score,
                        "policy_digest": finalist.policy_digest,
                        "seed": finalist.seed,
                    }
                    for finalist in finalists
                ),
            },
        )
        artifact = PolicyTrainingArtifact(
            configuration=request.configuration.name,
            seed_finalists=finalists,
        )
        if artifact.ensemble_policy_digest not in self.registry:
            member_records = tuple(
                self.registry[item.policy_digest]
                for item in artifact.deployment_members
            )
            first = member_records[0]
            self.registry[artifact.ensemble_policy_digest] = _PolicyRecord(
                path=first.path,
                algorithm=first.algorithm,
                normalizer=first.normalizer,
                sequence_normalizer=first.sequence_normalizer,
                run=first.run,
                members=tuple(
                    (item.policy_digest, record.path)
                    for item, record in zip(
                        artifact.deployment_members,
                        member_records,
                        strict=True,
                    )
                ),
            )
        return artifact


class MarketCandidateEvaluator:
    """Evaluate only the exact range carried by a candidate evaluation request."""

    def __init__(
        self,
        *,
        dataset: MarketDataset,
        baseline_run: TrainingRunConfig,
        registry: dict[str, _PolicyRecord],
        checkpoint_loader: PolicyCheckpointLoader | None = None,
    ) -> None:
        self.dataset = dataset
        self.baseline_run = baseline_run
        self.registry = registry
        self.checkpoint_loader = checkpoint_loader or StableBaselines3CheckpointLoader()
        self._model_cache: dict[str, Any] = {}
        self.outer_test_counts: dict[int, int] = {}

    def _evaluation_inputs(
        self,
        *,
        configuration: str,
        policy_digest: str | None,
    ) -> tuple[
        TrainingRunConfig,
        ObservationNormalizer | None,
        SequenceFeatureNormalizer | None,
        Any | None,
        bool,
    ]:
        baseline = configuration == BASELINE_CONFIGURATION
        if baseline:
            return self.baseline_run, None, None, None, True
        if policy_digest is None:
            raise ValueError("residual evaluation requires a policy digest")
        record = self.registry.get(policy_digest)
        if record is None:
            raise ValueError("candidate evaluation policy is not registered")
        model = self._model_cache.get(policy_digest)
        if model is None:
            if record.members:
                models = tuple(
                    self.checkpoint_loader.load(
                        PolicyCheckpoint(path=path, algorithm=record.algorithm)
                    )
                    for _, path in record.members
                )
                model = _DeterministicMeanPolicy(models)
            else:
                model = self.checkpoint_loader.load(
                    PolicyCheckpoint(path=record.path, algorithm=record.algorithm)
                )
            self._model_cache[policy_digest] = model
        return (
            record.run,
            record.normalizer,
            record.sequence_normalizer,
            model,
            False,
        )

    def evaluate_sensitivity(
        self,
        *,
        configuration: str,
        policy_digest: str | None,
        evaluation_range: IndexRange,
        rule_stress: ExecutionRuleStress,
    ) -> RangeEvaluation:
        """Replay a frozen selection without changing selection counters."""

        run, normalizer, sequence_normalizer, model, baseline = self._evaluation_inputs(
            configuration=configuration,
            policy_digest=policy_digest,
        )
        return evaluate_range_evidence(
            dataset=self.dataset,
            evaluation_range=evaluation_range,
            run=run,
            normalizer=normalizer,
            sequence_normalizer=sequence_normalizer,
            model=model,
            baseline=baseline,
            execution_rule_stress=rule_stress,
        )

    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
        if request.dataset_id != self.dataset.dataset_id:
            raise ValueError("candidate evaluation dataset identity mismatch")
        (
            run,
            normalizer,
            sequence_normalizer,
            model,
            baseline,
        ) = self._evaluation_inputs(
            configuration=request.configuration,
            policy_digest=request.policy_digest,
        )
        evidence = evaluate_range_evidence(
            dataset=self.dataset,
            evaluation_range=request.evaluation_range,
            run=run,
            normalizer=normalizer,
            sequence_normalizer=sequence_normalizer,
            model=model,
            baseline=baseline,
        )
        if request.phase is EvaluationPhase.OUTER_TEST:
            self.outer_test_counts[request.fold_index] = (
                self.outer_test_counts.get(request.fold_index, 0) + 1
            )
        score = sum(math.log1p(value) for value in evidence.returns.values)
        duration_days = max(
            request.evaluation_range.size * self.dataset.bar_hours / 24.0,
            1e-12,
        )
        turnover_per_day = evidence.diagnostics.turnover_total / duration_days
        cost_fraction = (
            evidence.diagnostics.total_cost / run.environment.initial_capital
        )
        maximum_drawdown = evaluate_performance(evidence.returns).max_drawdown
        digest = content_digest(
            {
                "configuration": request.configuration,
                "dataset_id": request.dataset_id,
                "fold_index": request.fold_index,
                "phase": request.phase.value,
                "policy_digest": request.policy_digest,
                "range": (
                    request.evaluation_range.start,
                    request.evaluation_range.stop,
                ),
                "diagnostics": evidence.diagnostics.digest_payload(),
                "returns": evidence.returns.values,
                "score": score,
                "turnover_per_day": turnover_per_day,
                "cost_fraction": cost_fraction,
                "maximum_drawdown": maximum_drawdown,
                "schema_version": "market_candidate_evaluation_v3",
            }
        )
        return CandidateEvaluation(
            score=score,
            returns=evidence.returns,
            evaluation_digest=digest,
            diagnostics=evidence.diagnostics,
            turnover_per_day=turnover_per_day,
            cost_fraction=cost_fraction,
            maximum_drawdown=maximum_drawdown,
        )


def _total_return(values: tuple[float, ...]) -> float:
    wealth = 1.0
    for value in values:
        wealth *= 1.0 + float(value)
    return wealth - 1.0


class _SensitivityMetrics(TypedDict):
    cost_fraction: float
    diagnostics: dict[str, object]
    maximum_drawdown: float
    n_trades: int
    returns: tuple[float, ...]
    rule_burden_percentiles: dict[str, object] | None
    total_return: float
    turnover_per_day: float


def _sensitivity_metrics(
    evidence: RangeEvaluation,
    *,
    initial_capital: float,
    duration_days: float,
) -> _SensitivityMetrics:
    values = tuple(float(value) for value in evidence.returns.values)
    return {
        "cost_fraction": evidence.diagnostics.total_cost / initial_capital,
        "diagnostics": evidence.diagnostics.digest_payload(),
        "maximum_drawdown": evaluate_performance(evidence.returns).max_drawdown,
        "n_trades": evidence.diagnostics.n_trades,
        "returns": values,
        "rule_burden_percentiles": evidence.execution_rule_burden,
        "total_return": _total_return(values),
        "turnover_per_day": evidence.diagnostics.turnover_total
        / max(duration_days, 1e-12),
    }


def _evaluate_execution_sensitivity(
    *,
    config: ExecutionSensitivityConfig,
    dataset: MarketDataset,
    result: WalkForwardExecutionResult,
    evaluator: MarketCandidateEvaluator,
    experiment_plan_digest: str,
) -> dict[str, Any] | None:
    if not config.enabled:
        return None
    scenario_pack_digest = content_digest(config.digest_payload())
    folds_payload: list[dict[str, Any]] = []
    required_selected_returns: list[float] = []
    required_baseline_returns: list[float] = []
    required_fold_drawdowns: list[float] = []
    for fold, fold_result in zip(result.folds, result.fold_results, strict=True):
        access = fold_result.sealed_test_access
        if access is None:
            raise RuntimeError("sealed test access evidence is missing for sensitivity")
        selected_configuration = fold_result.selection.selected_configuration
        selected_policy_digest = fold_result.selection.selected_policy_digest
        duration_days = max(fold.test.size * dataset.bar_hours / 24.0, 1e-12)
        scenario_payloads: list[dict[str, Any]] = []
        for scenario in config.scenarios:
            stress = scenario.stress()
            if scenario.name == "nominal":
                nominal_burden = MarketExecutor(
                    dataset,
                    evaluator.baseline_run.environment.execution_cost,
                    rule_stress=stress,
                ).rule_burden_percentiles(
                    start=fold.test.start,
                    stop=fold.test.stop,
                )
                selected_evidence = RangeEvaluation(
                    returns=fold_result.selected_oos.returns,
                    diagnostics=fold_result.selected_oos.diagnostics,
                    execution_rule_burden=nominal_burden,
                )
                baseline_evidence = RangeEvaluation(
                    returns=fold_result.baseline_oos.returns,
                    diagnostics=fold_result.baseline_oos.diagnostics,
                    execution_rule_burden=nominal_burden,
                )
            else:
                selected_evidence = evaluator.evaluate_sensitivity(
                    configuration=selected_configuration,
                    policy_digest=selected_policy_digest,
                    evaluation_range=fold.test,
                    rule_stress=stress,
                )
                baseline_evidence = evaluator.evaluate_sensitivity(
                    configuration=BASELINE_CONFIGURATION,
                    policy_digest=None,
                    evaluation_range=fold.test,
                    rule_stress=stress,
                )
            selected_metrics = _sensitivity_metrics(
                selected_evidence,
                initial_capital=evaluator.baseline_run.environment.initial_capital,
                duration_days=duration_days,
            )
            baseline_metrics = _sensitivity_metrics(
                baseline_evidence,
                initial_capital=evaluator.baseline_run.environment.initial_capital,
                duration_days=duration_days,
            )
            uplift = float(selected_metrics["total_return"]) - float(
                baseline_metrics["total_return"]
            )
            scenario_result = {
                "baseline": baseline_metrics,
                "baseline_uplift": uplift,
                "report_only": scenario.report_only,
                "scenario": scenario.digest_payload(),
                "selected": selected_metrics,
            }
            scenario_result["scenario_result_digest"] = content_digest(scenario_result)
            scenario_payloads.append(scenario_result)
            if scenario.name == config.required_scenario:
                required_selected_returns.extend(selected_evidence.returns.values)
                required_baseline_returns.extend(baseline_evidence.returns.values)
                required_fold_drawdowns.append(
                    float(selected_metrics["maximum_drawdown"])
                )
        access_payload = {
            "base_access_digest": access.access_digest,
            "dataset_id": access.dataset_id,
            "experiment_plan_digest": experiment_plan_digest,
            "fold_index": fold.fold_index,
            "purpose": "post_selection_execution_sensitivity",
            "scenario_pack_digest": scenario_pack_digest,
            "selected_configuration": selected_configuration,
            "selected_policy_digest": selected_policy_digest,
            "test_range": [fold.test.start, fold.test.stop],
        }
        access_payload["access_digest"] = content_digest(access_payload)
        folds_payload.append(
            {
                "access": access_payload,
                "fold_index": fold.fold_index,
                "scenarios": tuple(scenario_payloads),
            }
        )
    selected_values = tuple(float(value) for value in required_selected_returns)
    baseline_values = tuple(float(value) for value in required_baseline_returns)
    selected_total = _total_return(selected_values)
    baseline_total = _total_return(baseline_values)
    required_drawdown = max(required_fold_drawdowns, default=1.0)
    gate = {
        "baseline_total_return": baseline_total,
        "baseline_uplift": selected_total - baseline_total,
        "maximum_fold_drawdown": required_drawdown,
        "maximum_drawdown_threshold": config.maximum_drawdown,
        "minimum_baseline_uplift": config.minimum_baseline_uplift,
        "minimum_selected_return": config.minimum_selected_return,
        "required_scenario": config.required_scenario,
        "selected_total_return": selected_total,
    }
    gate["passed"] = (
        selected_total > config.minimum_selected_return
        and selected_total - baseline_total >= config.minimum_baseline_uplift
        and required_drawdown <= config.maximum_drawdown
    )
    payload: dict[str, Any] = {
        "dataset_id": dataset.dataset_id,
        "experiment_plan_digest": experiment_plan_digest,
        "folds": tuple(folds_payload),
        "gate": gate,
        "production_status": "NO-GO",
        "scenario_pack_digest": scenario_pack_digest,
        "schema_version": "execution_sensitivity_v1",
    }
    payload["artifact_digest"] = content_digest(payload)
    return payload


def _fold_payload(
    fold: WalkForwardFold,
    result: Any,
    *,
    sealed_test_evaluations: int,
    initial_capital: float,
    bar_hours: float,
) -> dict[str, object]:
    access = result.sealed_test_access
    if access is None:
        raise RuntimeError("sealed test access evidence is missing")
    return {
        "baseline_diagnostics": result.baseline_oos.diagnostics.digest_payload(),
        "baseline_returns": result.baseline_oos.returns.values,
        "checkpoint_range": [
            fold.checkpoint_validation.start,
            fold.checkpoint_validation.stop,
        ],
        "fold_index": fold.fold_index,
        "schema_version": "market_walk_forward_fold_v4_deployable_ensemble",
        "sealed_test_access": {
            "access_digest": access.access_digest,
            "dataset_id": access.dataset_id,
            "experiment_plan_digest": access.experiment_plan_digest,
            "fold_index": access.fold_index,
            "schema_version": "sealed_test_access_v1",
            "selected_configuration": access.selected_configuration,
            "selected_policy_digest": access.selected_policy_digest,
            "test_range": [access.test_range.start, access.test_range.stop],
        },
        "sealed_test_evaluations": sealed_test_evaluations,
        "selected_configuration": result.selection.selected_configuration,
        "selected_policy_digest": result.selection.selected_policy_digest,
        "selected_member_policy_digests": result.selected_member_policy_digests,
        "selected_member_seeds": result.selected_member_seeds,
        "selected_diagnostics": result.selected_oos.diagnostics.digest_payload(),
        "selected_cost_fraction": (
            result.selected_oos.diagnostics.total_cost / initial_capital
        ),
        "selected_turnover_per_day": (
            result.selected_oos.diagnostics.turnover_total
            / max(fold.test.size * bar_hours / 24.0, 1e-12)
        ),
        "selected_returns": result.selected_oos.returns.values,
        "candidate_aggregates": tuple(
            item.digest_payload() for item in result.candidate_aggregates
        ),
        "seed_finalists": tuple(
            {
                "checkpoint_evaluation_digest": item.checkpoint_evaluation_digest,
                "checkpoint_score": item.checkpoint_score,
                "configuration": item.configuration,
                "policy_digest": item.policy_digest,
                "seed": item.seed,
                "selection_evaluation_digest": item.selection_evaluation_digest,
                "selection_score": item.selection_score,
            }
            for item in result.seed_finalists
        ),
        "selection_digest": result.selection.digest,
        "selection_range": [
            fold.configuration_selection.start,
            fold.configuration_selection.stop,
        ],
        "test_evaluation_digest": result.test_evaluation_digest,
        "test_range": [fold.test.start, fold.test.stop],
        "train_range": [fold.train.start, fold.train.stop],
    }


def _artifact_paths(root: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name != "run.json"
        )
    )


def _validate_for_store(path: Path) -> bool:
    validate_walk_forward_run_directory(path)
    return True


def _sealed_test_ledger() -> PostgresSealedTestLedger | None:
    database_url = os.environ.get("TRADE_RL_DATABASE_URL")
    if not database_url:
        return None
    catalog = PostgresArtifactCatalog(database_url)
    catalog.migrate()
    return PostgresSealedTestLedger(catalog)


def execute_market_walk_forward(
    *,
    config_path: Path,
    dataset_path: Path,
    store_root: Path,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> WalkForwardRunResult:
    """Run concrete nested walk-forward research and publish immutable evidence."""

    resolved_created_at = created_at or datetime.now(UTC)
    resolved_run_id = run_id or resolved_created_at.strftime("wf-%Y%m%dT%H%M%SZ")
    dataset = load_market_dataset_artifact(dataset_path)
    config = MarketWalkForwardConfig.from_json(config_path, n_bars=dataset.n_bars)
    resolved_signal = resolve_signal_digest(config, dataset_id=dataset.dataset_id)
    config = replace(config, signal_digest=resolved_signal)
    config_digest = content_digest(config.digest_payload())
    provenance = capture_runtime_provenance(
        Path(__file__).resolve().parents[2],
        git_commit=config.candidates[0].run.git_commit,
        git_dirty=config.candidates[0].run.git_dirty,
        deterministic_seed_config={
            "candidate_seeds": tuple(
                {
                    "name": item.name,
                    "seeds": item.run.training.seeds,
                }
                for item in config.candidates
            ),
            "workflow_config_digest": config_digest,
        },
    )
    experiment_plan_digest = _experiment_plan_digest(
        config,
        dataset_id=dataset.dataset_id,
    )
    store = ArtifactStore(store_root)
    stage = store.stage_run(resolved_run_id)
    _write_json(stage / "provenance.json", asdict(provenance))
    registry: dict[str, _PolicyRecord] = {}
    candidate_map = {item.name: item.run for item in config.candidates}
    trainer = MarketCandidateTrainer(
        dataset=dataset,
        candidates=candidate_map,
        root=stage,
        created_at=resolved_created_at,
        registry=registry,
        checkpoint_finalists_per_seed=config.checkpoint_finalists_per_seed,
    )
    evaluator = MarketCandidateEvaluator(
        dataset=dataset,
        baseline_run=config.candidates[0].run,
        registry=registry,
    )
    try:
        fold_runner = ConcreteFoldRunner(
            config=FoldExecutionConfig(
                dataset_id=dataset.dataset_id,
                signal_digest=config.signal_digest,
                candidates=tuple(
                    CandidateConfiguration(item.name) for item in config.candidates
                ),
                minimum_selection_uplift=config.minimum_selection_uplift,
                minimum_selection_score=config.minimum_selection_score,
                minimum_seed_success_fraction=(config.minimum_seed_success_fraction),
                minimum_worst_seed_uplift=config.minimum_worst_seed_uplift,
                maximum_seed_score_std=config.maximum_seed_score_std,
                maximum_selection_turnover_per_day=(
                    config.maximum_selection_turnover_per_day
                ),
                maximum_selection_cost_fraction=(
                    config.maximum_selection_cost_fraction
                ),
                maximum_selection_drawdown=config.maximum_selection_drawdown,
                selected_at=resolved_created_at,
                experiment_plan_digest=experiment_plan_digest,
            ),
            trainer=trainer,
            evaluator=evaluator,
            sealed_test_ledger=_sealed_test_ledger(),
        )
        result: WalkForwardExecutionResult = execute_walk_forward(
            config.workflow,
            dataset_id=dataset.dataset_id,
            runner=fold_runner,
        )
        sensitivity_payload = _evaluate_execution_sensitivity(
            config=config.execution_sensitivity,
            dataset=dataset,
            result=result,
            evaluator=evaluator,
            experiment_plan_digest=experiment_plan_digest,
        )
        sensitivity_by_fold: dict[int, dict[str, Any]] = {}
        if sensitivity_payload is not None:
            _write_json(stage / "execution-sensitivity.json", sensitivity_payload)
            sensitivity_by_fold = {
                int(item["fold_index"]): item
                for item in sensitivity_payload["folds"]
                if isinstance(item, dict)
            }
        folds_payload: list[dict[str, object]] = []
        for fold, fold_result in zip(
            result.folds,
            result.fold_results,
            strict=True,
        ):
            sealed_count = evaluator.outer_test_counts.get(fold.fold_index, 0)
            expected_count = (
                1 if fold_result.selection.selected_policy_digest is None else 2
            )
            if sealed_count != expected_count:
                raise RuntimeError(
                    "sealed outer test evaluation count violates the fold contract"
                )
            payload = _fold_payload(
                fold,
                fold_result,
                sealed_test_evaluations=sealed_count,
                initial_capital=config.candidates[0].run.environment.initial_capital,
                bar_hours=dataset.bar_hours,
            )
            sensitivity_fold = sensitivity_by_fold.get(fold.fold_index)
            if sensitivity_fold is not None:
                access_payload = sensitivity_fold.get("access")
                payload["execution_sensitivity_access"] = access_payload
                payload["execution_sensitivity_scenario_digests"] = tuple(
                    item.get("scenario_result_digest")
                    for item in sensitivity_fold.get("scenarios", ())
                    if isinstance(item, dict)
                )
            folds_payload.append(payload)
            _write_json(
                stage / f"fold-{fold.fold_index:03d}" / "result.json",
                payload,
            )
        walk_forward_payload = {
            "baseline_metrics": (
                None
                if result.baseline_metrics is None
                else asdict(result.baseline_metrics)
            ),
            "baseline_independent_summary": (
                None
                if result.baseline_independent_summary is None
                else asdict(result.baseline_independent_summary)
            ),
            "dataset_id": dataset.dataset_id,
            "evaluation_digest": result.evaluation_digest,
            "execution_sensitivity_digest": (
                None
                if sensitivity_payload is None
                else sensitivity_payload["artifact_digest"]
            ),
            "execution_sensitivity_gate": (
                None if sensitivity_payload is None else sensitivity_payload["gate"]
            ),
            "experiment_plan_digest": experiment_plan_digest,
            "folds": tuple(folds_payload),
            "production_status": "NO-GO",
            "schema_version": "market_walk_forward_run_v5_deployable_ensemble",
            "selected_metrics": (
                None
                if result.selected_metrics is None
                else asdict(result.selected_metrics)
            ),
            "selected_independent_summary": (
                None
                if result.selected_independent_summary is None
                else asdict(result.selected_independent_summary)
            ),
            "stitch_mode": config.workflow.stitch_mode.value,
        }
        _write_json(stage / "walk-forward.json", walk_forward_payload)
        _write_json(stage / "walk-forward-config.json", config.digest_payload())
        _write_json(
            stage / "dataset-reference.json",
            {
                "artifact_path": str(dataset_path),
                "dataset_id": dataset.dataset_id,
                "feature_config_digest": dataset.feature_config_digest,
                "schema_version": "dataset_reference_v2",
            },
        )
        policy_digest = content_digest(
            {
                "policies": tuple(
                    {
                        "algorithm": record.algorithm,
                        "normalizer_digest": record.normalizer.digest,
                        "policy_digest": digest,
                        "run_config_digest": content_digest(
                            record.run.digest_payload()
                        ),
                    }
                    for digest, record in sorted(registry.items())
                ),
                "schema_version": "walk_forward_policy_set_v2",
            }
        )
        environment_digest = content_digest(
            {
                "candidates": tuple(
                    {
                        "name": item.name,
                        "run_environment": {
                            "action": asdict(item.run.action),
                            "environment": asdict(item.run.environment),
                            "risk": asdict(item.run.risk),
                            "reward": asdict(item.run.reward),
                            "trend": asdict(item.run.trend),
                        },
                    }
                    for item in config.candidates
                ),
                "schema_version": "walk_forward_environment_set_v1",
            }
        )
        run_manifest = WalkForwardRunManifest.build(
            root=stage,
            run_id=resolved_run_id,
            dataset_id=dataset.dataset_id,
            environment_digest=environment_digest,
            evaluation_digest=result.evaluation_digest,
            workflow_config_digest=config_digest,
            policy_set_digest=policy_digest,
            provenance_digest=provenance.digest,
            fold_count=len(result.folds),
            artifact_paths=_artifact_paths(stage),
            created_at=resolved_created_at,
        )
        write_walk_forward_run_manifest(stage, run_manifest)
        validate_walk_forward_run_directory(stage)
        published = store.publish_run(resolved_run_id, validate=_validate_for_store)
        return WalkForwardRunResult(
            run_id=resolved_run_id,
            status="published",
            path=published,
            run_digest=run_manifest.digest,
            evaluation_digest=result.evaluation_digest,
            dataset_id=dataset.dataset_id,
        )
    except Exception as error:
        if stage.is_dir():
            try:
                store.mark_failed(resolved_run_id)
            except Exception as isolation_error:
                error.add_note(
                    "failed to isolate partial walk-forward artifacts: "
                    f"{type(isolation_error).__name__}: {isolation_error}"
                )
        raise
