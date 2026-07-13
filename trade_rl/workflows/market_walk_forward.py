"""Concrete real-market nested walk-forward training and evaluation adapters."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.run_manifest import (
    TrainingRunManifest,
    validate_training_run_directory,
    write_training_run_manifest,
)
from trade_rl.artifacts.store import ArtifactStore
from trade_rl.data.artifacts import MarketDatasetView, load_market_dataset_artifact
from trade_rl.data.market import MarketDataset
from trade_rl.domain.datasets import DatasetManifest
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.evaluation.walk_forward.folds import IndexRange, WalkForwardFold
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.observations import observation_passthrough_indices
from trade_rl.rl.training import StableBaselines3Backend, train_residual_ensemble
from trade_rl.strategies.trend import TrendStrategy
from trade_rl.workflows.fold_runner import (
    BASELINE_CONFIGURATION,
    CandidateConfiguration,
    CandidateEvaluation,
    CandidateEvaluationRequest,
    CandidateTrainer,
    CandidateTrainingRequest,
    ConcreteFoldRunner,
    EvaluationPhase,
    FoldExecutionConfig,
    PolicyTrainingArtifact,
)
from trade_rl.workflows.training_run import TrainingRunConfig
from trade_rl.workflows.walk_forward import (
    WalkForwardExecutionResult,
    WalkForwardWorkflowConfig,
    execute_walk_forward,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class NamedCandidateRun:
    name: str
    run: TrainingRunConfig

    def __post_init__(self) -> None:
        if not self.name or self.name == BASELINE_CONFIGURATION:
            raise ValueError("candidate name is empty or reserved")


@dataclass(frozen=True, slots=True)
class MarketWalkForwardConfig:
    workflow: WalkForwardWorkflowConfig
    candidates: tuple[NamedCandidateRun, ...]
    minimum_selection_uplift: float = 0.0
    signal_digest: str = ""
    schema_version: str = "market_walk_forward_config_v1"

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("walk-forward requires at least one candidate")
        names = tuple(item.name for item in self.candidates)
        if len(set(names)) != len(names):
            raise ValueError("walk-forward candidate names must be unique")
        if (
            not math.isfinite(self.minimum_selection_uplift)
            or self.minimum_selection_uplift < 0.0
        ):
            raise ValueError("minimum_selection_uplift must be non-negative")
        if not self.signal_digest:
            object.__setattr__(
                self,
                "signal_digest",
                content_digest(
                    {
                        "schema_version": "trend_baseline_signal_v1",
                        "trend": asdict(self.candidates[0].run.trend),
                    }
                ),
            )
        if len(self.signal_digest) != 64:
            raise ValueError("signal_digest must be a SHA-256 digest")
        common = {
            content_digest(
                {
                    "action": asdict(item.run.action),
                    "alpha_contract": asdict(item.run.alpha_contract),
                    "environment": asdict(item.run.environment),
                    "risk": asdict(item.run.risk),
                    "reward": asdict(item.run.reward),
                    "trend": asdict(item.run.trend),
                }
            )
            for item in self.candidates
        }
        if len(common) != 1:
            raise ValueError(
                "walk-forward candidates must share environment, action, risk, reward, "
                "and trend contracts"
            )
        if self.schema_version != "market_walk_forward_config_v1":
            raise ValueError("unsupported market walk-forward configuration schema")

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        n_bars: int,
    ) -> MarketWalkForwardConfig:
        payload = _mapping(
            json.loads(path.read_text(encoding="utf-8")),
            field="walk-forward config",
        )
        workflow_payload = _mapping(payload.get("workflow"), field="workflow")
        workflow_payload.pop("n_bars", None)
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("candidates must be a non-empty list")
        candidates: list[NamedCandidateRun] = []
        for index, raw_candidate in enumerate(raw_candidates):
            candidate = _mapping(raw_candidate, field=f"candidates[{index}]")
            name = candidate.get("name")
            if not isinstance(name, str):
                raise ValueError(f"candidates[{index}].name must be a string")
            candidates.append(
                NamedCandidateRun(
                    name=name,
                    run=TrainingRunConfig.from_mapping(candidate.get("run")),
                )
            )
        signal = payload.get("signal_digest", "")
        if not isinstance(signal, str):
            raise ValueError("signal_digest must be a string")
        return cls(
            workflow=WalkForwardWorkflowConfig(
                n_bars=n_bars,
                **workflow_payload,
            ),
            candidates=tuple(candidates),
            minimum_selection_uplift=float(
                payload.get("minimum_selection_uplift", 0.0)
            ),
            signal_digest=signal,
            schema_version=str(
                payload.get("schema_version", "market_walk_forward_config_v1")
            ),
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "candidates": tuple(
                {"name": item.name, "run": item.run.digest_payload()}
                for item in self.candidates
            ),
            "minimum_selection_uplift": self.minimum_selection_uplift,
            "schema_version": self.schema_version,
            "signal_digest": self.signal_digest,
            "workflow": asdict(self.workflow),
        }


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
    run: TrainingRunConfig


def _load_model(algorithm: str, path: Path) -> Any:
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(str(path), device="cpu")
    if algorithm == "sac":
        from stable_baselines3 import SAC

        return SAC.load(str(path), device="cpu")
    if algorithm == "td3":
        from stable_baselines3 import TD3

        return TD3.load(str(path), device="cpu")
    if algorithm == "tqc":
        from sb3_contrib import TQC

        return TQC.load(str(path), device="cpu")
    raise ValueError("unsupported walk-forward policy algorithm")


def _environment(
    dataset: MarketDataset,
    run: TrainingRunConfig,
    *,
    normalizer: ObservationNormalizer | None,
    episode_bars: int,
    liquidate_on_end: bool,
) -> ResidualMarketEnv:
    environment_config = replace(
        run.environment,
        episode_bars=episode_bars,
        episode_hour_choices=(),
        initial_state_modes=("cash",),
        liquidate_on_end=liquidate_on_end,
    )
    return ResidualMarketEnv(
        dataset,
        trend_strategy=TrendStrategy(run.trend),
        alpha_enabled=False,
        alpha_contract=run.alpha_contract,
        action_spec=run.action,
        pre_trade_risk=PreTradeRisk(run.risk),
        normalizer=normalizer,
        config=environment_config,
    )


def _training_view(
    dataset: MarketDataset,
    train_range: IndexRange,
    run: TrainingRunConfig,
) -> MarketDataset:
    trend = TrendStrategy(run.trend)
    history = trend.minimum_history_for(dataset)
    history_start = max(0, train_range.start - history)
    return MarketDatasetView(dataset, history_start, train_range.stop).materialize()


def _fit_normalizer(
    dataset: MarketDataset,
    train_range: IndexRange,
    run: TrainingRunConfig,
) -> ObservationNormalizer:
    training_dataset = _training_view(dataset, train_range, run)
    trend = TrendStrategy(run.trend)
    start = trend.minimum_history_for(training_dataset)
    episode_bars = training_dataset.n_bars - 1 - start
    if episode_bars <= 0:
        raise ValueError("training range is too short to fit an observation normalizer")
    env = _environment(
        training_dataset,
        run,
        normalizer=None,
        episode_bars=episode_bars,
        liquidate_on_end=False,
    )
    observations: list[np.ndarray] = []
    try:
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
                np.zeros(run.action.size, dtype=np.float32)
            )
    finally:
        env.close()
    matrix = np.stack(observations, axis=0)
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
        dataset_id=None,
    )


def _normalizer_payload(
    normalizer: ObservationNormalizer,
    *,
    dataset_id: str,
    train_range: IndexRange,
) -> dict[str, object]:
    return {
        "absolute_train_range": [train_range.start, train_range.stop],
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


def _evaluate_range(
    *,
    dataset: MarketDataset,
    evaluation_range: IndexRange,
    run: TrainingRunConfig,
    normalizer: ObservationNormalizer | None,
    model: Any | None,
    baseline: bool,
) -> ReturnSeries:
    start_index = evaluation_range.start - 1
    minimum = TrendStrategy(run.trend).minimum_history_for(dataset)
    if start_index < minimum:
        raise ValueError("evaluation range lacks causal trend history")
    env = _environment(
        dataset,
        run,
        normalizer=normalizer,
        episode_bars=evaluation_range.size,
        liquidate_on_end=True,
    )
    try:
        observation, _ = env.reset(
            seed=0,
            options={
                "episode_bars": evaluation_range.size,
                "initial_state_mode": "cash",
                "start_idx": start_index,
            },
        )
        terminated = False
        truncated = False
        while not terminated and not truncated:
            if baseline:
                action = np.zeros(run.action.size, dtype=np.float32)
            else:
                if model is None:
                    raise RuntimeError("residual evaluation requires a loaded model")
                raw_action, _ = model.predict(observation, deterministic=True)
                action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            observation, _, terminated, truncated, _ = env.step(action)
        values = tuple(
            float(value)
            for value in (
                env.shadow.returns_history if baseline else env.hybrid.returns_history
            )
        )
    finally:
        env.close()
    if len(values) != evaluation_range.size:
        raise ValueError(
            "range-restricted environment produced an unexpected return length"
        )
    return ReturnSeries(
        values=values,
        kind=ReturnKind.BASE_BAR,
        periods_per_year=dataset.periods_per_year,
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
    ) -> None:
        self.dataset = dataset
        self.candidates = candidates
        self.root = root
        self.created_at = created_at
        self.registry = registry
        self._normalizers: dict[int, ObservationNormalizer] = {}

    def _normalizer(
        self,
        request: CandidateTrainingRequest,
        run: TrainingRunConfig,
    ) -> ObservationNormalizer:
        existing = self._normalizers.get(request.fold_index)
        if existing is not None:
            return existing
        normalizer = _fit_normalizer(self.dataset, request.train, run)
        self._normalizers[request.fold_index] = normalizer
        _write_json(
            self.root / f"fold-{request.fold_index:03d}" / "normalizer.json",
            _normalizer_payload(
                normalizer,
                dataset_id=self.dataset.dataset_id,
                train_range=request.train,
            ),
        )
        return normalizer

    def train(self, request: CandidateTrainingRequest) -> PolicyTrainingArtifact:
        if request.dataset_id != self.dataset.dataset_id:
            raise ValueError("candidate training dataset identity mismatch")
        run = self.candidates[request.configuration.name]
        normalizer = self._normalizer(request, run)
        training_dataset = _training_view(self.dataset, request.train, run)
        trend = TrendStrategy(run.trend)
        maximum_episode = training_dataset.n_bars - 1 - trend.minimum_history_for(
            training_dataset
        )
        if maximum_episode <= 0:
            raise ValueError("fold training range is too short")
        configured_episode = run.environment.resolve_nominal_episode_bars(
            training_dataset
        )
        episode_bars = min(configured_episode, maximum_episode)
        training_environment = replace(
            run.environment,
            episode_bars=episode_bars,
            episode_hour_choices=(),
            liquidate_on_end=False,
        )

        def factory() -> ResidualMarketEnv:
            return ResidualMarketEnv(
                training_dataset,
                trend_strategy=TrendStrategy(run.trend),
                alpha_enabled=False,
                alpha_contract=run.alpha_contract,
                action_spec=run.action,
                pre_trade_risk=PreTradeRisk(run.risk),
                normalizer=normalizer,
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
            backend=StableBaselines3Backend(factory),
            output_dir=candidate_root / "members",
            created_at=self.created_at,
        )
        _write_json(candidate_root / "ensemble.json", asdict(ensemble))
        _write_json(candidate_root / "training-config.json", run.digest_payload())

        scored: list[tuple[float, str]] = []
        for index, member in enumerate(ensemble.members):
            path = candidate_root / "members" / f"member-{index:03d}" / "policy.zip"
            record = _PolicyRecord(
                path=path,
                algorithm=run.training.algorithm,
                normalizer=normalizer,
                run=run,
            )
            self.registry[member.checkpoint_digest] = record
            model = _load_model(record.algorithm, record.path)
            returns = _evaluate_range(
                dataset=self.dataset,
                evaluation_range=request.checkpoint_validation,
                run=run,
                normalizer=normalizer,
                model=model,
                baseline=False,
            )
            score = sum(math.log1p(value) for value in returns.values)
            scored.append((score, member.checkpoint_digest))
        selected_digest = min(scored, key=lambda item: (-item[0], item[1]))[1]
        _write_json(
            candidate_root / "checkpoint-selection.json",
            {
                "candidates": tuple(
                    {"policy_digest": digest, "score": score}
                    for score, digest in sorted(scored, key=lambda item: item[1])
                ),
                "checkpoint_range": [
                    request.checkpoint_validation.start,
                    request.checkpoint_validation.stop,
                ],
                "schema_version": "checkpoint_selection_v1",
                "selected_policy_digest": selected_digest,
            },
        )
        return PolicyTrainingArtifact(
            configuration=request.configuration.name,
            policy_digest=selected_digest,
        )


class MarketCandidateEvaluator:
    """Evaluate only the exact range carried by a candidate evaluation request."""

    def __init__(
        self,
        *,
        dataset: MarketDataset,
        baseline_run: TrainingRunConfig,
        registry: dict[str, _PolicyRecord],
    ) -> None:
        self.dataset = dataset
        self.baseline_run = baseline_run
        self.registry = registry
        self._model_cache: dict[str, Any] = {}
        self.outer_test_counts: dict[int, int] = {}

    def evaluate(self, request: CandidateEvaluationRequest) -> CandidateEvaluation:
        if request.dataset_id != self.dataset.dataset_id:
            raise ValueError("candidate evaluation dataset identity mismatch")
        baseline = request.configuration == BASELINE_CONFIGURATION
        if baseline:
            run = self.baseline_run
            normalizer = None
            model = None
        else:
            if request.policy_digest is None:
                raise ValueError("residual evaluation requires a policy digest")
            record = self.registry.get(request.policy_digest)
            if record is None:
                raise ValueError("candidate evaluation policy is not registered")
            run = record.run
            normalizer = record.normalizer
            model = self._model_cache.get(request.policy_digest)
            if model is None:
                model = _load_model(record.algorithm, record.path)
                self._model_cache[request.policy_digest] = model
        returns = _evaluate_range(
            dataset=self.dataset,
            evaluation_range=request.evaluation_range,
            run=run,
            normalizer=normalizer,
            model=model,
            baseline=baseline,
        )
        if request.phase is EvaluationPhase.OUTER_TEST:
            self.outer_test_counts[request.fold_index] = (
                self.outer_test_counts.get(request.fold_index, 0) + 1
            )
        score = sum(math.log1p(value) for value in returns.values)
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
                "returns": returns.values,
                "score": score,
                "schema_version": "market_candidate_evaluation_v1",
            }
        )
        return CandidateEvaluation(
            score=score,
            returns=returns,
            evaluation_digest=digest,
        )


def _fold_payload(
    fold: WalkForwardFold,
    result: Any,
    *,
    sealed_test_evaluations: int,
) -> dict[str, object]:
    return {
        "baseline_returns": result.baseline_oos.returns.values,
        "checkpoint_range": [
            fold.checkpoint_validation.start,
            fold.checkpoint_validation.stop,
        ],
        "fold_index": fold.fold_index,
        "schema_version": "market_walk_forward_fold_v1",
        "sealed_test_evaluations": sealed_test_evaluations,
        "selected_configuration": result.selection.selected_configuration,
        "selected_policy_digest": result.selection.selected_policy_digest,
        "selected_returns": result.selected_oos.returns.values,
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
    validate_training_run_directory(path)
    return True


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
    store = ArtifactStore(store_root)
    stage = store.stage_run(resolved_run_id)
    registry: dict[str, _PolicyRecord] = {}
    candidate_map = {item.name: item.run for item in config.candidates}
    trainer = MarketCandidateTrainer(
        dataset=dataset,
        candidates=candidate_map,
        root=stage,
        created_at=resolved_created_at,
        registry=registry,
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
                selected_at=resolved_created_at,
            ),
            trainer=trainer,
            evaluator=evaluator,
        )
        result: WalkForwardExecutionResult = execute_walk_forward(
            config.workflow,
            dataset_id=dataset.dataset_id,
            runner=fold_runner,
        )
        folds_payload: list[dict[str, object]] = []
        for fold, fold_result in zip(
            result.folds,
            result.fold_results,
            strict=True,
        ):
            payload = _fold_payload(
                fold,
                fold_result,
                sealed_test_evaluations=evaluator.outer_test_counts.get(
                    fold.fold_index, 0
                ),
            )
            folds_payload.append(payload)
            _write_json(
                stage / f"fold-{fold.fold_index:03d}" / "result.json",
                payload,
            )
        walk_forward_payload = {
            "baseline_metrics": asdict(result.baseline_metrics),
            "dataset_id": dataset.dataset_id,
            "evaluation_digest": result.evaluation_digest,
            "folds": tuple(folds_payload),
            "production_status": "NO-GO",
            "schema_version": "market_walk_forward_run_v1",
            "selected_metrics": asdict(result.selected_metrics),
        }
        _write_json(stage / "walk-forward.json", walk_forward_payload)
        _write_json(stage / "walk-forward-config.json", config.digest_payload())
        _write_json(
            stage / "dataset-reference.json",
            {
                "artifact_path": str(dataset_path),
                "dataset_id": dataset.dataset_id,
                "schema_version": "dataset_reference_v1",
            },
        )
        policy_digest = content_digest(
            {
                "policies": tuple(sorted(registry)),
                "schema_version": "walk_forward_policy_set_v1",
            }
        )
        environment_digest = content_digest(
            {
                "action": asdict(config.candidates[0].run.action),
                "environment": asdict(config.candidates[0].run.environment),
                "risk": asdict(config.candidates[0].run.risk),
                "reward": asdict(config.candidates[0].run.reward),
                "trend": asdict(config.candidates[0].run.trend),
            }
        )
        config_digest = content_digest(config.digest_payload())
        run_manifest = TrainingRunManifest.build(
            root=stage,
            run_id=resolved_run_id,
            dataset_id=dataset.dataset_id,
            environment_digest=environment_digest,
            ensemble_digest=policy_digest,
            training_config_digest=config_digest,
            artifact_paths=_artifact_paths(stage),
            created_at=resolved_created_at,
        )
        write_training_run_manifest(stage, run_manifest)
        validate_training_run_directory(stage)
        published = store.publish_run(resolved_run_id, validate=_validate_for_store)
        return WalkForwardRunResult(
            run_id=resolved_run_id,
            status="published",
            path=published,
            run_digest=run_manifest.digest,
            evaluation_digest=result.evaluation_digest,
            dataset_id=dataset.dataset_id,
        )
    except Exception:
        if stage.is_dir():
            store.mark_failed(resolved_run_id)
        raise
