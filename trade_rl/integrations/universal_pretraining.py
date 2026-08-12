"""Preassembled multi-symbol Universal BC and critic warm-start evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.integrations.policy_stage_snapshot import write_policy_stage_snapshot
from trade_rl.integrations.sb3_behavior_cloning import (
    _behavior_cloning_gate_thresholds,
    _behavior_cloning_quality,
    _evaluate_hierarchical_behavior_cloning_gate,
    _hierarchical_behavior_cloning_config,
    _hierarchical_teacher_labels,
    _teacher_change_labels,
)
from trade_rl.integrations.universal_behavior_cloning import pretrain_universal_policy
from trade_rl.integrations.universal_critic_warm_start import (
    warm_start_policy_actor_critic,
)
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaTeacherHoldoutMetric,
    evaluate_causal_alpha_teacher_admission,
)
from trade_rl.learning.direct_bc_evaluation import (
    evaluate_direct_behavior_cloning_gates,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_bc import (
    EpisodeBehaviorCloningHoldoutEvaluation,
    aggregate_episode_behavior_cloning_holdouts,
    evaluate_episode_action_path,
    evaluate_episode_behavior_cloning_holdout,
)
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.learning.evaluation import write_learning_evaluation
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
from trade_rl.learning.universal_bc import CriticWarmStartPlan, UniversalTeacherArtifact
from trade_rl.rl.training import ResidualTrainingConfig

SymbolTeacherInput = tuple[SupervisedPolicyDataset, BehaviorCloningSplit, np.ndarray]


@dataclass(frozen=True, slots=True)
class UniversalPretrainingBundle:
    """One immutable train-only teacher bundle shared by all Universal PPO members."""

    dataset: SupervisedPolicyDataset
    split: BehaviorCloningSplit
    symbol_sample_indices: Mapping[str, tuple[int, ...]]
    symbol_splits: Mapping[str, BehaviorCloningSplit]
    critic_targets: np.ndarray
    train_symbols: tuple[str, ...]
    teacher_artifact: UniversalTeacherArtifact
    episode_batches: Mapping[str, EpisodeOracleBatch] = field(default_factory=dict)
    causal_teacher_selection_evidence: Mapping[str, object] | None = None
    causal_teacher_episode_hours: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, SupervisedPolicyDataset):
            raise TypeError("dataset must be SupervisedPolicyDataset")
        if not isinstance(self.split, BehaviorCloningSplit):
            raise TypeError("split must be BehaviorCloningSplit")
        symbols = tuple(self.train_symbols)
        if not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("train_symbols must be non-empty and unique")
        if set(self.symbol_sample_indices) != set(symbols):
            raise ValueError("symbol sample scope must exactly match train_symbols")
        if set(self.symbol_splits) != set(symbols) or any(
            not isinstance(split, BehaviorCloningSplit)
            for split in self.symbol_splits.values()
        ):
            raise ValueError("symbol split scope must exactly match train_symbols")
        episode_batches = dict(self.episode_batches)
        if episode_batches and (
            set(episode_batches) != set(symbols)
            or any(
                not isinstance(batch, EpisodeOracleBatch)
                for batch in episode_batches.values()
            )
        ):
            raise ValueError("episode batch scope must exactly match train_symbols")
        train_scope = {int(value) for value in self.split.train_indices}
        observed: set[int] = set()
        normalized: dict[str, tuple[int, ...]] = {}
        for symbol in symbols:
            values = tuple(self.symbol_sample_indices[symbol])
            if not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ):
                raise ValueError("symbol sample indices are invalid")
            if len(set(values)) != len(values) or not set(values) <= train_scope:
                raise ValueError("symbol sample indices leave the train scope")
            if observed.intersection(values):
                raise ValueError("symbol sample indices overlap")
            observed.update(values)
            normalized[symbol] = values
        if observed != train_scope:
            raise ValueError("symbol sample indices must close over the train scope")
        targets = np.asarray(self.critic_targets, dtype=np.float32).copy(order="C")
        if targets.ndim != 1 or targets.shape[0] != self.dataset.sample_count:
            raise ValueError(
                "critic_targets must align with the combined teacher dataset"
            )
        if not np.isfinite(targets).all():
            raise ValueError("critic_targets must be finite")
        targets.setflags(write=False)
        if self.teacher_artifact.train_symbols != symbols:
            raise ValueError("teacher artifact train symbol scope mismatch")
        selection_evidence = self.causal_teacher_selection_evidence
        episode_hours = self.causal_teacher_episode_hours
        if selection_evidence is None:
            if episode_hours is not None:
                raise ValueError(
                    "causal teacher episode hours require selection evidence"
                )
        else:
            selection_evidence = dict(selection_evidence)
            if (
                selection_evidence.get("schema_version")
                != "causal_alpha_selection_evidence_v1"
            ):
                raise ValueError("causal teacher selection evidence schema mismatch")
            artifact_digest = selection_evidence.get("artifact_digest")
            if not isinstance(artifact_digest, str) or len(artifact_digest) != 64:
                raise ValueError("causal teacher selection evidence digest is invalid")
            if (
                episode_hours is None
                or not math.isfinite(episode_hours)
                or episode_hours <= 0.0
            ):
                raise ValueError("causal teacher episode hours must be positive")
        object.__setattr__(self, "train_symbols", symbols)
        object.__setattr__(self, "symbol_sample_indices", normalized)
        object.__setattr__(self, "symbol_splits", dict(self.symbol_splits))
        object.__setattr__(self, "episode_batches", episode_batches)
        object.__setattr__(
            self, "causal_teacher_selection_evidence", selection_evidence
        )
        object.__setattr__(self, "causal_teacher_episode_hours", episode_hours)
        object.__setattr__(self, "critic_targets", targets)


def _validated_local_partition(
    dataset: SupervisedPolicyDataset,
    split: BehaviorCloningSplit,
) -> None:
    arrays = (
        split.train_indices,
        split.validation_indices,
        split.purged_indices,
    )
    partition = np.concatenate(arrays)
    expected = np.arange(dataset.sample_count, dtype=np.int64)
    if partition.size != expected.size or not np.array_equal(
        np.sort(partition), expected
    ):
        raise ValueError("symbol teacher split must partition the full dataset")


def _concatenate_observations(
    datasets: Sequence[SupervisedPolicyDataset],
) -> np.ndarray | Mapping[str, np.ndarray]:
    first = datasets[0].observations
    if isinstance(first, Mapping):
        keys = tuple(sorted(first))
        combined: dict[str, np.ndarray] = {}
        for key in keys:
            arrays: list[np.ndarray] = []
            trailing_shape: tuple[int, ...] | None = None
            for dataset in datasets:
                observations = dataset.observations
                if (
                    not isinstance(observations, Mapping)
                    or tuple(sorted(observations)) != keys
                ):
                    raise ValueError(
                        "Universal teacher observation key closure mismatch"
                    )
                array = np.asarray(observations[key])
                shape = tuple(int(value) for value in array.shape[1:])
                if trailing_shape is None:
                    trailing_shape = shape
                elif shape != trailing_shape:
                    raise ValueError("Universal teacher observation shape mismatch")
                arrays.append(array)
            combined[key] = np.concatenate(arrays, axis=0)
        return combined
    if any(isinstance(dataset.observations, Mapping) for dataset in datasets[1:]):
        raise ValueError("Universal teacher observation representation mismatch")
    arrays = [np.asarray(dataset.observations) for dataset in datasets]
    trailing = tuple(int(value) for value in arrays[0].shape[1:])
    if any(
        tuple(int(value) for value in array.shape[1:]) != trailing for array in arrays
    ):
        raise ValueError("Universal teacher observation shape mismatch")
    return np.concatenate(arrays, axis=0)


def _offset_episode_ids(
    values: np.ndarray,
    *,
    mapping: Mapping[int, int],
) -> np.ndarray:
    return np.asarray([mapping[int(value)] for value in values], dtype=np.int64)


def combine_symbol_teachers(
    symbol_teachers: Mapping[str, SymbolTeacherInput],
    *,
    train_symbols: Sequence[str],
    normalizer_digest: str,
    feature_schema_digest: str,
) -> UniversalPretrainingBundle:
    """Combine per-symbol teacher evidence without losing split provenance."""

    symbols = tuple(train_symbols)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("train_symbols must be non-empty and unique")
    if set(symbol_teachers) != set(symbols):
        raise ValueError("symbol_teachers must exactly match train_symbols")

    datasets: list[SupervisedPolicyDataset] = []
    targets: list[np.ndarray] = []
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []
    purged_indices: list[np.ndarray] = []
    train_episode_ids: list[np.ndarray] = []
    validation_episode_ids: list[np.ndarray] = []
    purged_episode_ids: list[np.ndarray] = []
    symbol_sample_indices: dict[str, tuple[int, ...]] = {}
    symbol_splits: dict[str, BehaviorCloningSplit] = {}
    offset = 0
    next_episode_id = 0
    action_spec_digest: str | None = None
    teacher_config_digest: str | None = None
    teacher_identity_rows: list[dict[str, object]] = []

    for symbol in symbols:
        dataset, split, raw_targets = symbol_teachers[symbol]
        if not isinstance(dataset, SupervisedPolicyDataset):
            raise TypeError("symbol teacher dataset must be SupervisedPolicyDataset")
        if not isinstance(split, BehaviorCloningSplit):
            raise TypeError("symbol teacher split must be BehaviorCloningSplit")
        _validated_local_partition(dataset, split)
        critic = np.asarray(raw_targets, dtype=np.float32)
        if critic.ndim != 1 or critic.size != dataset.sample_count:
            raise ValueError("symbol critic targets must align with teacher samples")
        if not np.isfinite(critic).all():
            raise ValueError("symbol critic targets must be finite")
        if action_spec_digest is None:
            action_spec_digest = dataset.action_spec_digest
        elif dataset.action_spec_digest != action_spec_digest:
            raise ValueError("Universal teacher action specification mismatch")
        if teacher_config_digest is None:
            teacher_config_digest = dataset.teacher_config_digest
        elif dataset.teacher_config_digest != teacher_config_digest:
            raise ValueError("Universal teacher configuration mismatch")

        datasets.append(dataset)
        targets.append(critic)
        shifted_train = np.asarray(split.train_indices + offset, dtype=np.int64)
        shifted_validation = np.asarray(
            split.validation_indices + offset, dtype=np.int64
        )
        shifted_purged = np.asarray(split.purged_indices + offset, dtype=np.int64)
        train_indices.append(shifted_train)
        validation_indices.append(shifted_validation)
        purged_indices.append(shifted_purged)
        symbol_sample_indices[symbol] = tuple(int(value) for value in shifted_train)
        symbol_splits[symbol] = split

        local_episode_ids = sorted(
            {
                int(value)
                for array in (
                    split.train_episode_ids,
                    split.validation_episode_ids,
                    split.purged_episode_ids,
                )
                for value in array
            }
        )
        episode_mapping = {
            local: next_episode_id + index
            for index, local in enumerate(local_episode_ids)
        }
        next_episode_id += len(local_episode_ids)
        train_episode_ids.append(
            _offset_episode_ids(split.train_episode_ids, mapping=episode_mapping)
        )
        validation_episode_ids.append(
            _offset_episode_ids(split.validation_episode_ids, mapping=episode_mapping)
        )
        purged_episode_ids.append(
            _offset_episode_ids(split.purged_episode_ids, mapping=episode_mapping)
        )
        teacher_identity_rows.append(
            {
                "action_digest": dataset.action_digest,
                "critic_target_digest": content_digest(critic.tolist()),
                "dataset_id": dataset.dataset_id,
                "environment_digest": dataset.environment_digest,
                "observation_digest": dataset.observation_digest,
                "split": {
                    "purged": tuple(int(value) for value in split.purged_indices),
                    "train": tuple(int(value) for value in split.train_indices),
                    "validation": tuple(
                        int(value) for value in split.validation_indices
                    ),
                },
                "symbol": symbol,
            }
        )
        offset += dataset.sample_count

    assert action_spec_digest is not None
    assert teacher_config_digest is not None
    observations = _concatenate_observations(datasets)
    actions = np.concatenate(
        [np.asarray(dataset.actions) for dataset in datasets], axis=0
    )
    total = int(actions.shape[0])
    combined_dataset = SupervisedPolicyDataset(
        observations=observations,
        actions=actions,
        dataset_id=content_digest(
            {
                "dataset_ids": tuple(dataset.dataset_id for dataset in datasets),
                "schema_version": "universal_teacher_dataset_v1",
                "train_symbols": symbols,
            }
        ),
        train_start=0,
        train_stop=total + 1,
        environment_digest=content_digest(
            {
                "environment_digests": tuple(
                    dataset.environment_digest for dataset in datasets
                ),
                "schema_version": "universal_teacher_environment_bundle_v1",
            }
        ),
        action_spec_digest=action_spec_digest,
        teacher_config_digest=teacher_config_digest,
    )
    combined_split = BehaviorCloningSplit(
        train_indices=np.concatenate(train_indices),
        validation_indices=np.concatenate(validation_indices),
        purged_indices=np.concatenate(purged_indices),
        train_episode_ids=np.concatenate(train_episode_ids),
        validation_episode_ids=np.concatenate(validation_episode_ids),
        purged_episode_ids=np.concatenate(purged_episode_ids),
    )
    teacher_artifact = UniversalTeacherArtifact.create(
        teacher_digest=content_digest(
            {
                "schema_version": "symbol_balanced_universal_teacher_v1",
                "symbols": teacher_identity_rows,
            }
        ),
        train_symbols=symbols,
        teacher_symbols=symbols,
        normalizer_digest=normalizer_digest,
        feature_schema_digest=feature_schema_digest,
    )
    return UniversalPretrainingBundle(
        dataset=combined_dataset,
        split=combined_split,
        symbol_sample_indices=symbol_sample_indices,
        symbol_splits=symbol_splits,
        critic_targets=np.concatenate(targets),
        train_symbols=symbols,
        teacher_artifact=teacher_artifact,
    )


def build_universal_pretraining_hook(
    bundle: UniversalPretrainingBundle,
    *,
    symbol_environment_factories: Mapping[str, Callable[[], Any]] | None = None,
) -> Any:
    """Return the SB3 pretraining hook for one immutable Universal teacher bundle."""

    if not isinstance(bundle, UniversalPretrainingBundle):
        raise TypeError("bundle must be UniversalPretrainingBundle")
    environment_factories = dict(symbol_environment_factories or {})
    if environment_factories and set(environment_factories) != set(
        bundle.train_symbols
    ):
        raise ValueError("Universal holdout environment scope mismatch")

    def hook(
        *,
        policy: Any,
        config: ResidualTrainingConfig,
        behavior_cloning_seed: int,
        member_seed: int,
        output_root: Path,
    ) -> dict[str, object]:
        if config.behavior_cloning_teacher == "causal_alpha_ridge":
            selection = bundle.causal_teacher_selection_evidence
            episode_hours = bundle.causal_teacher_episode_hours
            if selection is None or episode_hours is None:
                raise RuntimeError(
                    "Universal causal teacher selection evidence is unavailable"
                )
            if not bundle.episode_batches:
                raise RuntimeError(
                    "Universal causal teacher episode batches are unavailable"
                )
            if set(environment_factories) != set(bundle.train_symbols):
                raise RuntimeError(
                    "Universal causal teacher holdout environment factories are unavailable"
                )
            atomic_write_bytes(
                output_root / "causal-teacher-selection.json",
                canonical_json_bytes(selection) + b"\n",
            )
            episode_days = episode_hours / 24.0
            teacher_metrics: list[CausalAlphaTeacherHoldoutMetric] = []
            for symbol in bundle.train_symbols:
                batch = bundle.episode_batches[symbol]
                if not batch.contracts or len(batch.targets) != len(batch.contracts):
                    raise RuntimeError(
                        f"Universal causal teacher holdout batch is invalid for {symbol}"
                    )
                evaluation = evaluate_episode_action_path(
                    environment_factories[symbol],
                    batch.contracts[-1],
                    actions=batch.targets[-1],
                )
                performance = evaluation.performance
                teacher_metrics.append(
                    CausalAlphaTeacherHoldoutMetric(
                        symbol=symbol,
                        gross_return=float(performance.gross_return),
                        net_return=float(performance.net_return),
                        turnover_per_day=float(performance.turnover_total)
                        / episode_days,
                        total_execution_cost=float(performance.cost_total),
                        trade_count=int(performance.trade_count),
                        maximum_drawdown=float(performance.maximum_drawdown),
                    )
                )
            teacher_admission = evaluate_causal_alpha_teacher_admission(
                tuple(teacher_metrics)
            )
            atomic_write_bytes(
                output_root / "causal-teacher-admission.json",
                canonical_json_bytes(teacher_admission.to_payload()) + b"\n",
            )
            if not teacher_admission.passed:
                raise RuntimeError(
                    "Universal causal teacher admission failed before behavior cloning"
                )
        bc_config = _hierarchical_behavior_cloning_config(config)
        validation_count = int(bundle.split.validation_indices.size)
        if validation_count:
            validation_fraction = math.nextafter(
                validation_count / bundle.dataset.sample_count,
                1.0,
            )
            if validation_fraction >= 0.5:
                raise ValueError(
                    "Universal episode holdout must remain below half the samples"
                )
            bc_config = replace(
                bc_config,
                validation_fraction=validation_fraction,
            )
        labels = _hierarchical_teacher_labels(
            policy=policy,
            teacher_dataset=bundle.dataset,
            config=config,
        )

        progress_history: list[dict[str, object]] = []

        def write_behavior_cloning_progress(progress: dict[str, object]) -> None:
            progress_history.append(dict(progress))
            payload = {
                "behavior_cloning_seed": behavior_cloning_seed,
                "history": tuple(progress_history),
                "member_seed": member_seed,
                "schema_version": "universal_behavior_cloning_progress_v1",
                **progress,
            }
            atomic_write_bytes(
                output_root / "behavior-cloning-progress.json",
                canonical_json_bytes(payload) + b"\n",
            )

        bc_result = pretrain_universal_policy(
            policy,
            bundle.dataset,
            symbol_sample_indices=bundle.symbol_sample_indices,
            train_symbols=bundle.train_symbols,
            config=bc_config,
            split=bundle.split,
            seed=behavior_cloning_seed,
            observation_provider=None,
            output_root=output_root / "behavior-cloning",
            hierarchical_labels=labels,
            progress_callback=write_behavior_cloning_progress,
        )
        relative_improvement, bc_passed = _behavior_cloning_quality(
            initial_mse=float(bc_result.initial_mse),
            final_mse=float(bc_result.final_mse),
            required_relative_improvement=(
                config.behavior_cloning_required_relative_improvement
            ),
        )
        if not bc_passed:
            raise RuntimeError(
                "Universal behavior cloning failed the reconstruction improvement gate"
            )
        bc_digest = getattr(bc_result, "digest", None)
        if not isinstance(bc_digest, str) or len(bc_digest) != 64:
            raise ValueError("Universal behavior cloning result digest is invalid")
        hierarchical_payload = {
            name: (
                None
                if (value := getattr(bc_result, attribute, None)) is None
                else asdict(value)
            )
            for name, attribute in (
                ("initial_losses", "initial_hierarchical_losses"),
                ("final_losses", "final_hierarchical_losses"),
                ("validation_losses", "validation_hierarchical_losses"),
                ("initial_metrics", "initial_hierarchical_metrics"),
                ("final_metrics", "final_hierarchical_metrics"),
                ("validation_metrics", "validation_hierarchical_metrics"),
            )
        }
        result_payload = {
            "behavior_cloning_digest": bc_digest,
            "best_epoch": int(bc_result.best_epoch),
            "excluded_sample_count": int(bc_result.excluded_sample_count),
            "final_mse": float(bc_result.final_mse),
            "hierarchical": hierarchical_payload,
            "initial_mse": float(bc_result.initial_mse),
            "sample_count": int(bc_result.sample_count),
            "schema_version": "universal_behavior_cloning_result_v2",
            "training_sample_count": int(bc_result.training_sample_count),
            "validation_mse": (
                None
                if bc_result.validation_mse is None
                else float(bc_result.validation_mse)
            ),
            "validation_sample_count": int(bc_result.validation_sample_count),
        }
        result_artifact_digest = content_digest(result_payload)
        atomic_write_bytes(
            output_root / "behavior-cloning-result.json",
            canonical_json_bytes(
                {**result_payload, "artifact_digest": result_artifact_digest}
            )
            + b"\n",
        )
        write_policy_stage_snapshot(
            policy,
            output_root=output_root,
            stage="behavior_cloning",
            member_seed=member_seed,
        )

        holdout_digest: str | None = None
        gate_digest: str | None = None
        if config.behavior_cloning_validation_fraction > 0.0:
            if not bundle.episode_batches:
                raise RuntimeError("Universal Oracle episode batches are unavailable")
            if set(environment_factories) != set(bundle.train_symbols):
                raise RuntimeError(
                    "Universal causal holdout environment factories are unavailable"
                )
            holdouts: list[EpisodeBehaviorCloningHoldoutEvaluation] = []
            for symbol in bundle.train_symbols:
                _, holdout = evaluate_episode_behavior_cloning_holdout(
                    environment_factory=environment_factories[symbol],
                    model=policy,
                    batch=bundle.episode_batches[symbol],
                    split=bundle.symbol_splits[symbol],
                    output_root=output_root / "behavior-cloning-holdout" / symbol,
                    bootstrap_confidence_level=(
                        config.behavior_cloning_causal_holdout_confidence_level
                    ),
                    bootstrap_resamples=(
                        config.behavior_cloning_causal_holdout_bootstrap_resamples
                    ),
                )
                if holdout is None:
                    raise RuntimeError(
                        f"Universal causal holdout is empty for {symbol}"
                    )
                holdouts.append(holdout)
            aggregate_holdout = aggregate_episode_behavior_cloning_holdouts(
                tuple(holdouts),
                seed_material=content_digest(
                    {
                        "member_seed": member_seed,
                        "teacher_artifact_digest": (
                            bundle.teacher_artifact.artifact_digest
                        ),
                    }
                ),
            )
            holdout_payload = aggregate_holdout.to_dict()
            holdout_digest = content_digest(holdout_payload)
            atomic_write_bytes(
                output_root / "behavior-cloning-holdout.json",
                canonical_json_bytes(
                    {**holdout_payload, "artifact_digest": holdout_digest}
                )
                + b"\n",
            )
            thresholds = _behavior_cloning_gate_thresholds(config)
            if labels is not None:
                gate = _evaluate_hierarchical_behavior_cloning_gate(
                    cloning=bc_result,
                    holdout=aggregate_holdout,
                    thresholds=thresholds,
                )
            else:
                teacher_changes = _teacher_change_labels(
                    teacher_dataset=bundle.dataset,
                    config=config,
                )
                if teacher_changes is None:
                    raise RuntimeError(
                        "Universal teacher change labels are unavailable"
                    )
                gate = evaluate_direct_behavior_cloning_gates(
                    initial_mse=float(bc_result.initial_mse),
                    final_mse=float(bc_result.final_mse),
                    teacher_change_support=(
                        teacher_changes.diagnostics.gate_positive_count
                    ),
                    holdout=aggregate_holdout,
                    thresholds=thresholds,
                )
            gate_digest = write_learning_evaluation(
                output_root / "behavior-cloning-gates.json",
                gate,
            )
            gate.require_passed()

        critic_digest: str | None = None
        critic_only_steps = config.behavior_cloning_critic_warm_start_steps
        joint_steps = config.behavior_cloning_joint_warm_start_steps
        if critic_only_steps > 0 or joint_steps > 0:
            plan = CriticWarmStartPlan(
                critic_only_steps=critic_only_steps,
                joint_fine_tune_steps=joint_steps,
                joint_actor_learning_rate_scale=(
                    config.behavior_cloning_joint_warm_start_actor_lr_scale
                ),
            )
            warm_result = warm_start_policy_actor_critic(
                policy,
                bundle.dataset,
                bundle.critic_targets,
                plan=plan,
                batch_size=config.behavior_cloning_batch_size,
                learning_rate=(config.behavior_cloning_critic_warm_start_learning_rate),
                seed=behavior_cloning_seed,
                sample_indices=bundle.split.train_indices,
                observation_provider=None,
            )
            critic_only_drift = float(
                getattr(warm_result, "actor_max_abs_drift_critic_only")
            )
            if critic_only_drift != 0.0:
                raise RuntimeError(
                    "Universal critic-only warm start changed actor parameters"
                )
            critic_digest = content_digest(
                {
                    "actor_max_abs_drift_critic_only": critic_only_drift,
                    "actor_max_abs_drift_joint": float(
                        getattr(warm_result, "actor_max_abs_drift_joint")
                    ),
                    "behavior_cloning_digest": bc_digest,
                    "member_seed": member_seed,
                    "plan": {
                        "critic_only_steps": critic_only_steps,
                        "joint_actor_learning_rate_scale": (
                            config.behavior_cloning_joint_warm_start_actor_lr_scale
                        ),
                        "joint_fine_tune_steps": joint_steps,
                        "learning_rate": (
                            config.behavior_cloning_critic_warm_start_learning_rate
                        ),
                    },
                    "schema_version": "universal_critic_warm_start_evidence_v1",
                    "teacher_artifact_digest": bundle.teacher_artifact.artifact_digest,
                }
            )
        write_policy_stage_snapshot(
            policy,
            output_root=output_root,
            stage="behavior_cloning_critic",
            member_seed=member_seed,
        )

        return {
            "schema_version": "universal_pretraining_evidence_v1",
            "passed": True,
            "teacher_artifact_digest": bundle.teacher_artifact.artifact_digest,
            "behavior_cloning_digest": bc_digest,
            "behavior_cloning_result_artifact_digest": result_artifact_digest,
            "critic_warm_start_digest": critic_digest,
            "behavior_cloning_relative_improvement": relative_improvement,
            "behavior_cloning_holdout_digest": holdout_digest,
            "behavior_cloning_gate_digest": gate_digest,
            "train_symbols": bundle.train_symbols,
            "train_sample_count": int(bundle.split.train_indices.size),
            "validation_sample_count": int(bundle.split.validation_indices.size),
            "purged_sample_count": int(bundle.split.purged_indices.size),
        }

    return hook


__all__ = [
    "SymbolTeacherInput",
    "UniversalPretrainingBundle",
    "build_universal_pretraining_hook",
    "combine_symbol_teachers",
]
