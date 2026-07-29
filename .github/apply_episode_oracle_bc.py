from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "trade_rl/rl/environment.py",
    '''        return hard_constrained.weights, peak

    def _make_initial_book(
''',
    '''        return hard_constrained.weights, peak

    def initial_weights_for_reset(self, mode: str, start: int) -> np.ndarray:
        """Return deterministic reset weights for episode-aligned teachers."""

        if mode not in {"cash", "baseline"}:
            raise ValueError(
                "episode teacher initial weights support only cash and baseline"
            )
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < self.minimum_start_index
            or start >= self.dataset.n_bars
        ):
            raise ValueError("episode teacher reset start is outside the dataset")
        weights, _ = self._initial_weights(mode=mode, start=start)
        resolved = np.asarray(weights, dtype=np.float64).copy(order="C")
        resolved.setflags(write=False)
        return resolved

    def _make_initial_book(
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''from trade_rl.learning.hierarchical_teacher_labels import (
    HierarchicalTeacherLabels,
    build_hierarchical_teacher_labels,
)
''',
    '''from trade_rl.learning.episode_behavior_cloning import (
    BehaviorCloningSplit,
    align_behavior_cloning_validation,
)
from trade_rl.learning.episode_oracle_bc import (
    EpisodeBehaviorCloningHoldoutEvaluation,
    evaluate_episode_behavior_cloning_holdout,
    oracle_episode_sampling_config,
    resolve_episode_initial_weights,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeSamplingConfig,
    build_episode_oracle_batch,
)
from trade_rl.learning.episode_teacher_artifact import (
    EPISODE_TEACHER_ARTIFACT_SCHEMA,
    EpisodeSupervisedPolicyDataset,
    collect_episode_teacher_rollout,
    load_episode_teacher_artifact,
    write_episode_teacher_artifact,
)
from trade_rl.learning.hierarchical_teacher_labels import (
    HierarchicalTeacherLabels,
    build_hierarchical_teacher_labels,
)
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''def _configure_torch_cuda_runtime(
''',
    '''def _oracle_episode_sampling_config(
    environment: Any,
    *,
    train_range: tuple[int, int],
    seed: int,
) -> OracleEpisodeSamplingConfig:
    return oracle_episode_sampling_config(
        environment,
        train_range=train_range,
        seed=seed,
    )


def _configure_torch_cuda_runtime(
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''        self._oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._trend_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._teacher_dataset_cache: dict[
            tuple[str, int, int, str, str, str], SupervisedPolicyDataset
        ] = {}

    def _oracle_targets(
''',
    '''        self._oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._oracle_episode_batch_cache: dict[
            tuple[str, int, int, str, str], EpisodeOracleBatch
        ] = {}
        self._trend_target_cache: dict[tuple[str, int, int, str], np.ndarray] = {}
        self._teacher_dataset_cache: dict[
            tuple[str, int, int, str, str, str], SupervisedPolicyDataset
        ] = {}
        self._episode_teacher_dataset_cache: dict[
            tuple[str, int, int, str, str, str], EpisodeSupervisedPolicyDataset
        ] = {}

    def _oracle_episode_batch(
        self,
        environment: Any,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
        sampling_config: OracleEpisodeSamplingConfig,
    ) -> EpisodeOracleBatch:
        dataset = environment.dataset
        dataset_id = getattr(dataset, "dataset_id", None)
        if not isinstance(dataset_id, str):
            raise ValueError("Oracle episode dataset must expose dataset_id")
        start, stop = train_range
        key = (
            dataset_id,
            int(start),
            int(stop),
            teacher_config.digest,
            sampling_config.digest,
        )
        cached = self._oracle_episode_batch_cache.get(key)
        if cached is not None:
            return cached
        batch = build_episode_oracle_batch(
            dataset,
            minimum_start_index=start,
            sampling_config=sampling_config,
            teacher_config=teacher_config,
            initial_weight_provider=lambda mode, index: resolve_episode_initial_weights(
                environment,
                mode,
                index,
            ),
        )
        self._oracle_episode_batch_cache[key] = batch
        return batch

    def _episode_teacher_dataset(
        self,
        environment: Any,
        batch: EpisodeOracleBatch,
        *,
        train_range: tuple[int, int],
        teacher_config: OracleTeacherConfig,
    ) -> EpisodeSupervisedPolicyDataset:
        start, stop = train_range
        environment_digest = getattr(environment, "environment_digest", None)
        action_spec_digest = getattr(environment, "action_spec_digest", None)
        if not isinstance(environment_digest, str):
            raise ValueError("episode teacher environment must expose environment_digest")
        if not isinstance(action_spec_digest, str):
            raise ValueError("episode teacher environment must expose action_spec_digest")
        teacher_identity = content_digest(
            {
                "episode_batch_digest": batch.digest,
                "schema_version": EPISODE_TEACHER_ARTIFACT_SCHEMA,
                "teacher_config_digest": teacher_config.digest,
            }
        )
        key = (
            batch.dataset_id,
            int(start),
            int(stop),
            environment_digest,
            action_spec_digest,
            teacher_identity,
        )
        cached = self._episode_teacher_dataset_cache.get(key)
        if cached is not None:
            return cached
        cache_path: Path | None = None
        if self.teacher_cache_root is not None:
            cache_path = self.teacher_cache_root / _teacher_cache_key(
                dataset_id=batch.dataset_id,
                train_range=(start, stop),
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_identity,
            )
            if cache_path.exists():
                _, teacher_dataset = load_episode_teacher_artifact(
                    cache_path,
                    expected_dataset_id=batch.dataset_id,
                    expected_environment_digest=environment_digest,
                    expected_action_spec_digest=action_spec_digest,
                )
                if teacher_dataset.teacher_config_digest != teacher_identity:
                    raise ValueError("cached episode teacher identity mismatch")
                self._episode_teacher_dataset_cache[key] = teacher_dataset
                return teacher_dataset
        teacher_dataset = collect_episode_teacher_rollout(
            environment,
            batch,
            teacher_config_digest=teacher_identity,
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{cache_path.name}.", dir=str(cache_path.parent))
            )
            try:
                write_episode_teacher_artifact(temporary, teacher_dataset)
                try:
                    temporary.replace(cache_path)
                except FileExistsError:
                    pass
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        self._episode_teacher_dataset_cache[key] = teacher_dataset
        return teacher_dataset

    def _oracle_targets(
''',
)

old_teacher_block = '''                    teacher_kind = config.behavior_cloning_teacher
                    if teacher_kind == "oracle":
                        risk_config = unwrapped_teacher.pre_trade_risk.config
                        teacher_config: Any = OracleTeacherConfig(
                            execution_cost=unwrapped_teacher.config.execution_cost,
                            portfolio_risk=unwrapped_teacher.portfolio_risk.config,
                            max_gross=risk_config.max_gross,
                            max_abs_weight=risk_config.max_abs_weight,
                            entry_threshold=risk_config.entry_threshold,
                            exit_threshold=risk_config.exit_threshold,
                            no_trade_band=risk_config.no_trade_band,
                            reference_portfolio_value=(
                                unwrapped_teacher.initial_capital
                            ),
                            signal_delay_decisions=(
                                unwrapped_teacher.config.signal_delay_decisions
                            ),
                        )
                        targets = self._oracle_targets(
                            dataset, train_range, teacher_config
                        )
                    else:
                        trend_strategy = getattr(
                            unwrapped_teacher, "trend_strategy", None
                        )
                        if trend_strategy is None or not callable(
                            getattr(trend_strategy, "targets", None)
                        ):
                            raise ValueError(
                                "trend behavior cloning requires a trend strategy"
                            )
                        teacher_config = _TeacherIdentity(
                            digest=content_digest(
                                {
                                    "schema_version": (
                                        "causal_trend_baseline_teacher_v1"
                                    ),
                                    "signal_delay_decisions": (
                                        unwrapped_teacher.config.signal_delay_decisions
                                    ),
                                    "trend": trend_strategy.config,
                                }
                            )
                        )
                        targets = self._trend_baseline_targets(
                            dataset,
                            train_range,
                            trend_strategy,
                            teacher_digest=teacher_config.digest,
                        )
                    teacher_dataset = self._teacher_dataset(
                        teacher_environment,
                        targets,
                        dataset_id=dataset.dataset_id,
                        train_range=train_range,
                        teacher_config=teacher_config,
                    )
                    teacher_digest = write_teacher_artifact(
                        output_path.parent / "teacher",
                        teacher_dataset,
                    )
'''
new_teacher_block = '''                    teacher_kind = config.behavior_cloning_teacher
                    episode_batch: EpisodeOracleBatch | None = None
                    episode_split: BehaviorCloningSplit | None = None
                    if teacher_kind == "oracle":
                        risk_config = unwrapped_teacher.pre_trade_risk.config
                        teacher_config: Any = OracleTeacherConfig(
                            execution_cost=unwrapped_teacher.config.execution_cost,
                            portfolio_risk=unwrapped_teacher.portfolio_risk.config,
                            max_gross=risk_config.max_gross,
                            max_abs_weight=risk_config.max_abs_weight,
                            entry_threshold=risk_config.entry_threshold,
                            exit_threshold=risk_config.exit_threshold,
                            no_trade_band=risk_config.no_trade_band,
                            reference_portfolio_value=(
                                unwrapped_teacher.initial_capital
                            ),
                            signal_delay_decisions=(
                                unwrapped_teacher.config.signal_delay_decisions
                            ),
                        )
                        sampling_config = _oracle_episode_sampling_config(
                            unwrapped_teacher,
                            train_range=train_range,
                            seed=seed,
                        )
                        episode_batch = self._oracle_episode_batch(
                            unwrapped_teacher,
                            train_range,
                            teacher_config,
                            sampling_config,
                        )
                        targets = np.concatenate(episode_batch.targets, axis=0)
                        teacher_dataset = self._episode_teacher_dataset(
                            teacher_environment,
                            episode_batch,
                            train_range=train_range,
                            teacher_config=teacher_config,
                        )
                        teacher_digest = write_episode_teacher_artifact(
                            output_path.parent / "teacher",
                            teacher_dataset,
                        )
                    else:
                        trend_strategy = getattr(
                            unwrapped_teacher, "trend_strategy", None
                        )
                        if trend_strategy is None or not callable(
                            getattr(trend_strategy, "targets", None)
                        ):
                            raise ValueError(
                                "trend behavior cloning requires a trend strategy"
                            )
                        teacher_config = _TeacherIdentity(
                            digest=content_digest(
                                {
                                    "schema_version": (
                                        "causal_trend_baseline_teacher_v1"
                                    ),
                                    "signal_delay_decisions": (
                                        unwrapped_teacher.config.signal_delay_decisions
                                    ),
                                    "trend": trend_strategy.config,
                                }
                            )
                        )
                        targets = self._trend_baseline_targets(
                            dataset,
                            train_range,
                            trend_strategy,
                            teacher_digest=teacher_config.digest,
                        )
                        teacher_dataset = self._teacher_dataset(
                            teacher_environment,
                            targets,
                            dataset_id=dataset.dataset_id,
                            train_range=train_range,
                            teacher_config=teacher_config,
                        )
                        teacher_digest = write_teacher_artifact(
                            output_path.parent / "teacher",
                            teacher_dataset,
                        )
'''
replace_once("trade_rl/integrations/sb3_training.py", old_teacher_block, new_teacher_block)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''                    cloning = pretrain_policy(
                        model.policy,
                        teacher_dataset,
                        config=cloning_config,
''',
    '''                    if episode_batch is not None:
                        cloning_config, episode_split = align_behavior_cloning_validation(
                            cloning_config,
                            teacher_dataset,
                        )
                    cloning = pretrain_policy(
                        model.policy,
                        teacher_dataset,
                        config=cloning_config,
''',
)

start_marker = '''                    oracle_audit_payload: dict[str, object] | None = None
                    holdout_evaluation: BehaviorCloningHoldoutEvaluation | None = None
                    if teacher_kind == "oracle":
'''
end_marker = '''                    gate_evaluation: BehaviorCloningGateEvaluation | None = None
'''
sb3_path = ROOT / "trade_rl/integrations/sb3_training.py"
sb3_text = sb3_path.read_text(encoding="utf-8")
start = sb3_text.index(start_marker)
end = sb3_text.index(end_marker, start)
new_audit = '''                    oracle_audit_payload: dict[str, object] | None = None
                    holdout_evaluation: (
                        BehaviorCloningHoldoutEvaluation
                        | EpisodeBehaviorCloningHoldoutEvaluation
                        | None
                    ) = None
                    if teacher_kind == "oracle":
                        if episode_batch is None or episode_split is None:
                            raise RuntimeError("Oracle episode teacher evidence is unavailable")
                        (
                            oracle_audit_payload,
                            holdout_evaluation,
                        ) = evaluate_episode_behavior_cloning_holdout(
                            environment_factory=self.environment_factory,
                            model=model,
                            batch=episode_batch,
                            split=episode_split,
                            output_root=output_path.parent,
                        )
'''
sb3_path.write_text(sb3_text[:start] + new_audit + sb3_text[end:], encoding="utf-8")

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''                        "teacher_kind": teacher_kind,
                        "oracle_reproduction": oracle_audit_payload,
                        "schema_version": "behavior_cloning_run_v5",
''',
    '''                        "teacher_kind": teacher_kind,
                        "episode_sampling": (
                            None
                            if episode_batch is None
                            else {
                                "batch_digest": episode_batch.digest,
                                "decision_count": episode_batch.decision_count,
                                "episode_count": episode_batch.episode_count,
                                "sampling_config_digest": (
                                    episode_batch.sampling_config_digest
                                ),
                            }
                        ),
                        "oracle_reproduction": oracle_audit_payload,
                        "schema_version": "behavior_cloning_run_v6",
''',
)

replace_once(
    "tests/learning/test_episode_teacher_integration.py",
    '''from trade_rl.learning.behavior_cloning import behavior_cloning_split
''',
    '''from trade_rl.learning.episode_behavior_cloning import behavior_cloning_split
''',
)
replace_once(
    "tests/learning/test_episode_teacher_integration.py",
    '''from trade_rl.learning.teacher_artifact import (
    SupervisedPolicyDataset,
    collect_episode_teacher_rollout,
    load_teacher_artifact,
    write_teacher_artifact,
)
''',
    '''from trade_rl.learning.episode_teacher_artifact import (
    EpisodeSupervisedPolicyDataset,
    collect_episode_teacher_rollout,
    load_episode_teacher_artifact,
    write_episode_teacher_artifact,
)
from trade_rl.learning.teacher_artifact import SupervisedPolicyDataset
''',
)
replace_once(
    "tests/learning/test_episode_teacher_integration.py",
    '''    write_teacher_artifact(tmp_path, supervised)
    manifest, loaded = load_teacher_artifact(tmp_path)
''',
    '''    write_episode_teacher_artifact(tmp_path, supervised)
    manifest, loaded = load_episode_teacher_artifact(tmp_path)
''',
)
replace_once(
    "tests/learning/test_episode_teacher_integration.py",
    '''def _supervised_with_episode_ids() -> SupervisedPolicyDataset:
''',
    '''def _supervised_with_episode_ids() -> EpisodeSupervisedPolicyDataset:
''',
)
replace_once(
    "tests/learning/test_episode_teacher_integration.py",
    '''    return SupervisedPolicyDataset(
        observations=np.zeros((sample_count, 2), dtype=np.float32),
''',
    '''    return EpisodeSupervisedPolicyDataset(
        observations=np.zeros((sample_count, 2), dtype=np.float32),
''',
)
replace_once(
    "tests/learning/test_episode_teacher_api_contract.py",
    '''    from trade_rl.learning.behavior_cloning import behavior_cloning_split
    from trade_rl.learning.teacher_artifact import collect_episode_teacher_rollout
''',
    '''    from trade_rl.learning.episode_behavior_cloning import behavior_cloning_split
    from trade_rl.learning.episode_teacher_artifact import (
        collect_episode_teacher_rollout,
    )
''',
)
