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
    "trade_rl/learning/episode_teacher_artifact.py",
    '''    if batch.teacher_config_digest != teacher_config_digest:
        raise ValueError("episode teacher batch configuration identity mismatch")
    flat_observations: list[np.ndarray] = []
''',
    '''    require_sha256(teacher_config_digest, field="teacher_config_digest")
    flat_observations: list[np.ndarray] = []
''',
)

replace_once(
    "trade_rl/learning/episode_behavior_cloning.py",
    '''    episode_ids = np.asarray(dataset.episode_ids, dtype=np.int64).reshape(-1)
''',
    '''    episode_ids = np.asarray(
        getattr(dataset, "episode_ids", np.zeros(sample_count, dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''    BehaviorCloningHoldoutEvaluation,
    OracleTeacherConfig,
    OracleTeacherEvaluation,
    StructuredTeacherObservationProvider,
    SupervisedPolicyDataset,
    collect_teacher_rollout,
    evaluate_action_path,
    evaluate_behavior_cloning_gates,
    evaluate_behavior_cloning_holdout,
''',
    '''    BehaviorCloningHoldoutEvaluation,
    OracleTeacherConfig,
    StructuredTeacherObservationProvider,
    SupervisedPolicyDataset,
    collect_teacher_rollout,
    evaluate_behavior_cloning_gates,
''',
)

replace_once(
    "trade_rl/integrations/sb3_training.py",
    '''def _evaluate_hierarchical_behavior_cloning_gate(
    *,
    cloning: object,
    holdout: BehaviorCloningHoldoutEvaluation | None,
    thresholds: BehaviorCloningGateThresholds,
) -> BehaviorCloningGateEvaluation:
''',
    '''def _evaluate_hierarchical_behavior_cloning_gate(
    *,
    cloning: object,
    holdout: Any,
    thresholds: BehaviorCloningGateThresholds,
) -> BehaviorCloningGateEvaluation:
''',
)
