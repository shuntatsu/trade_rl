"""Supervised pretraining data and oracle-teacher contracts."""

from trade_rl.learning.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningResult,
    pretrain_policy,
)
from trade_rl.learning.oracle_teacher import (
    ORACLE_TEACHER_SCHEMA,
    OracleTeacherConfig,
    oracle_target_path,
)
from trade_rl.learning.teacher_artifact import (
    TEACHER_ARRAYS_NAME,
    TEACHER_ARTIFACT_SCHEMA,
    TEACHER_MANIFEST_NAME,
    SupervisedPolicyDataset,
    TeacherArtifactManifest,
    collect_teacher_rollout,
    load_teacher_artifact,
    write_teacher_artifact,
)

__all__ = [
    "ORACLE_TEACHER_SCHEMA",
    "TEACHER_ARRAYS_NAME",
    "TEACHER_ARTIFACT_SCHEMA",
    "TEACHER_MANIFEST_NAME",
    "OracleTeacherConfig",
    "BehaviorCloningConfig",
    "BehaviorCloningResult",
    "SupervisedPolicyDataset",
    "TeacherArtifactManifest",
    "collect_teacher_rollout",
    "load_teacher_artifact",
    "oracle_target_path",
    "pretrain_policy",
    "write_teacher_artifact",
]
