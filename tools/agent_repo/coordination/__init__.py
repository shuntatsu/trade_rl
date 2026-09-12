"""Agent Coordination Plane primitives.

This package is repository tooling. Production ``trade_rl`` modules must not depend on it.
"""

from tools.agent_repo.coordination.model import (
    Capability,
    DependencyKind,
    EvidenceKind,
    EvidenceRecord,
    EvidenceResult,
    ExecutionMode,
    RiskLevel,
    TaskCondition,
    TaskDependency,
    TaskPacket,
    TaskPhase,
    TaskStatus,
    WriteScope,
)
from tools.agent_repo.coordination.state import evidence_is_current, validate_transition

__all__ = [
    "Capability",
    "DependencyKind",
    "EvidenceKind",
    "EvidenceRecord",
    "EvidenceResult",
    "ExecutionMode",
    "RiskLevel",
    "TaskCondition",
    "TaskDependency",
    "TaskPacket",
    "TaskPhase",
    "TaskStatus",
    "WriteScope",
    "evidence_is_current",
    "validate_transition",
]
