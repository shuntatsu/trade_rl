"""Immutable data contracts for deterministic machine-generated run reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

RUN_REPORT_STAGE_ORDER = (
    "signal",
    "selection",
    "teacher_admission",
    "teacher_package",
    "behavior_cloning",
    "critic_warm_start",
    "ppo",
    "zero_shot",
    "sealed_evaluation",
)
_RUN_REPORT_SCHEMA = "run_report_v1"


class RunStageStatus(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
    IN_PROGRESS = "IN_PROGRESS"
    NOT_RUN = "NOT_RUN"
    MISSING = "MISSING"
    INVALID = "INVALID"


def _mapping_copy(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class RunStageReport:
    name: str
    status: RunStageStatus
    metrics: Mapping[str, object] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    artifact_digests: Mapping[str, str] = field(default_factory=dict)
    source_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in RUN_REPORT_STAGE_ORDER:
            raise ValueError("run report stage name is unsupported")
        if not isinstance(self.status, RunStageStatus):
            raise ValueError("run report stage status is invalid")
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("run report stage reasons must be non-empty strings")
        if len(set(reasons)) != len(reasons):
            raise ValueError("run report stage reasons must be unique")
        digests = dict(self.artifact_digests)
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for name, digest in digests.items()
        ):
            raise ValueError("run report artifact digests are invalid")
        source_paths = tuple(self.source_paths)
        if any(not isinstance(path, str) or not path for path in source_paths):
            raise ValueError("run report source paths are invalid")
        object.__setattr__(self, "metrics", _mapping_copy(self.metrics))
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "artifact_digests", MappingProxyType(digests))
        object.__setattr__(self, "source_paths", source_paths)

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_digests": dict(self.artifact_digests),
            "metrics": dict(self.metrics),
            "name": self.name,
            "reasons": list(self.reasons),
            "source_paths": list(self.source_paths),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class RunReport:
    root: str
    identities: Mapping[str, object]
    stages: tuple[RunStageReport, ...]
    schema_version: str = _RUN_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root:
            raise ValueError("run report root must be non-empty")
        if self.schema_version != _RUN_REPORT_SCHEMA:
            raise ValueError("unsupported run report schema")
        stages = tuple(self.stages)
        if tuple(stage.name for stage in stages) != RUN_REPORT_STAGE_ORDER:
            raise ValueError("run report stage order is invalid")
        object.__setattr__(self, "identities", _mapping_copy(self.identities))
        object.__setattr__(self, "stages", stages)

    def to_payload(self) -> dict[str, object]:
        return {
            "identities": dict(self.identities),
            "root": self.root,
            "schema_version": self.schema_version,
            "stages": [stage.to_payload() for stage in self.stages],
        }

    def to_json(self) -> str:
        return (
            json.dumps(
                self.to_payload(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


__all__ = [
    "RUN_REPORT_STAGE_ORDER",
    "RunReport",
    "RunStageReport",
    "RunStageStatus",
]
