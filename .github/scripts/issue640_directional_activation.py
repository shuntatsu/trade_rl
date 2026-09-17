"""Result-blind global activation contract for Issue #640 directional execution."""

from __future__ import annotations

from collections.abc import Mapping

from trade_rl.artifacts import canonical_json_bytes, content_digest

SCHEMA_VERSION = "issue640_directional_activation_v1"
ISSUE_NUMBER = 640
ACTIVATION_ARTIFACT_NAME = "issue640-directional-activation-v1"


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _text(value: object, *, field: str, length: int | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if length is not None and len(value) != length:
        raise ValueError(f"{field} must be exactly {length} characters")
    return value


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return dict(value)


def _runtime_packages(protocol: Mapping[str, object]) -> dict[str, str]:
    raw = _mapping(protocol.get("required_runtime_packages"), field="runtime packages")
    required = ("lightgbm", "stable-baselines3", "torch", "scikit-learn")
    if set(raw) != set(required):
        raise ValueError("runtime package roster differs from frozen directional study")
    result: dict[str, str] = {}
    for name in required:
        version = raw[name]
        if not isinstance(version, str) or not version:
            raise ValueError("runtime packages must all have concrete versions")
        result[name] = version
    return result


def build_activation_claim(
    protocol: Mapping[str, object],
    *,
    workflow_run_id: int,
    workflow_run_attempt: int,
    implementation_head: str,
) -> dict[str, object]:
    """Build the only pre-economic activation claim for the frozen Issue #640 run."""

    run_id = _positive_int(workflow_run_id, field="workflow_run_id")
    if workflow_run_attempt != 1 or isinstance(workflow_run_attempt, bool):
        raise ValueError("workflow_run_attempt must be exactly 1")
    head = _text(implementation_head, field="implementation_head", length=40)
    if any(char not in "0123456789abcdef" for char in head):
        raise ValueError(
            "implementation_head must be a lowercase hexadecimal commit SHA"
        )

    protocol_dict = _mapping(protocol, field="protocol")
    runtime = _runtime_packages(protocol_dict)
    provenance = _mapping(protocol_dict.get("provenance"), field="protocol provenance")
    implementation_digest = _text(
        provenance.get("implementation_digest"),
        field="implementation_digest",
        length=64,
    )
    runtime_environment_digest = _text(
        provenance.get("runtime_environment_digest"),
        field="runtime_environment_digest",
        length=64,
    )
    arms = protocol_dict.get("arms")
    if (
        not isinstance(arms, list)
        or not arms
        or any(not isinstance(item, str) or not item for item in arms)
    ):
        raise ValueError("protocol arm roster must be a non-empty string array")

    claim: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "issue_number": ISSUE_NUMBER,
        "workflow_run_id": run_id,
        "workflow_run_attempt": 1,
        "implementation_head": head,
        "protocol_digest": content_digest(protocol_dict),
        "implementation_digest": implementation_digest,
        "runtime_environment_digest": runtime_environment_digest,
        "source_dataset_id": _text(
            protocol_dict.get("source_dataset_id"), field="source_dataset_id"
        ),
        "source_artifact_digest": _text(
            protocol_dict.get("source_artifact_digest"), field="source_artifact_digest"
        ),
        "source_study_digest": _text(
            protocol_dict.get("source_study_digest"), field="source_study_digest"
        ),
        "source_plan_sha256": _text(
            protocol_dict.get("source_plan_sha256"), field="source_plan_sha256"
        ),
        "required_runtime_packages": runtime,
        "arm_roster": list(arms),
        "local_output_root_is_authority": False,
        "economic_execution_started": False,
        "economic_result_inspected": False,
        "unused_data_accessed": False,
        "production_eligible": False,
    }
    return claim


def activation_claim_bytes(claim: Mapping[str, object]) -> bytes:
    """Return canonical bytes for one activation claim."""

    return canonical_json_bytes(dict(claim))


def validate_remote_slot(
    payload: Mapping[str, object],
    *,
    state: str,
    workflow_run_id: int | None = None,
) -> dict[str, object] | None:
    """Validate the repository-global activation Artifact slot."""

    data = _mapping(payload, field="artifact listing")
    total_count = data.get("total_count")
    artifacts = data.get("artifacts")
    if (
        isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or total_count < 0
    ):
        raise ValueError("artifact total_count must be a non-negative integer")
    if not isinstance(artifacts, list) or any(
        not isinstance(item, Mapping) for item in artifacts
    ):
        raise ValueError("artifact listing must contain an artifact array")
    if total_count != len(artifacts):
        raise ValueError("artifact total_count differs from artifact array length")

    matching = [
        dict(item) for item in artifacts if item.get("name") == ACTIVATION_ARTIFACT_NAME
    ]
    if state == "empty":
        if matching:
            raise RuntimeError("directional activation slot is already claimed")
        return None
    if state != "claimed":
        raise ValueError("state must be empty or claimed")
    if len(matching) != 1:
        raise RuntimeError("directional activation slot must contain exactly one claim")

    artifact = matching[0]
    if artifact.get("expired") is not False:
        raise RuntimeError(
            "directional activation Artifact must be present and unexpired"
        )
    if workflow_run_id is None:
        raise ValueError(
            "workflow_run_id is required for claimed activation validation"
        )
    expected_run = _positive_int(workflow_run_id, field="workflow_run_id")
    run = _mapping(artifact.get("workflow_run"), field="activation artifact workflow")
    if run.get("id") != expected_run:
        raise RuntimeError(
            "directional activation Artifact belongs to another workflow"
        )
    return artifact


def validate_activation_claim(
    claim: Mapping[str, object],
    protocol: Mapping[str, object],
    *,
    workflow_run_id: int,
    implementation_head: str,
) -> None:
    """Require an exact canonical claim for this workflow/protocol/head."""

    actual = _mapping(claim, field="activation claim")
    expected = build_activation_claim(
        protocol,
        workflow_run_id=workflow_run_id,
        workflow_run_attempt=1,
        implementation_head=implementation_head,
    )
    if set(actual) != set(expected):
        raise ValueError("activation claim keys differ from protocol authority")
    for field, value in expected.items():
        if type(actual[field]) is not type(value) or actual[field] != value:
            raise ValueError(
                f"activation claim {field} differs from protocol authority"
            )
    if activation_claim_bytes(actual) != activation_claim_bytes(expected):
        raise ValueError(
            "activation claim canonical bytes differ from protocol authority"
        )


__all__ = [
    "ACTIVATION_ARTIFACT_NAME",
    "SCHEMA_VERSION",
    "activation_claim_bytes",
    "build_activation_claim",
    "validate_activation_claim",
    "validate_remote_slot",
]
