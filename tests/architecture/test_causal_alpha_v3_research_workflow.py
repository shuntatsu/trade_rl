from __future__ import annotations

from collections.abc import Mapping

import yaml

from tests.architecture.repository_paths import REPOSITORY_ROOT

WORKFLOW = REPOSITORY_ROOT / ".github/workflows/causal-alpha-v3-research.yml"


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, Mapping)
    return {str(key): item for key, item in value.items()}


def _load() -> dict[str, object]:
    payload = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return _mapping(payload)


def test_causal_alpha_v3_research_workflow_is_owner_only_manual_control() -> None:
    workflow = _load()
    triggers = _mapping(workflow["on"])
    assert set(triggers) == {"workflow_dispatch"}
    dispatch = _mapping(triggers["workflow_dispatch"])
    inputs = _mapping(dispatch["inputs"])
    assert set(inputs) == {"operation", "generation"}

    operation = _mapping(inputs["operation"])
    assert operation["required"] == "true"
    assert operation["type"] == "choice"
    assert operation["options"] == ["start", "status", "collect", "stop"]
    generation = _mapping(inputs["generation"])
    assert generation["required"] == "true"
    assert generation["type"] == "string"

    assert _mapping(workflow["permissions"]) == {"contents": "read"}
    concurrency = _mapping(workflow["concurrency"])
    assert concurrency["group"] == "causal-alpha-v3-control"
    assert concurrency["cancel-in-progress"] == "false"

    jobs = _mapping(workflow["jobs"])
    assert set(jobs) == {"control"}
    job = _mapping(jobs["control"])
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "gpu", "nvidia"]
    assert job["environment"] == "gpu-full-training"
    condition = str(job["if"])
    assert "github.actor == github.repository_owner" in condition
    assert "github.ref == 'refs/heads/main'" in condition


def test_causal_alpha_v3_research_workflow_uses_exact_checkout_and_trusted_roots() -> (
    None
):
    workflow = _load()
    job = _mapping(_mapping(workflow["jobs"])["control"])
    environment = _mapping(job["env"])
    assert environment["TRADE_RL_UNIVERSAL_ARTIFACT_ROOT"] == (
        "${{ vars.TRADE_RL_UNIVERSAL_ARTIFACT_ROOT }}"
    )
    assert environment["TRADE_RL_CAUSAL_ALPHA_V3_STATE_ROOT"] == (
        "${{ vars.TRADE_RL_CAUSAL_ALPHA_V3_STATE_ROOT }}"
    )
    assert environment["TRADE_RL_CAUSAL_ALPHA_V3_OPERATION"] == (
        "${{ inputs.operation }}"
    )
    assert environment["TRADE_RL_CAUSAL_ALPHA_V3_GENERATION"] == (
        "${{ inputs.generation }}"
    )

    steps = [_mapping(item) for item in job["steps"]]
    checkout = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    assert checkout["uses"] == (
        "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5"
    )
    checkout_with = _mapping(checkout["with"])
    assert checkout_with["ref"] == "${{ github.sha }}"
    assert checkout_with["persist-credentials"] == "false"

    setup_uv = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    )
    assert setup_uv["uses"] == (
        "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"
    )

    run_steps = "\n".join(str(step.get("run", "")) for step in steps)
    assert "scripts/control_causal_alpha_v3_research_generation.py" in run_steps
    assert '--operation "$TRADE_RL_CAUSAL_ALPHA_V3_OPERATION"' in run_steps
    assert '--generation "$TRADE_RL_CAUSAL_ALPHA_V3_GENERATION"' in run_steps
    assert "${{ inputs." not in run_steps
    assert "--runtime-artifact-root" not in run_steps
    assert "--compose-file" not in run_steps


def test_causal_alpha_v3_research_workflow_retains_control_evidence_on_failure() -> (
    None
):
    workflow = _load()
    job = _mapping(_mapping(workflow["jobs"])["control"])
    steps = [_mapping(item) for item in job["steps"]]

    upload = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    )
    assert upload["uses"] == (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
    )
    assert upload["if"] == "always()"
    upload_with = _mapping(upload["with"])
    assert upload_with["name"] == "causal-alpha-v3-control-evidence"
    assert "control-result.json" in str(upload_with["path"])
    assert "retained" in str(upload_with["path"])
    assert upload_with["if-no-files-found"] == "warn"

    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request:" not in text
    assert "pull_request_target:" not in text
    assert "secrets." not in text
    for forbidden in (
        "runtime_manifest",
        "runtime-artifact-root",
        "frozen-metadata-root",
        "output-root",
        "compose-file",
        "research-config",
        "run-config",
    ):
        assert forbidden not in str(
            _mapping(_mapping(workflow["on"])["workflow_dispatch"])["inputs"]
        )
