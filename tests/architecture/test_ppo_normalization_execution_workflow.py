import re
from pathlib import Path

import pytest

WORKFLOW = Path(".github/workflows/ppo-normalization-execution.yml")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


# The economic one-shot has a separate, exact 3.12.12 interpreter contract.
# Unlike generic CI, stages must not silently select an updated toolcache patch.
_PINNED_UV_SETUP = (
    "      - name: Set up uv\n"
    "        uses: astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2\n"
    "        with:\n"
    '          python-version: "3.12.12"\n'
    "          enable-cache: true\n"
    "          working-directory: target\n"
)
_PINNED_INSTALL = (
    "      - name: Install exact PPO runtime\n"
    "        working-directory: target\n"
    "        run: uv sync --extra dev --extra train-sb3\n"
)
_ECONOMIC_STAGES = ("execution", "verify", "finalize")


def _economic_job(text: str, stage: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(stage)}:\n(.*?)(?=^  (?:request|execution|verify|finalize):\n|\Z)",
        text,
    )
    assert match is not None
    return match.group(1)


def _require_pinned_one_shot_runtime(job: str) -> None:
    assert job.count(_PINNED_UV_SETUP) == 1
    assert job.count(_PINNED_INSTALL) == 1
    assert job.index(_PINNED_UV_SETUP) < job.index(_PINNED_INSTALL)
    assert job.count('python-version: "3.12.12"') == 1
    assert job.count("target/.venv/bin/python") == 1
    assert "ref: ${{ needs.request.outputs.execution_source_sha }}" in job
    assert "ref: ${{ github.workflow_sha }}" in job
    assert "actions/setup-python@" not in job
    assert "UV_PYTHON_PREFERENCE" not in job
    assert "UV_PYTHON_DOWNLOADS" not in job


def test_one_shot_economic_stages_keep_exact_runtime_across_jobs() -> None:
    """All three independently hosted stages retain identical pinned Python."""
    text = _text()
    for stage in _ECONOMIC_STAGES:
        _require_pinned_one_shot_runtime(_economic_job(text, stage))


@pytest.mark.parametrize("stage", _ECONOMIC_STAGES)
@pytest.mark.parametrize("alternative", ("3.12", "3.12.16"))
def test_one_shot_runtime_contract_refuses_minor_only_or_new_patch(
    stage: str, alternative: str
) -> None:
    """A future stage-specific interpreter drift must fail its source oracle."""
    job = _economic_job(_text(), stage)
    mutant = job.replace(
        'python-version: "3.12.12"',
        f'python-version: "{alternative}"',
        1,
    )
    assert mutant != job
    with pytest.raises(AssertionError):
        _require_pinned_one_shot_runtime(mutant)


def test_normalization_execution_workflow_is_comment_triggered_and_globally_serialized() -> (
    None
):
    text = _text()

    assert "issue_comment:" in text
    assert "workflow_dispatch:" not in text
    assert "ppo-normalization-execution-request-v2" in text
    assert "concurrency:" in text
    assert "ppo-normalization-corrected-v1" in text
    assert "cancel-in-progress: false" in text


def test_normalization_execution_workflow_separates_execution_verification_and_reveal() -> (
    None
):
    text = _text()

    assert "name: Execute all ten immutable slots" in text
    assert "name: Verify all ten slots without refit" in text
    assert "name: Publish comparison after verifier authority" in text

    execute = text.index("name: Execute all ten immutable slots")
    upload_execution = text.index("name: Upload complete execution evidence")
    verify = text.index("name: Verify all ten slots without refit")
    upload_verification = text.index("name: Upload complete verification evidence")
    publish = text.index("name: Publish comparison after verifier authority")

    assert execute < upload_execution < verify < upload_verification < publish


def test_normalization_execution_workflow_never_uploads_per_slot_artifacts() -> None:
    text = _text()

    assert "control_raw_seed" not in text
    assert "candidate_normalized_seed" not in text
    assert "Upload complete execution evidence" in text
    assert "Upload complete verification evidence" in text
    assert "Upload final comparison evidence" in text


def test_normalization_execution_workflow_failure_upload_is_non_economic_only() -> None:
    text = _text()

    assert "name: Ensure non-economic execution failure receipt" in text
    assert "name: Upload non-economic execution failure receipt" in text
    assert "if: failure()" in text
    assert "path: output/execution-failure.json" in text
    assert "path: output/execution\n" in text
    execute = text.index("name: Execute all ten immutable slots")
    ensure = text.index("name: Ensure non-economic execution failure receipt")
    failure = text.index("name: Upload non-economic execution failure receipt")
    verify = text.index("name: Verify all ten slots without refit")
    assert execute < ensure < failure < verify


def test_normalization_execution_step_times_out_before_job_hard_limit() -> None:
    text = _text()
    execution_job = text[text.index("  execution:") : text.index("  verify:")]
    execute_step = execution_job[
        execution_job.index("name: Execute all ten immutable slots") :
    ]

    assert "timeout-minutes: 350" in execution_job
    assert "timeout-minutes: 330" in execute_step
    assert "ppo_normalization_actions.py failure-receipt" in execution_job


def test_normalization_execution_workflow_has_trusted_write_boundary() -> None:
    text = _text()

    assert "contents: write" in text
    assert "actions: read" in text
    assert "pull-requests: read" in text
    assert "issues: read" in text
    assert "persist-credentials: false" in text


def test_normalization_execution_workflow_uses_trusted_raw_artifact_redownload() -> (
    None
):
    text = _text()

    assert "actions/download-artifact@" not in text
    assert "EXECUTION_ARTIFACT_ID" in text
    assert "EXECUTION_ARTIFACT_SHA256" in text
    assert "VERIFICATION_ARTIFACT_ID" in text
    assert "VERIFICATION_ARTIFACT_SHA256" in text


def test_normalization_transport_is_type_checked_in_push_and_full_ci() -> None:
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert ci.count("name: Research transport types") == 2
    assert ci.count("tools/ppo_normalization_actions.py") == 2


def test_normalization_execution_workflow_separates_request_and_sealed_source_checkouts() -> (
    None
):
    text = _text()

    assert "ppo-normalization-execution-request-v2" in text
    assert "request_code_sha:" in text
    assert "execution_source_sha:" in text
    assert "path: request" in text
    assert "path: target" in text
    assert "ref: ${{ needs.request.outputs.request_code_sha }}" in text
    assert "ref: ${{ needs.request.outputs.execution_source_sha }}" in text
