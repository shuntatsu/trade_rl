from pathlib import Path

WORKFLOW = Path(".github/workflows/ppo-normalization-execution.yml")


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_normalization_execution_workflow_is_comment_triggered_and_globally_serialized() -> (
    None
):
    text = _text()

    assert "issue_comment:" in text
    assert "workflow_dispatch:" not in text
    assert "ppo-normalization-execution-request-v1" in text
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
