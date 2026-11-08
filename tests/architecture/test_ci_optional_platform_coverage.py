from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "m1-optional-platform-coverage.yml"


def _job(source: str, name: str, next_name: str | None = None) -> str:
    section = source.split(f"  {name}:\n", 1)[1]
    return section.split(f"  {next_name}:\n", 1)[0] if next_name else section


def test_optional_platform_workflow_is_read_only_and_narrowly_triggered() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert "  workflow_dispatch:" in source
    assert "  pull_request:" in source
    assert "  push:" in source
    assert source.count("    paths:") == 2
    assert "m1-optional-platform-coverage.yml" in source
    assert "test_process_containment.py" in source
    assert "test_dev_realdata_phase2_trials.py" in source
    assert "permissions:\n  contents: read\n" in source
    assert "persist-credentials: false" in source
    assert "continue-on-error:" not in source
    head_ref = chr(36) + "{{ github.event.pull_request.head.sha || github.sha }}"
    assert source.count("ref: " + head_ref) == 3
    assert len(re.findall(r"uses: actions/checkout@[0-9a-f]{40}", source)) == 3
    assert len(re.findall(r"uses: astral-sh/setup-uv@[0-9a-f]{40}", source)) == 3
    assert "secrets." not in source


def test_windows_job_object_checks_run_on_real_windows_runner() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    job = _job(source, "windows-job-object", "forecast-gbm")
    assert "runs-on: windows-latest" in job
    assert "uv sync --locked --extra dev" in job
    assert "assert os.name == 'nt'" in job
    assert (
        "tests/tools/tier_d_nonrl/test_process_containment.py"
        "::test_assignment_failure_terminates_and_reaps_suspended_process" in job
    )
    assert (
        "tests/tools/tier_d_nonrl/test_process_containment.py"
        "::test_job_contains_live_child_after_root_exits_and_cleanup_kills_it" in job
    )


def test_forecast_extra_checks_pin_lightgbm_and_execute_both_skipped_cases() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    job = _job(source, "forecast-gbm")
    assert "runs-on: ubuntu-latest" in job
    assert "uv sync --locked --extra dev --extra forecast-gbm" in job
    assert "import lightgbm" in job
    assert (
        "tests/evaluation/test_dev_realdata_phase2_trials.py::test_lightgbm_signal_ignores_the_future"
        in job
    )
    assert (
        "tests/evaluation/test_dev_realdata_phase2_trials.py::test_lightgbm_trial_fits_with_the_fixed_parameters"
        in job
    )


def test_optional_jobs_retrigger_when_dependency_implementation_changes() -> None:
    """Changing the implementation, not just its tests, must rerun native checks."""
    source = WORKFLOW.read_text(encoding="utf-8")
    # Both the PR and main-push path filters must cover these actual imports.
    for implementation_path in (
        "trade_rl/evaluation/dev_realdata_study/**",
        "trade_rl/data/market.py",
        "tools/tier_d_nonrl/tier_d_process_tree.py",
    ):
        assert source.count("      - " + implementation_path + "\n") == 2


def test_full_history_ci_runs_provenance_tests_without_skipping() -> None:
    """Shallow CI must not silently skip the provenance/audit obligations."""
    source = WORKFLOW.read_text(encoding="utf-8")
    job = _job(source, "history-provenance")
    assert "runs-on: ubuntu-latest" in job
    assert "fetch-depth: 0" in job
    assert "git rev-parse --is-shallow-repository" in job
    assert "uv sync --locked --extra dev" in job
    assert (
        "test_forward_study_seal.py::test_the_real_checkout_identity_matches_git_when_history_is_available"
        in job
    )
    for name in (
        "test_the_dependency_walk_covers_the_paper_code_and_excludes_the_market_store",
        "test_launch_commit_runs_packet_bound_code",
        "test_heads_pin_the_code_as_of_the_frozen_packet_revision",
    ):
        assert (
            "tests/evaluation/test_paper_attempt2_assurance_packet.py::" + name in job
        )
    assert "--junitxml=" in job
    assert "skipped" in job
    assert "tests" in job
    assert "== 4" in job
    for path in (
        "tests/evaluation/test_forward_study_seal.py",
        "tests/evaluation/test_paper_attempt2_assurance_packet.py",
    ):
        assert source.count("      - " + path + "\n") == 2
