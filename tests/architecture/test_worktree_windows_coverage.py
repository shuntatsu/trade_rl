from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/m1-optional-platform-coverage.yml"
)


def test_windows_worktree_coverage_retriggers() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    for path in (
        "tools/worktree_hygiene.py",
        "tests/tools/test_worktree_hygiene.py",
        "tests/architecture/test_worktree_windows_coverage.py",
    ):
        assert source.count(f"      - {path}\n") == 2


def test_windows_worktree_cases_execute_with_exact_count() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    job = source.split("  windows-job-object:\n", 1)[1].split("  forecast-gbm:\n", 1)[0]
    assert "runs-on: windows-latest" in job
    execute = "      - name: Execute native Windows worktree safety tests\n"
    require = "      - name: Require native Windows worktree cases without skips\n"
    assert execute in job and require in job
    command = job.split(execute, 1)[1].split(require, 1)[0]
    assert "uv run --no-sync python -m pytest -q -rs" in command
    assert '--junitxml="${{ runner.temp }}/m1-windows-worktree.xml"' in command
    assert command.count("tests/tools/test_worktree_hygiene.py::") == 2
    for case in (
        "test_open_handle_blocks_removal_on_windows",
        "test_windows_case_and_slash_variants_are_equivalent",
    ):
        assert f"tests/tools/test_worktree_hygiene.py::{case}" in command
    assert (
        "run: uv run --no-sync python -m tools.ci_exact_pytest_counts "
        "m1-windows-worktree.xml 2"
        in job.split(require, 1)[1].split("      - name:", 1)[0]
    )
