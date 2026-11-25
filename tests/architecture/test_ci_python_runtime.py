"""The required CI jobs use runner-cached Python instead of fragile release downloads."""

from __future__ import annotations

import re
from pathlib import Path

from tests.architecture.workflow_action_refs import setup_uv_action_matches

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

SETUP_PYTHON = "actions/setup-python@6d4fd22b20465ccc0f7354476588e8df7c9ca0e0"
SETUP_UV = "astral-sh/setup-uv@1db35dd8eee11c5894872e0b383ae6c406cf738d"


def test_all_ci_uv_jobs_preinstall_python_from_runner_toolcache() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert (
        "env:\n"
        "  UV_PYTHON_PREFERENCE: only-system\n"
        "  UV_PYTHON_DOWNLOADS: never\n" in source
    )

    setup_pattern = re.compile(
        r"      - name: Set up runner Python 3\.12\n"
        rf"        uses: {re.escape(SETUP_PYTHON)}\n"
        r"        with:\n"
        r'          python-version: "3\.12"\n'
        r"          check-latest: false\n"
        r"\n"
        r"      - name: Set up uv\n"
        rf"        uses: {re.escape(SETUP_UV)}\n"
    )
    # All Fast Push / Core / PPO / PostgreSQL / software and research reviews
    # must use one toolcache interpreter and never trigger standalone downloads.
    assert len(setup_pattern.findall(source)) == 7
    assert source.count(f"uses: {SETUP_PYTHON}") == 7
    assert source.count(f"uses: {SETUP_UV}") == 7
    assert source.count('python-version: "3.12"') == 7
    assert "UV_PYTHON_DOWNLOADS: automatic" not in source
    assert "UV_PYTHON_DOWNLOADS: never" in source


def test_lean_core_fetches_full_history_for_research_packet_source_binding() -> None:
    """A shallow success can skip the source-closure comparison altogether."""
    source = WORKFLOW.read_text(encoding="utf-8")
    core = source.split("  core:\n", 1)[1].split("\n  ppo-runtime:\n", 1)[0]
    checkout = core.split("- name: Checkout exact head\n", 1)[1].split(
        "- name: Set up runner Python 3.12\n", 1
    )[0]
    assert "fetch-depth: 0" in checkout


def test_ci_keeps_exact_head_and_dependency_install_steps() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert source.count("name: Checkout exact head") >= 7
    assert "uv sync --extra dev" in source
    assert "uv sync --extra dev --extra train-sb3" in source
    assert "uv sync --extra dev --extra store" in source
    assert "uv run --no-sync python -m pytest -q -p tools.ci_test_shard tests" in source


def test_setup_uv_action_commit_is_consistent_across_all_workflows() -> None:
    """A reviewed SHA update must cover *all* 17 Action steps, not just Core CI."""
    workflows = ROOT / ".github" / "workflows"
    refs: list[str] = []
    for path in sorted((*workflows.glob("*.yml"), *workflows.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        refs.extend(match.group("ref") for match in setup_uv_action_matches(text, path))
    assert len(refs) == 17
    assert refs == [SETUP_UV.removeprefix("astral-sh/setup-uv@")] * 17


def test_full_core_shards_preserve_test_coverage_and_skip_reasons() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    shards = source.split("  core-tests:\n", 1)[1].split("\n  core:\n", 1)[0]
    assert "fetch-depth: 0" in shards
    assert "ref: ${{ github.event.pull_request.head.sha }}" in shards
    assert "run: uv sync --locked --extra dev" in shards
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in shards
    assert "SHARD_INDEX: ${{ matrix.shard }}" in shards
    assert "path: ${{ runner.temp }}/core-shard.json" in shards
    assert "fail-fast: false" in shards
    assert "max-parallel: 4" in shards
    assert "shard: [0, 1, 2, 3]" in shards
    assert "continue-on-error" not in shards
    command = shards.split("      - name: Core tests shard\n", 1)[1].split(
        "      - name: Upload completed shard receipt\n", 1
    )[0]
    assert 'PYTEST_ADDOPTS: "-rs"' in command
    assert (
        "uv run --no-sync python -m pytest -q -p tools.ci_test_shard tests" in command
    )
    assert '--ci-shard-index "$SHARD_INDEX" --ci-shard-count 4' in command
    assert '--ci-shard-head "$HEAD_SHA"' in command
    assert "--durations=25" in command
    assert "if-no-files-found: error" in shards
    assert (
        "name: core-test-shard-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.shard }}"
        in shards
    )
    assert "if: always()" not in shards


def test_lean_core_is_a_fail_closed_aggregate_of_the_exact_attempt() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    core = source.split("  core:\n", 1)[1].split("\n  ppo-runtime:\n", 1)[0]
    assert "needs: [core-tests]" in core
    assert "if: always() &&" in core
    assert "SHARD_RESULT: ${{ needs.core-tests.result }}" in core
    assert 'run: test "$SHARD_RESULT" = "success"' in core
    assert "continue-on-error" not in core
    assert (
        "pattern: core-test-shard-${{ github.run_id }}-${{ github.run_attempt }}-*"
        in core
    )
    assert "path: ${{ runner.temp }}/core-shards" in core
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in core
    assert "merge-multiple: false" in core
    assert "uv run --no-sync python -m tools.ci_test_shard" in core
    assert '--directory "$RUNNER_TEMP/core-shards" --count 4 --head "$HEAD_SHA"' in core
    assert (
        core.index("Require all test shards to succeed")
        < core.index("Verify complete Core test collection")
        < core.index("- name: Build")
    )
