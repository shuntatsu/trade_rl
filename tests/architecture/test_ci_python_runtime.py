"""The required CI jobs use runner-cached Python instead of fragile release downloads."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

SETUP_PYTHON = "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
SETUP_UV = "astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2"


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
    assert len(setup_pattern.findall(source)) == 6
    assert source.count(f"uses: {SETUP_PYTHON}") == 6
    assert source.count(f"uses: {SETUP_UV}") == 6
    assert source.count('python-version: "3.12"') == 6
    assert "UV_PYTHON_DOWNLOADS: automatic" not in source
    assert "UV_PYTHON_DOWNLOADS: never" in source


def test_ci_keeps_exact_head_and_dependency_install_steps() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    assert source.count("name: Checkout exact head") >= 6
    assert "uv sync --extra dev" in source
    assert "uv sync --extra dev --extra train-sb3" in source
    assert "uv sync --extra dev --extra store" in source
    assert "uv run pytest -q tests" in source
