from pathlib import Path
from textwrap import dedent

ci = dedent(
    '''
    name: CI

    on:
      push:
        branches:
          - main
      pull_request:
        branches:
          - main
          - agent/causal-training-hardening

    permissions:
      contents: read

    concurrency:
      group: ci-${{ github.workflow }}-${{ github.ref }}
      cancel-in-progress: true

    jobs:
      quality:
        name: Rebuilt Core
        runs-on: ubuntu-latest
        steps:
          - name: Checkout
            uses: actions/checkout@v4

          - name: Set up uv
            uses: astral-sh/setup-uv@v5
            with:
              python-version: "3.12"
              enable-cache: true

          - name: Install
            run: uv sync --extra dev

          - name: Ruff
            run: uv run ruff check .

          - name: Format
            run: uv run ruff format --check --diff .

          - name: Mypy
            run: uv run mypy trade_rl

          - name: Import architecture
            run: uv run lint-imports

          - name: Dead-code report
            continue-on-error: true
            run: uv run vulture trade_rl tests --min-confidence 100

          - name: Tests and coverage
            run: uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing

          - name: CLI smoke test
            run: uv run trade-rl --version
    '''
).lstrip()
Path(".github/workflows/ci.yml").write_text(ci, encoding="utf-8")
for path in (
    ".github/workflows/implementation-fix.yml",
    "tools/apply_final_hardening.py",
    "tools/fix_f402.py",
    "tools/restore_ci.py",
):
    Path(path).unlink(missing_ok=True)
