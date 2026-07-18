from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / ".github" / "check_workflow_security.py"


def _validate(root: Path) -> tuple[str, ...]:
    namespace = runpy.run_path(str(CHECKER))
    return tuple(namespace["validate_workflow_security"](root))


def test_repository_workflows_satisfy_privileged_runner_policy() -> None:
    assert _validate(ROOT) == ()


def test_policy_rejects_pull_request_controlled_self_hosted_runner(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / ".github" / "workflows" / "unsafe.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        """name: unsafe
on: [pull_request]
jobs:
  unsafe:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
""",
        encoding="utf-8",
    )

    errors = _validate(tmp_path)

    assert any("pull_request" in error and "self-hosted" in error for error in errors)
    assert any("mutable action reference" in error for error in errors)


def test_policy_requires_main_owner_environment_for_privileged_dispatch(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / ".github" / "workflows" / "unsafe.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(
        """name: unsafe
on:
  workflow_dispatch:
jobs:
  unsafe:
    runs-on: [self-hosted, gpu]
    steps:
      - run: nvidia-smi
""",
        encoding="utf-8",
    )

    errors = _validate(tmp_path)

    assert any("gpu-full-training" in error for error in errors)
    assert any("repository_owner" in error for error in errors)
    assert any("refs/heads/main" in error for error in errors)
