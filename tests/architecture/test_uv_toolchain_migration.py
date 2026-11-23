"""Fail closed on mixed uv toolchains before re-enabling Dependabot (#1175)."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import ppo_checkpoint_actions as checkpoint

ROOT = Path(__file__).resolve().parents[2]
CURRENT_UV = "0.13.9"


def test_repository_requires_the_current_compatible_uv() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["uv"]["required-version"] == f"=={CURRENT_UV}"


def test_every_setup_uv_invocation_pins_the_current_release() -> None:
    total = 0
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(
            r"(?m)^\s+uses: astral-sh/setup-uv@[0-9a-f]{40}\n",
            source,
        ):
            following = source[match.end() :]
            following = following.split("\n      - name:", 1)[0]
            following = following.split("\n      - uses:", 1)[0]
            assert re.search(rf'(?m)^\s+version: "{CURRENT_UV}"$', following), path
            total += 1
    assert total >= 15  # all CI and manual/research workflows


@pytest.mark.parametrize(
    ("reported", "allowed"),
    [
        ("0.13.9", True),
        ("0.13.9 (x86_64-unknown-linux-gnu)", True),
        ("0.10.0", False),
        ("0.13.7", False),
        ("0.13.90", False),
    ],
)
def test_checkpoint_runtime_provenance_preserves_exact_version_lineage(
    reported: str, allowed: bool
) -> None:
    completed = subprocess.CompletedProcess(
        args=["uv", "--version"], returncode=0, stdout=f"uv {reported}\n", stderr=""
    )
    with patch.object(checkpoint.subprocess, "run", return_value=completed) as call:
        if allowed:
            assert checkpoint._runtime_provenance()["uv_version"] == CURRENT_UV
        else:
            with pytest.raises(ValueError, match="actual uv executable"):
                checkpoint._runtime_provenance()
        assert call.call_args.args[0] == ["uv", "--version"]


@pytest.mark.parametrize(
    "workflow",
    ["ppo-feature-checkpoint.yml", "ppo-checkpoint-runtime-preflight.yml"],
)
def test_manual_checkpoint_version_guard_accepts_platform_suffix(
    workflow: str,
) -> None:
    source = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    assert "grep -Eq '^uv 0\\.13\\.9([[:space:]]|$)'" in source
    assert "grep -Fx 'uv 0.10.0'" not in source
