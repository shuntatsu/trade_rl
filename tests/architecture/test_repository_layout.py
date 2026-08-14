"""Repository-root contract independent of the installed ``trade_rl`` namespace."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_repository_uses_flat_responsibility_roots() -> None:
    required = (
        ROOT / "frontend",
        ROOT / "trade_rl",
        ROOT / "scripts",
        ROOT / "tests",
        ROOT / "docs",
        ROOT / "examples",
        ROOT / "docker",
    )
    forbidden = (
        ROOT / "src",
        ROOT / "studio",
        ROOT / "apps",
        ROOT / "tools",
        ROOT / "scripts" / "ci",
        ROOT / "test_support",
    )

    assert all(path.is_dir() for path in required)
    assert all(not path.exists() for path in forbidden)


def test_active_repository_contracts_do_not_reference_src_package() -> None:
    forbidden = "src" + "/trade_rl"
    targets = [
        ROOT / "pyproject.toml",
        ROOT / "docker" / "docker/Dockerfile.training",
        ROOT / "README.md",
        ROOT / "START.md",
    ]
    targets.extend(sorted((ROOT / ".github" / "workflows").glob("*.yml")))
    targets.extend(sorted((ROOT / "scripts").rglob("*.py")))
    targets.extend(sorted((ROOT / "tests").rglob("*.py")))

    violations = [
        path.relative_to(ROOT).as_posix()
        for path in targets
        if path.is_file() and forbidden in path.read_text(encoding="utf-8")
    ]

    assert violations == []
