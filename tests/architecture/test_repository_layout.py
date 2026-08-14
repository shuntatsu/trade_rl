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
        ROOT / "mars_lite",
        ROOT / "legacy_tests",
    )

    assert all(path.is_dir() for path in required)
    assert all(not path.exists() for path in forbidden)


def test_active_repository_contracts_do_not_reference_src_package() -> None:
    forbidden = "src" + "/trade_rl"
    required_targets = [
        ROOT / "pyproject.toml",
        ROOT / "docker" / "Dockerfile.training",
        ROOT / "README.md",
        ROOT / "START.md",
    ]
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in required_targets
        if not path.is_file()
    ]
    assert missing == []

    targets = [*required_targets]
    targets.extend(sorted((ROOT / ".github" / "workflows").glob("*.yml")))
    targets.extend(sorted((ROOT / "scripts").rglob("*.py")))
    targets.extend(sorted((ROOT / "tests").rglob("*.py")))

    violations = [
        path.relative_to(ROOT).as_posix()
        for path in targets
        if forbidden in path.read_text(encoding="utf-8")
    ]

    assert violations == []
