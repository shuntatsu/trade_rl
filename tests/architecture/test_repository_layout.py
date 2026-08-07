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
