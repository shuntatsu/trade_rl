"""Repository-root contract independent of the installed ``trade_rl`` namespace."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_repository_uses_explicit_source_app_and_script_roots() -> None:
    required = (
        ROOT / "src" / "trade_rl",
        ROOT / "apps" / "studio-web",
        ROOT / "scripts" / "ci",
    )
    forbidden = (
        ROOT / "trade_rl",
        ROOT / "studio",
        ROOT / "tools",
    )

    assert all(path.is_dir() for path in required)
    assert all(not path.exists() for path in forbidden)
