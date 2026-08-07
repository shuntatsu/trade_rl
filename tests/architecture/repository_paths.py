"""Canonical repository paths for architecture tests.

Python import namespaces remain ``trade_rl.*`` even though the physical package
lives under ``src/trade_rl``. Architecture tests that inspect repository files
should depend on these paths instead of repeating the physical layout.
"""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "trade_rl"
STUDIO_WEB_ROOT = REPOSITORY_ROOT / "apps" / "studio-web"
CI_SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts" / "ci"

__all__ = [
    "CI_SCRIPTS_ROOT",
    "PYTHON_SOURCE_ROOT",
    "REPOSITORY_ROOT",
    "STUDIO_WEB_ROOT",
]
