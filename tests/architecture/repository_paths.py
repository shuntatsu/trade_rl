"""Canonical repository paths for architecture tests.

The import namespace and physical Python package both live at ``trade_rl/``.
Architecture tests that inspect repository files depend on these paths instead
of repeating repository layout knowledge.
"""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYTHON_SOURCE_ROOT = REPOSITORY_ROOT / "trade_rl"
STUDIO_WEB_ROOT = REPOSITORY_ROOT / "frontend"
CI_SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"

__all__ = [
    "CI_SCRIPTS_ROOT",
    "PYTHON_SOURCE_ROOT",
    "REPOSITORY_ROOT",
    "STUDIO_WEB_ROOT",
]
