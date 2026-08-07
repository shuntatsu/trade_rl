"""Resolve repository files required by source-checkout-only workflows."""

from __future__ import annotations

from pathlib import Path


def source_checkout_root() -> Path:
    """Return the repository root for the maintained root-package layout."""

    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parent
    expected_package_root = repository_root / "trade_rl"
    if expected_package_root != package_root:
        raise RuntimeError("trade_rl source checkout layout is unavailable")
    if not (repository_root / "pyproject.toml").is_file():
        raise RuntimeError("trade_rl repository root is unavailable")
    return repository_root
