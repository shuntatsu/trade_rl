from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_legacy_execution_trees_are_absent() -> None:
    for name in ("mars_lite", "legacy_tests", "scripts"):
        assert not (ROOT / name).exists(), f"legacy path still exists: {name}"


def test_only_trade_rl_is_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["project"]["name"] == "trade-rl"
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["trade_rl*"]
    assert config["project"]["scripts"] == {"trade-rl": "trade_rl.cli.app:main"}


def test_source_contains_no_maintained_direct_action_mode() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "trade_rl").rglob("*.py"))
    )

    assert "action_mode" not in source
    assert "direct-action" not in source
    assert "MarsLiteEnv" not in source
