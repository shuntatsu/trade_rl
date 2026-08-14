from __future__ import annotations

import tomllib
from typing import Any

from tests.architecture.repository_paths import REPOSITORY_ROOT


def import_linter_config() -> dict[str, Any]:
    payload = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return dict(payload["tool"]["importlinter"])


def import_linter_contract(contract_id: str) -> dict[str, Any]:
    contracts = import_linter_config()["contracts"]
    matches = [
        dict(contract)
        for contract in contracts
        if isinstance(contract, dict) and contract.get("id") == contract_id
    ]
    assert len(matches) == 1
    return matches[0]


def configured_layers() -> tuple[str, ...]:
    return tuple(str(value) for value in import_linter_contract("layers")["layers"])
