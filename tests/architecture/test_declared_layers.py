from __future__ import annotations

from tests.architecture.import_linter_config import (
    configured_layers,
    import_linter_contract,
)


def test_required_import_contracts_are_declared() -> None:
    layers = configured_layers()
    assert "trade_rl.learning" in layers
    assert "trade_rl.release" in layers
    for contract_id in (
        "release",
        "learning-frameworks",
        "workflow-frameworks",
        "training-core",
    ):
        assert import_linter_contract(contract_id)["type"] == "forbidden"
