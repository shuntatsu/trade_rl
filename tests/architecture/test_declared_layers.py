from __future__ import annotations

from tests.architecture.import_linter_config import (
    configured_layers,
    import_linter_contract,
)


def test_import_contract_declares_learning_and_release_layers() -> None:
    layers = configured_layers()
    assert "trade_rl.learning" in layers
    assert "trade_rl.release" in layers
    assert import_linter_contract("release")["type"] == "forbidden"
    assert import_linter_contract("learning-frameworks")["type"] == "forbidden"
