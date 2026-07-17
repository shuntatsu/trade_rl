from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_import_contract_declares_learning_and_release_layers() -> None:
    content = (ROOT / ".importlinter").read_text(encoding="utf-8")

    assert "    trade_rl.learning\n" in content
    assert "    trade_rl.release\n" in content
    assert "[importlinter:contract:release]" in content
    assert "[importlinter:contract:learning-frameworks]" in content
