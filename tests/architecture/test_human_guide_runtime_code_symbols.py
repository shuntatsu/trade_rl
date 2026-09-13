from __future__ import annotations

import importlib
import json
from pathlib import Path


def _code_symbols_module():
    return importlib.import_module("guide.tools.code_symbols")


def _symbol_names(index: dict[str, object]) -> list[str]:
    raw = index["symbols"]
    assert isinstance(raw, list)
    return [str(entry["qualified_name"]) for entry in raw if isinstance(entry, dict)]


def _write_topic(path: Path, *, symbol: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": path.stem,
                "code_references": [
                    {
                        "id": "ref",
                        "symbol": symbol,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_runtime_index_keeps_only_symbols_referenced_by_topics(tmp_path: Path) -> None:
    code_symbols = _code_symbols_module()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "demo.py").write_text(
        "def keep(value: int) -> int:\n"
        "    return value\n\n"
        "def drop(value: int) -> int:\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    topics = tmp_path / "topics"
    topics.mkdir()
    _write_topic(topics / "one.json", symbol="trade_rl.demo.keep")
    _write_topic(topics / "two.json", symbol="trade_rl.demo.keep")

    index = code_symbols.build_runtime_symbol_index(
        package,
        topics,
        revision="a" * 40,
    )

    assert _symbol_names(index) == ["trade_rl.demo.keep"]
    assert index["source_revision"] == "a" * 40


def test_runtime_index_rejects_unknown_referenced_symbol(tmp_path: Path) -> None:
    code_symbols = _code_symbols_module()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "demo.py").write_text(
        "def known() -> None:\n    pass\n",
        encoding="utf-8",
    )
    topics = tmp_path / "topics"
    topics.mkdir()
    _write_topic(topics / "demo.json", symbol="trade_rl.demo.missing")

    try:
        code_symbols.build_runtime_symbol_index(
            package,
            topics,
            revision="b" * 40,
        )
    except code_symbols.GuideCodeSymbolError as exc:
        assert "missing referenced code symbol" in str(exc)
        assert "trade_rl.demo.missing" in str(exc)
    else:
        raise AssertionError("expected GuideCodeSymbolError for unknown runtime symbol")
