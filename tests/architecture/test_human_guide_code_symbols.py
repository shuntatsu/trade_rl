from importlib import import_module
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _code_symbols():
    return import_module("guide.tools.code_symbols")


def _symbols(index: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = index["symbols"]
    assert isinstance(raw, list)
    return {
        str(entry["qualified_name"]): entry
        for entry in raw
        if isinstance(entry, dict)
    }


def test_index_resolves_replay_risk_execution_and_local_names() -> None:
    code_symbols = _code_symbols()
    index = code_symbols.build_symbol_index(ROOT / "trade_rl", revision="a" * 40)
    symbols = _symbols(index)

    replay = symbols["trade_rl.evaluation.replay.run_single_symbol_replay"]
    assert replay["kind"] == "function"
    assert replay["path"] == "trade_rl/evaluation/replay.py"
    assert "desired_quantity" in replay["local_names"]
    assert "target_weight" in replay["local_names"]

    risk = symbols["trade_rl.risk.pretrade.PreTradeRisk.constrain"]
    assert risk["kind"] == "method"
    assert risk["path"] == "trade_rl/risk/pretrade.py"

    execution = symbols[
        "trade_rl.simulation.execution.MarketExecutor.execute_interval"
    ]
    assert execution["kind"] == "method"
    assert execution["path"] == "trade_rl/simulation/execution.py"


def test_index_parses_module_without_executing_it(tmp_path: Path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "danger.py").write_text(
        "raise RuntimeError('must not execute')\n\n"
        "def safe(value: int) -> int:\n"
        "    result = value + 1\n"
        "    return result\n",
        encoding="utf-8",
    )

    index = code_symbols.build_symbol_index(package, revision="b" * 40)
    symbols = _symbols(index)

    assert "trade_rl.danger.safe" in symbols
    assert symbols["trade_rl.danger.safe"]["local_names"] == ["result", "value"]


def test_nested_scope_names_do_not_leak_into_parent_function(tmp_path: Path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "nested.py").write_text(
        "def outer(value: int) -> int:\n"
        "    result = value\n"
        "    def inner() -> int:\n"
        "        nested_only = 1\n"
        "        return nested_only\n"
        "    class Local:\n"
        "        class_only = 1\n"
        "    return result\n",
        encoding="utf-8",
    )

    symbols = _symbols(code_symbols.build_symbol_index(package, revision="c" * 40))
    outer = symbols["trade_rl.nested.outer"]

    assert "result" in outer["local_names"]
    assert "value" in outer["local_names"]
    assert "nested_only" not in outer["local_names"]
    assert "class_only" not in outer["local_names"]


def test_index_rejects_non_commit_revision(tmp_path: Path) -> None:
    code_symbols = _code_symbols()
    package = tmp_path / "trade_rl"
    package.mkdir()
    (package / "demo.py").write_text("def demo() -> None:\n    pass\n", encoding="utf-8")

    try:
        code_symbols.build_symbol_index(package, revision="main")
    except code_symbols.GuideCodeSymbolError as exc:
        assert "40-character hexadecimal" in str(exc)
    else:
        raise AssertionError("expected GuideCodeSymbolError for non-commit revision")
