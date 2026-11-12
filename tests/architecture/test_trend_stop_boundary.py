"""The trend-stop study reuses the canonical replay and never leaves the box.

It must not grow a second account or fill model, must not reach the network, must keep
the exploratory data boundary, and must leave existing strategies and defaults alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.strategies as strategies
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
STUDY = PACKAGE / "evaluation" / "trend_stop_study"
RULE = PACKAGE / "strategies" / "rules" / "trend_stop.py"

FORBIDDEN_ACCOUNTING = {
    "BookState",
    "MarketExecutor",
    "execute_native_target_transition",
    "SharedPortfolioSession",
}
NETWORK_MODULES = ("socket", "urllib", "http", "requests", "ssl", "aiohttp")


def _imports(path: Path) -> set[str]:
    return ImportCollector(PACKAGE).collect(path)


def _names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def test_the_study_package_has_the_documented_modules() -> None:
    for name in (
        "__init__.py",
        "__main__.py",
        "protocol.py",
        "metrics.py",
        "runner.py",
        "data.py",
        "cli.py",
    ):
        assert (STUDY / name).is_file(), name


def test_no_second_backtester_the_study_calls_the_shared_replay() -> None:
    for path in sorted(STUDY.glob("*.py")):
        assert not FORBIDDEN_ACCOUNTING & _names(path), path.name
    assert "run_shared_portfolio_replay" in _names(STUDY / "runner.py")
    runner = (STUDY / "runner.py").read_text(encoding="utf-8")
    assert "settle_terminal_position=True" in runner
    assert "capture_ledger_evidence=True" in runner


def test_the_study_and_the_rule_never_reach_the_network() -> None:
    for path in (*sorted(STUDY.glob("*.py")), RULE):
        imported = _imports(path)
        assert not any(
            within_module(name, module)
            for name in imported
            for module in NETWORK_MODULES
        ), path.name
    data = (STUDY / "data.py").read_text(encoding="utf-8")
    assert "allow_network=False" in data
    assert "allow_network=True" not in data


def test_only_the_data_module_touches_integrations() -> None:
    for path in sorted(STUDY.glob("*.py")):
        uses = any(
            within_module(name, "trade_rl.integrations") for name in _imports(path)
        )
        assert uses == (path.name == "data.py"), path.name


def test_the_rule_is_strategy_layer_only() -> None:
    imported = _imports(RULE)
    for forbidden in (
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.integrations",
        "trade_rl.risk",
    ):
        assert not any(within_module(name, forbidden) for name in imported), forbidden
    assert "TrendStopPortfolioPolicy" not in strategies.__all__


def test_the_exploratory_data_boundary_is_pinned_in_the_protocol() -> None:
    from trade_rl.evaluation.trend_stop_study.protocol import (
        DATA_END_EXCLUSIVE,
        PROTOCOL,
    )

    assert DATA_END_EXCLUSIVE == "2026-08-01T00:00:00"
    assert PROTOCOL.evaluation_end <= DATA_END_EXCLUSIVE
    assert PROTOCOL.data_end_exclusive == DATA_END_EXCLUSIVE
