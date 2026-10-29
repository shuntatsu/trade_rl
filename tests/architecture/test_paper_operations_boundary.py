"""Attempt-2 tooling and the watchdog stay offline, journal-free and order-free."""

from __future__ import annotations

from pathlib import Path

import trade_rl.evaluation as evaluation
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
PAPER = PACKAGE / "evaluation" / "paper"
NETWORK = {"socket", "http", "urllib", "requests", "ssl", "ftplib"}


def _imports(name: str) -> set[str]:
    return ImportCollector(PACKAGE).collect(PAPER / name)


def test_attempt2_planning_has_no_network_collector_or_order_surface() -> None:
    imported = _imports("attempt2.py")
    assert not {item.split(".", 1)[0] for item in imported} & NETWORK
    for forbidden in (
        "trade_rl.integrations",
        "trade_rl.evaluation.paper.supervisor",
        "trade_rl.evaluation.paper.engine",
        "trade_rl.evaluation.paper.store",
        "trade_rl.evaluation.paper.assessment",
        "trade_rl.evaluation.final_test",
        "trade_rl.strategies",
        "trade_rl.simulation",
    ):
        assert not any(within_module(item, forbidden) for item in imported), forbidden


def test_watchdog_never_opens_the_journal_or_any_market_surface() -> None:
    imported = _imports("watchdog.py")
    assert not {item.split(".", 1)[0] for item in imported} & NETWORK
    for forbidden in (
        "trade_rl.integrations",
        "trade_rl.evaluation.paper.store",
        "trade_rl.evaluation.paper.engine",
        "trade_rl.evaluation.paper.assessment",
        "trade_rl.evaluation.paper.control",
        "trade_rl.strategies",
        "trade_rl.simulation",
    ):
        assert not any(within_module(item, forbidden) for item in imported), forbidden


def test_paper_operations_do_not_widen_the_public_evaluation_api() -> None:
    assert not any(
        name.lower().startswith(("attempt2", "watchdog", "heartbeat"))
        for name in evaluation.__all__
    )
