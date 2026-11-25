"""Closed study stays retired; the retained pure trend-stop policy stays core-owned."""

from __future__ import annotations

from pathlib import Path

import trade_rl.strategies as strategies
from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
STUDY = PACKAGE / "evaluation" / "trend_stop_study"
RULE = PACKAGE / "strategies" / "rules" / "trend_stop.py"

NETWORK_MODULES = ("socket", "urllib", "http", "requests", "ssl", "aiohttp")


def _imports(path: Path) -> set[str]:
    return ImportCollector(PACKAGE).collect(path)


def test_closed_study_is_not_distributed_or_imported() -> None:
    assert not STUDY.exists()
    for path in PACKAGE.rglob("*.py"):
        assert not any(
            within_module(name, "trade_rl.evaluation.trend_stop_study")
            for name in _imports(path)
        ), path


def test_the_retained_rule_never_reaches_network() -> None:
    assert not any(
        within_module(name, module)
        for name in _imports(RULE)
        for module in NETWORK_MODULES
    )


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


def test_closed_study_archive_owns_evidence_not_execution() -> None:
    text = (ROOT / "docs/specs/trend-following-stops.md").read_text(encoding="utf-8")
    assert "Status: Archived" in text
    assert "NO_WINNER" in text
    assert (
        "https://github.com/shuntatsu/trade_rl_research/blob/main/migration/trend_stop_archive.json"
        in text
    )
