"""Closed study stays retired; the retained pure trend-stop policy stays core-owned."""

from __future__ import annotations

from pathlib import Path

import trade_rl.strategies as strategies
from tests.evaluation.trend_stop_archive_support import SPEC_SOURCE, archived_text
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
    assert not (ROOT / "docs/specs/trend-following-stops.md").exists()
    text = (ROOT / "docs/research/current-status.md").read_text(encoding="utf-8")
    section = text.split("## Trend following with fast stops (", 1)[1].split(
        "\n## ", 1
    )[0]
    assert "NO_WINNER" in section and "stop_rule_triggered=true" in section
    assert (
        "https://github.com/shuntatsu/trade_rl_research/blob/main/migration/trend_stop_archive.json"
        in section
    )
    # The current navigation link is not the historical source identity.
    # Verify the exact pre-retirement spec using its immutable Git blob and SHA-256.
    historical_spec = archived_text(SPEC_SOURCE)
    assert "Status: Active" in historical_spec and "NO_WINNER" in historical_spec
