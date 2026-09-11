from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.data.contracts as data_contracts
from trade_rl.data import MarketCalendarKind as PublicMarketCalendarKind
from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig
from trade_rl.data.market import MarketCalendarKind as MarketModuleCalendarKind

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "trade_rl" / "data"


def test_market_calendar_kind_is_owned_by_shared_data_contracts() -> None:
    assert hasattr(data_contracts, "MarketCalendarKind")
    contract_kind = data_contracts.MarketCalendarKind

    assert contract_kind.__module__ == "trade_rl.data.contracts"
    assert PublicMarketCalendarKind is contract_kind
    assert MarketModuleCalendarKind is contract_kind


def test_market_build_config_keeps_string_payload_while_using_calendar_contract() -> (
    None
):
    contract_kind = data_contracts.MarketCalendarKind
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        calendar_kind=contract_kind.SESSION,
        session_periods_per_year=252,
    )

    assert type(config.calendar_kind) is str
    assert config.calendar_kind == contract_kind.SESSION.value
    assert config.canonical_payload()["calendar_kind"] == "session_calendar"


def test_market_module_no_longer_defines_second_calendar_enum() -> None:
    market_path = DATA / "market.py"
    tree = ast.parse(market_path.read_text(encoding="utf-8"), filename=str(market_path))

    assert all(
        not isinstance(node, ast.ClassDef) or node.name != "MarketCalendarKind"
        for node in tree.body
    )


def test_builder_uses_calendar_contract_instead_of_duplicate_literals() -> None:
    builder = (DATA / "build" / "builder.py").read_text(encoding="utf-8")

    assert "MarketCalendarKind" in builder
    assert '"continuous_24_7"' not in builder
    assert '"session_calendar"' not in builder
