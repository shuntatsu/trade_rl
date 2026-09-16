from __future__ import annotations

import ast
from pathlib import Path

import trade_rl.integrations as integrations
import trade_rl.integrations.binance as binance

ROOT = Path(__file__).resolve().parents[2]
INTEGRATIONS = ROOT / "trade_rl" / "integrations"
BINANCE = INTEGRATIONS / "binance"

EXPECTED_BINANCE_PUBLIC_API = {
    "BOOK_DEPTH_PERCENTAGE_BANDS",
    "BinanceAggTradesSeries",
    "BinanceBookDepthSeries",
    "BinanceDatasetBuildResult",
    "BinanceExchangeInfoSnapshot",
    "BinanceInstrumentMetadata",
    "BinanceMarket",
    "BinanceMarketDataSource",
    "BinancePublicTransport",
    "BinanceTransportError",
    "BinanceTransportMode",
    "BinanceUnsupportedContractError",
    "BinanceVisionCachePlan",
    "BinanceVisionCacheReport",
    "FrozenBinanceExchangeInfoTransport",
    "InstrumentExecutionRule",
    "binance_interval_milliseconds",
    "binance_multitimeframe_feature_specs",
    "build_binance_market_dataset",
    "inspect_binance_vision_cache",
    "inspect_binance_vision_urls",
    "parse_vision_agg_trades_archive",
    "parse_vision_book_depth_archive",
    "plan_binance_vision_cache",
    "plan_vision_agg_trades_urls",
    "plan_vision_book_depth_urls",
    "plan_vision_kline_urls",
    "require_complete_binance_vision_cache",
    "sync_binance_vision_cache",
    "sync_binance_vision_urls",
    "validate_book_depth_reference_alignment",
    "validate_cached_vision_payload",
    "vision_agg_trades_url",
    "vision_book_depth_url",
    "vision_cache_path",
    "vision_funding_url",
    "vision_kline_url",
    "vision_monthly_kline_url",
}

EXPECTED_INTEGRATIONS_PUBLIC_API = {
    "BinanceMarket",
    "BinancePublicTransport",
    "BinanceTransportMode",
    "FrozenBinanceExchangeInfoTransport",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def test_binance_adapter_is_a_responsibility_package() -> None:
    for name in (
        "__init__.py",
        "types.py",
        "vision.py",
        "book_depth.py",
        "agg_trades.py",
        "cache.py",
        "metadata.py",
        "transport.py",
        "dataset.py",
    ):
        assert (BINANCE / name).is_file(), name


def test_retired_flat_binance_modules_are_absent() -> None:
    for name in (
        "binance.py",
        "binance_cache.py",
        "frozen_binance_metadata.py",
    ):
        assert not (INTEGRATIONS / name).exists(), name


def test_binance_package_preserves_flat_module_public_api() -> None:
    assert set(binance.__all__) == EXPECTED_BINANCE_PUBLIC_API
    for name in EXPECTED_BINANCE_PUBLIC_API:
        assert hasattr(binance, name), name


def test_integrations_package_preserves_public_api() -> None:
    assert set(integrations.__all__) == EXPECTED_INTEGRATIONS_PUBLIC_API
    for name in EXPECTED_INTEGRATIONS_PUBLIC_API:
        assert hasattr(integrations, name), name


def test_binance_types_are_dependency_neutral() -> None:
    imports = _imports(BINANCE / "types.py")
    assert not any(
        name.startswith("trade_rl.integrations.binance.") for name in imports
    )


def test_binance_vision_has_no_upward_adapter_dependency() -> None:
    imports = _imports(BINANCE / "vision.py")
    forbidden = (
        "trade_rl.integrations.binance.cache",
        "trade_rl.integrations.binance.metadata",
        "trade_rl.integrations.binance.transport",
        "trade_rl.integrations.binance.dataset",
    )
    assert not any(name.startswith(forbidden) for name in imports)


def test_binance_book_depth_is_provider_evidence_not_dataset_assembly() -> None:
    imports = _imports(BINANCE / "book_depth.py")
    forbidden = (
        "trade_rl.data.market",
        "trade_rl.data.build",
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.integrations.binance.dataset",
    )
    assert not any(name.startswith(forbidden) for name in imports)


def test_binance_agg_trades_is_provider_evidence_not_dataset_assembly() -> None:
    path = BINANCE / "agg_trades.py"
    imports = _imports(path)
    forbidden = (
        "trade_rl.data.market",
        "trade_rl.data.build",
        "trade_rl.evaluation",
        "trade_rl.simulation",
        "trade_rl.integrations.binance.dataset",
    )
    assert not any(name.startswith(forbidden) for name in imports)
    assert "_csv_rows_from_zip" not in path.read_text(encoding="utf-8")


def test_binance_provider_evidence_boundary_is_documented_as_non_pnl() -> None:
    text = (ROOT / "docs" / "architecture" / "package-boundaries.md").read_text(
        encoding="utf-8"
    )
    assert "## Provider evidence boundary" in text
    assert "integrations/binance/book_depth.py" in text
    assert "integrations/binance/agg_trades.py" in text
    assert "top-of-book" in text
    assert "MarketDataset" in text
    assert "P&Lのauthorityにはしない" in text
    assert "reference valueの`available_at`" in text
    assert "event timestamp" in text
    assert "archive publication" in text
    assert "既存canonical Studyやfrozen Experiment evidenceを書き換えない" in text


def test_binance_cache_does_not_import_concrete_transport_or_dataset() -> None:
    imports = _imports(BINANCE / "cache.py")
    forbidden = (
        "trade_rl.integrations.binance.transport",
        "trade_rl.integrations.binance.dataset",
    )
    assert not any(name.startswith(forbidden) for name in imports)


def test_binance_metadata_does_not_import_concrete_transport_or_dataset() -> None:
    imports = _imports(BINANCE / "metadata.py")
    forbidden = (
        "trade_rl.integrations.binance.transport",
        "trade_rl.integrations.binance.dataset",
    )
    assert not any(name.startswith(forbidden) for name in imports)


def test_binance_transport_does_not_depend_on_dataset_assembly() -> None:
    imports = _imports(BINANCE / "transport.py")
    assert not any(
        name.startswith("trade_rl.integrations.binance.dataset") for name in imports
    )
