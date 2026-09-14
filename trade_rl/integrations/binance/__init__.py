"""Maintained Binance integration facade."""

from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.integrations.binance.cache import (
    BinanceVisionCachePlan,
    BinanceVisionCacheReport,
    inspect_binance_vision_cache,
    inspect_binance_vision_urls,
    plan_binance_vision_cache,
    require_complete_binance_vision_cache,
    sync_binance_vision_cache,
    sync_binance_vision_urls,
    validate_cached_vision_payload,
    vision_cache_path,
)
from trade_rl.integrations.binance.dataset import (
    BinanceDatasetBuildResult,
    BinanceMarketDataSource,
    binance_multitimeframe_feature_specs,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.metadata import (
    BinanceExchangeInfoSnapshot,
    BinanceInstrumentMetadata,
    FrozenBinanceExchangeInfoTransport,
)
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    BinanceTransportMode,
    BinanceUnsupportedContractError,
)
from trade_rl.integrations.binance.vision import (
    binance_interval_milliseconds,
    plan_vision_kline_urls,
    vision_funding_url,
    vision_kline_url,
    vision_monthly_kline_url,
)

__all__ = [
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
    "plan_binance_vision_cache",
    "plan_vision_kline_urls",
    "require_complete_binance_vision_cache",
    "sync_binance_vision_cache",
    "sync_binance_vision_urls",
    "validate_cached_vision_payload",
    "vision_cache_path",
    "vision_funding_url",
    "vision_kline_url",
    "vision_monthly_kline_url",
]
