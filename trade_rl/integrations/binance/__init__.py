"""Maintained Binance integration facade."""

from trade_rl.data.contracts import InstrumentExecutionRule
from trade_rl.integrations.binance.dataset import (
    BinanceDatasetBuildResult,
    BinanceMarketDataSource,
    binance_multitimeframe_feature_specs,
    build_binance_market_dataset,
)
from trade_rl.integrations.binance.metadata import (
    BinanceExchangeInfoSnapshot,
    BinanceInstrumentMetadata,
)
from trade_rl.integrations.binance.transport import BinancePublicTransport
from trade_rl.integrations.binance.types import (
    BinanceMarket,
    BinanceTransportError,
    BinanceTransportMode,
    BinanceUnsupportedContractError,
)
from trade_rl.integrations.binance.vision import (
    plan_vision_kline_urls,
    vision_funding_url,
    vision_kline_url,
    vision_monthly_kline_url,
)

__all__ = [
    "BinanceDatasetBuildResult",
    "BinanceExchangeInfoSnapshot",
    "BinanceInstrumentMetadata",
    "InstrumentExecutionRule",
    "BinanceMarket",
    "BinanceMarketDataSource",
    "BinancePublicTransport",
    "BinanceTransportError",
    "BinanceTransportMode",
    "BinanceUnsupportedContractError",
    "binance_multitimeframe_feature_specs",
    "build_binance_market_dataset",
    "plan_vision_kline_urls",
    "vision_funding_url",
    "vision_kline_url",
    "vision_monthly_kline_url",
]
