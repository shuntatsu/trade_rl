# Binance Adapter Definition Inventory

Date: 2026-09-08 (JST)
Base: `343c8af5dfcea8a472ccc44ef2b617ea7f4d0c68`
Inventory evidence: workflow run `34200128207`, exact inventory head `f76a28000bbbe02c8a97959fd498682ee6b1f514`

Every top-level class/function in the three pre-refactor Binance modules is classified below. No production split may delete an unclassified definition.

Legend:

```text
MOVE <old definition> -> <new owner>
DELETE <old definition> : <evidence-backed replacement>
```

## `trade_rl/integrations/binance.py`

### `types.py` — dependency-neutral shared contracts/helpers

```text
MOVE 0073-0076 BinanceMarket -> binance/types.py
MOVE 0079-0082 BinanceTransportMode -> binance/types.py
MOVE 0085-0086 BinanceTransportError -> binance/types.py
MOVE 0129-0130 BinanceUnsupportedContractError -> binance/types.py
MOVE 0223-0227 _market -> binance/types.py
MOVE 0230-0234 _mode -> binance/types.py
MOVE 0237-0240 _aware_utc -> binance/types.py
MOVE 0399-0408 _finite_float -> binance/types.py
```

These helpers are pure coercion/validation and preserve their existing error messages. `types.py` performs no filesystem/network IO and imports no sibling Binance module.

### `cache.py` — cache integrity

```text
MOVE 0089-0126 validate_cached_vision_payload -> binance/cache.py
```

### `metadata.py` — exchange/instrument metadata

```text
MOVE 0133-0140 _freeze_json -> binance/metadata.py
MOVE 0143-0146 _freeze_json_object -> binance/metadata.py
MOVE 0149-0154 _mutable_json -> binance/metadata.py
MOVE 0157-0158 _mutable_json_object -> binance/metadata.py
MOVE 0162-0200 BinanceInstrumentMetadata -> binance/metadata.py
MOVE 0204-0212 BinanceExchangeInfoSnapshot -> binance/metadata.py
MOVE 1084-1097 _filter_value -> binance/metadata.py
MOVE 1100-1158 _metadata_from_exchange_info -> binance/metadata.py
```

### `vision.py` — Binance Vision planning/archive parsing

```text
MOVE 0243-0244 _epoch_ms -> binance/vision.py
MOVE 0247-0254 _normalize_epoch_ms -> binance/vision.py
MOVE 0257-0261 _interval_ms -> binance/vision.py
MOVE 0264-0270 _day_floor_ms -> binance/vision.py
MOVE 0273-0278 _iter_days -> binance/vision.py
MOVE 0281-0289 _iter_months -> binance/vision.py
MOVE 0292-0307 vision_kline_url -> binance/vision.py
MOVE 0310-0327 vision_monthly_kline_url -> binance/vision.py
MOVE 0330-0333 _next_month -> binance/vision.py
MOVE 0336-0358 plan_vision_kline_urls -> binance/vision.py
MOVE 0361-0374 vision_funding_url -> binance/vision.py
MOVE 0377-0392 _csv_rows_from_zip -> binance/vision.py
MOVE 0395-0396 _looks_like_header -> binance/vision.py
```

### `transport.py` — public HTTP/REST/Vision transport

```text
MOVE 0411-0899 BinancePublicTransport -> binance/transport.py
```

### `dataset.py` — raw-row conversion/source/dataset assembly

```text
MOVE 0216-0220 BinanceDatasetBuildResult -> binance/dataset.py
MOVE 0902-0939 _parse_kline_rows -> binance/dataset.py
MOVE 0942-0970 _align_funding -> binance/dataset.py
MOVE 0973-1081 BinanceMarketDataSource -> binance/dataset.py
MOVE 1161-1172 _optional_values -> binance/dataset.py
MOVE 1175-1185 _optional_datetimes -> binance/dataset.py
MOVE 1188-1371 _extended_timeframe_feature_definitions -> binance/dataset.py
MOVE 1374-1437 binance_multitimeframe_feature_specs -> binance/dataset.py
MOVE 1440-1475 _default_features -> binance/dataset.py
MOVE 1478-1656 build_binance_market_dataset -> binance/dataset.py
```

### Old `binance.py.__all__` — exact public facade contract

The new `trade_rl.integrations.binance` package must export exactly these maintained names unless an independent existing package-level contract adds more:

```text
BinanceDatasetBuildResult
BinanceExchangeInfoSnapshot
BinanceInstrumentMetadata
InstrumentExecutionRule
BinanceMarket
BinanceMarketDataSource
BinancePublicTransport
BinanceTransportError
BinanceTransportMode
BinanceUnsupportedContractError
binance_multitimeframe_feature_specs
build_binance_market_dataset
plan_vision_kline_urls
vision_funding_url
vision_kline_url
vision_monthly_kline_url
```

## `trade_rl/integrations/binance_cache.py` -> `binance/cache.py`

```text
MOVE 0024-0027 _VisionArchiveTransport -> binance/cache.py
MOVE 0031-0047 BinanceVisionCachePlan -> binance/cache.py
MOVE 0051-0074 BinanceVisionCacheReport -> binance/cache.py
MOVE 0077-0080 _aware_utc -> DELETE : use dependency-neutral binance.types._aware_utc
MOVE 0083-0087 _ordered_nonempty -> binance/cache.py
MOVE 0090-0094 _official_urls -> binance/cache.py
MOVE 0097-0100 _next_month -> binance/cache.py
MOVE 0103-0148 plan_binance_vision_cache -> binance/cache.py
MOVE 0151-0157 vision_cache_path -> binance/cache.py
MOVE 0160-0194 inspect_binance_vision_urls -> binance/cache.py
MOVE 0197-0202 inspect_binance_vision_cache -> binance/cache.py
MOVE 0205-0217 require_complete_binance_vision_cache -> binance/cache.py
MOVE 0220-0263 sync_binance_vision_urls -> binance/cache.py
MOVE 0266-0273 sync_binance_vision_cache -> binance/cache.py
```

The concrete `BinancePublicTransport` union annotation is removed; the existing `_VisionArchiveTransport` protocol is the cache synchronization boundary.

Old `binance_cache.py.__all__` becomes the public surface of `trade_rl.integrations.binance.cache`:

```text
BinanceVisionCachePlan
BinanceVisionCacheReport
inspect_binance_vision_cache
inspect_binance_vision_urls
plan_binance_vision_cache
require_complete_binance_vision_cache
sync_binance_vision_cache
sync_binance_vision_urls
vision_cache_path
```

## `trade_rl/integrations/frozen_binance_metadata.py` -> `binance/metadata.py`

```text
MOVE 0021-0027 _ExchangeInfoTransport -> binance/metadata.py
MOVE 0030-0033 _require_mapping -> binance/metadata.py
MOVE 0036-0039 _require_non_empty -> binance/metadata.py
MOVE 0042-0050 _parse_utc -> binance/metadata.py
MOVE 0054-0154 FrozenBinanceExchangeInfoTransport -> binance/metadata.py
```

`FrozenBinanceExchangeInfoTransport` retains a protocol dependency and must not import concrete `transport.py`.

Old public surface preserved through `trade_rl.integrations`:

```text
FrozenBinanceExchangeInfoTransport
```

## Coverage check

Inventory workflow observed:

```text
binance.py: 41 top-level classes/functions
binance_cache.py: 13 top-level classes/functions
frozen_binance_metadata.py: 5 top-level classes/functions
```

All 59 observed definitions above have exactly one MOVE or DELETE disposition. The split implementation must repeat the AST inventory against the old exact base and fail if the observed name set differs from this document before applying migration.
