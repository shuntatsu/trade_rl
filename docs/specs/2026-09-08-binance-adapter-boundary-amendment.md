# Binance Adapter Boundary Amendment

Date: 2026-09-08 (JST)
Applies to: `docs/specs/2026-09-08-lean-package-boundaries-design.md`
Status: normative clarification discovered during Phase 2B dependency review

## Conclusion

The Binance adapter requires one additional IO-free module, `integrations/binance/types.py`, to prevent circular ownership between transport, Vision planning, cache verification, and metadata.

This is a structural clarification only. It does not change any Binance request, retry, cache, parsing, metadata, dataset, or execution semantics.

## Final Binance package

```text
trade_rl/integrations/binance/
├── __init__.py
├── types.py
├── vision.py
├── cache.py
├── metadata.py
├── transport.py
└── dataset.py
```

## Ownership

- `types.py`: shared enums, adapter errors, and dependency-neutral coercion/validation helpers required by multiple Binance submodules; no network/filesystem IO and no import of another Binance submodule.
- `vision.py`: Binance Vision URL planning and archive/CSV parsing.
- `cache.py`: cache paths/evidence validation, and cache planning/inspection/synchronization.
- `metadata.py`: exchange-info snapshot value objects, instrument metadata parsing/contract conversion, and frozen exchange-info transport wrapper through a local protocol.
- `transport.py`: bounded public HTTP/REST/Vision transport, retry behavior, REST query construction, and source retrieval orchestration.
- `dataset.py`: kline/funding conversion into market series, `BinanceMarketDataSource`, feature preset construction, and final `MarketDataset` assembly.

## Dependency direction

```text
types
  ^
  ├── vision
  ├── cache
  ├── metadata
  └── transport
       ^
       └── dataset

vision  <- cache
vision  <- transport
cache   <- transport   (transport may use low-level cache verification helpers)
metadata <- transport  (transport may construct exchange-info snapshots)
transport + metadata + data <- dataset
```

No module may import `dataset.py` except the package facade/tests. `types.py` must not import another Binance submodule.

`cache.py` must not import the concrete `BinancePublicTransport`; high-level cache synchronization uses a protocol so `transport.py` may safely use cache verification without a cycle.

`metadata.py` must not import the concrete transport implementation; `FrozenBinanceExchangeInfoTransport` uses a local protocol for its base transport.

## Public API

`trade_rl.integrations` keeps its current package-level exports. `trade_rl.integrations.binance` as a package must export the maintained symbols previously exported by the flat `binance.py` module. Private old paths `trade_rl.integrations.binance_cache` and `trade_rl.integrations.frozen_binance_metadata` are not preserved as forwarding shims.

## Invariants

- public URL strings and query parameters are unchanged;
- retry counts/backoff/timeout behavior are unchanged;
- cached raw bytes and evidence schema/digest rules are unchanged;
- archive parsing, epoch normalization, funding alignment and metadata parsing are unchanged;
- `build_binance_market_dataset()` returns semantically identical datasets for the same fixtures;
- package relocation alone does not alter dataset identity;
- no new network call is introduced by import or metadata inspection;
- no compatibility shim survives solely for retired private module paths.
