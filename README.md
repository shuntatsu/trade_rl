# Trade RL

Trade RL is a research-grade baseline-anchored residual reinforcement-learning core for portfolio allocation. The maintained policy does not choose arbitrary portfolio weights directly. It makes bounded residual decisions around a deterministic trend baseline, while an independent shadow book provides the reward and evaluation reference.

> Production status: **NO-GO**. The repository provides research, evaluation, artifact and serving contracts. It does not claim a profitable or production-authorized strategy.

## What changed

The repository was rebuilt around one authoritative `trade_rl` package. The former `mars_lite` stack, direct-action PPO path, legacy scripts, duplicate metric implementations and legacy tests were removed rather than deprecated.

Key boundaries are now explicit:

- immutable domain manifests for datasets, signals, policy ensembles, selections and releases;
- canonical, content-addressed artifacts with staged and atomic publication;
- one implementation of return, risk and paired-comparison metrics;
- pure leak-checked walk-forward fold construction and chronological OOS stitching;
- a two-dimensional residual action schema with exact zero-action baseline identity;
- isolated execution, accounting and pre-trade risk components;
- validated serving bundles, atomic registry activation and fail-closed hot swaps;
- one `trade-rl` CLI and typed workflow boundaries.

## Causal market data

The maintained data path builds `MarketDataset` directly from real per-symbol CSV bars. It preserves a regular union clock, point-in-time listing and delisting masks, bar-level tradability, per-feature availability and staleness, and explicit volume units. Feature configuration, ordered symbols, instrument contracts, normalization output and all resolved arrays are bound into `dataset_id`.

```python
from datetime import datetime, timezone

from trade_rl.data import (
    CsvMarketDataSource,
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    MarketDatasetBuilder,
)

config = MarketBuildConfig(
    base_timeframe="1h",
    features=(
        FeatureSpec(name="ret_1", kind=FeatureKind.LOG_RETURN),
        FeatureSpec(name="funding_bps", kind=FeatureKind.FUNDING_BPS),
    ),
)
contracts = (
    InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2019, 9, 1, tzinfo=timezone.utc),
    ),
)
dataset = MarketDatasetBuilder(config).build(
    CsvMarketDataSource("data/market"),
    contracts,
)
```

Each CSV uses `timestamp,open,high,low,close,volume` and may add `available_at`, `funding_rate`, and `tradable`. `timestamp` is the market event time; `available_at` is the earliest timestamp at which that row was knowable. A delayed row is never injected retroactively into a same-time feature. The prior causal feature may remain carried with increasing staleness until a new on-time feature is available.

One point-in-time eligibility mask is shared by Trend, alpha, and pre-trade target construction. It requires the instrument to be active, tradable, and information-available throughout the requested trailing window. Future execution-state tradability remains the executor's responsibility and is not used to construct the current target.

## Deterministic dataset artifacts

A strict JSON build request can drive the maintained CLI path:

```json
{
  "source_root": "data/market",
  "base_timeframe": "1h",
  "features": [
    {"name": "ret_1", "kind": "log_return", "lookback": 1}
  ],
  "instruments": [
    {
      "symbol": "BTCUSDT",
      "listed_at": "2019-09-01T00:00:00Z",
      "volume_unit": "quote_notional"
    }
  ]
}
```

```bash
uv run trade-rl data build --config market-build.json --output output/dataset
```

The output contains a canonical `manifest.json` and deterministic `arrays.npz`. Both are digest-verified when reloaded. Identical configuration and source bytes produce identical dataset and artifact identities.

## Observation and Serving identity

Training and Serving use the same `ObservationBuilder`. Serving bundles bind the ordered observation schema digest and exact vector size in addition to `dataset_id` and the action schema. Structured inference fails closed before policy execution when the dataset identity, observation schema, or vector size does not match the active bundle.

## Install

```bash
uv sync --extra dev
```

## Commands

```bash
uv run trade-rl --version
uv run trade-rl data build --config market-build.json --output output/dataset
uv run trade-rl train config --timesteps 1024 --gamma 0.5 --seed 0 --seed 1
uv run trade-rl walk-forward plan \
  --bars 220 --train-bars 80 --checkpoint-bars 10 \
  --selection-bars 10 --test-bars 20 --purge-bars 2 --max-folds 2
```

The `data`, `signal`, `evaluate`, `registry`, and `serve` command groups expose the corresponding architectural boundaries. Additional application adapters should be added behind these boundaries rather than inside domain or evaluation code.

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
```

See [Architecture](docs/ARCHITECTURE.md) and [Research Status](docs/RESEARCH_STATUS.md).
