from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from trade_rl.data import MarketDataset, write_market_dataset_files


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - window + 1)
        result[index] = float(np.mean(values[start : index + 1]))
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    for index in range(values.size):
        start = max(0, index - window + 1)
        result[index] = float(np.std(values[start : index + 1]))
    return result


def build_demo_dataset(n_bars: int = 1_024) -> MarketDataset:
    """Build a deterministic hourly BTC-like dataset for a training smoke run."""

    if isinstance(n_bars, bool) or not isinstance(n_bars, int) or n_bars < 512:
        raise ValueError("n_bars must be an integer of at least 512")

    rng = np.random.default_rng(20260714)
    timestamps = np.datetime64("2024-01-01T00:00:00", "ns") + np.arange(
        n_bars
    ) * np.timedelta64(1, "h")

    cycle = 0.00035 * np.sin(np.arange(n_bars, dtype=np.float64) / 36.0)
    innovations = rng.normal(loc=0.0, scale=0.0035, size=n_bars)
    log_returns = 0.00005 + cycle + innovations
    close = 30_000.0 * np.exp(np.cumsum(log_returns))
    open_price = np.concatenate(([close[0]], close[:-1]))
    intrabar_range = 0.0015 + np.abs(rng.normal(0.0, 0.001, size=n_bars))
    high = np.maximum(open_price, close) * (1.0 + intrabar_range)
    low = np.minimum(open_price, close) * (1.0 - intrabar_range)
    volume = 750.0 + 180.0 * np.abs(rng.normal(size=n_bars))

    observed_returns = np.zeros(n_bars, dtype=np.float64)
    observed_returns[1:] = np.log(close[1:] / close[:-1])
    volatility_24h = _rolling_std(observed_returns, 24)
    market_trend_24h = _rolling_mean(observed_returns, 24)

    features = np.stack((observed_returns, volatility_24h), axis=-1).astype(
        np.float32
    )[:, None, :]
    global_features = market_trend_24h.astype(np.float32)[:, None]
    prices = {
        "open": open_price[:, None],
        "high": high[:, None],
        "low": low[:, None],
        "close": close[:, None],
        "volume": volume[:, None],
    }
    feature_available = np.ones(features.shape, dtype=np.bool_)
    tradable = np.ones((n_bars, 1), dtype=np.bool_)
    funding_rate = np.zeros((n_bars, 1), dtype=np.float64)

    identity = hashlib.sha256()
    identity.update(b"trade-rl-quickstart-dataset-v1")
    for array in (
        timestamps.astype("datetime64[ns]").astype(np.int64),
        features,
        global_features,
        *prices.values(),
    ):
        identity.update(np.ascontiguousarray(array).tobytes(order="C"))

    return MarketDataset(
        dataset_id=identity.hexdigest(),
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=features,
        global_features=global_features,
        open=prices["open"],
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        volume=prices["volume"],
        funding_rate=funding_rate,
        tradable=tradable,
        feature_available=feature_available,
        feature_names=("log_return_1h", "volatility_24h"),
        global_feature_names=("market_trend_24h",),
        periods_per_year=8_760,
        fee_rate=np.full((n_bars, 1), 0.0005, dtype=np.float64),
        spread_rate=np.full((n_bars, 1), 0.0002, dtype=np.float64),
        max_participation_rate=np.full((n_bars, 1), 0.05, dtype=np.float64),
        borrow_available=np.ones((n_bars, 1), dtype=np.bool_),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic market dataset artifact for START.md."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/quickstart/dataset"),
        help="directory that receives manifest.json and arrays.npz",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=1_024,
        help="number of hourly bars; must be at least 512",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = build_demo_dataset(args.bars)
    result = write_market_dataset_files(args.output, dataset)
    print(
        json.dumps(
            {
                "arrays_path": str(result.arrays_path),
                "artifact_digest": result.artifact_digest,
                "dataset_id": dataset.dataset_id,
                "manifest_path": str(result.manifest_path),
                "n_bars": dataset.n_bars,
                "symbols": list(dataset.symbols),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
