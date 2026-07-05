"""
データソースモジュール

DataSource: 全ソース共通の抽象インターフェース。
FeaturePipeline はこれだけに依存し、実データ（CSV/Hyperliquid/Postgres）と
合成データ（SyntheticSource）を透過的に扱う。

load_orderflow / load_derivatives / load_funding はデータが無い銘柄・
ソースでは空DataFrameを返してよい（feature_pipelineがゼロ埋めする）。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from mars_lite.data.data_utils import TF_TO_MINUTES, resample_ohlcv


class DataSource(ABC):
    """マーケットデータソースの抽象基底クラス"""

    def __init__(self, symbols: List[str]):
        self.symbols = list(symbols)

    @abstractmethod
    def load_klines(
        self, symbol: str, timeframe: str = "1h",
        start: Optional[str] = None, end: Optional[str] = None,
    ) -> pd.DataFrame:
        """OHLCV。列: timestamp, open, high, low, close, volume"""
        raise NotImplementedError

    def load_orderflow(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None,
    ) -> pd.DataFrame:
        """1分オーダーフロー集計。列: timestamp, buy_volume, sell_volume,
        trade_count, avg_trade_size, volume_imbalance。デフォルトは空。"""
        return pd.DataFrame(columns=[
            "timestamp", "buy_volume", "sell_volume",
            "trade_count", "avg_trade_size", "volume_imbalance",
        ])

    def load_derivatives(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None,
    ) -> pd.DataFrame:
        """デリバティブ指標。列: timestamp, open_interest, ls_ratio, liq_notional"""
        return pd.DataFrame(columns=[
            "timestamp", "open_interest", "ls_ratio", "liq_notional",
        ])

    def load_funding(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None,
    ) -> pd.DataFrame:
        """funding rate実績。列: timestamp, funding_rate"""
        return pd.DataFrame(columns=["timestamp", "funding_rate"])


class SyntheticSource(DataSource):
    """
    合成マーケットデータソース（オフライン・アルファ注入対応）

    健全性試験（P0）・ユニットテスト・特徴パイプライン開発に使う。
    実データを一切使わずネットワーク非依存で動く。
    """

    DEFAULT_SYMBOLS = [
        "BTCUSDT", "XRPUSDT", "SUIUSDT", "BNBUSDT", "ETHUSDT", "PAXGUSDT", "ETHBTC",
    ]
    START_PRICES = {
        "BTCUSDT": 40000.0, "XRPUSDT": 0.6, "SUIUSDT": 1.5, "BNBUSDT": 300.0,
        "ETHUSDT": 2500.0, "PAXGUSDT": 2000.0, "ETHBTC": 0.06,
    }

    def __init__(
        self,
        n_days: int = 60,
        alpha: str = "none",
        alpha_strength: float = 0.002,
        seed: int = 0,
        symbols: Optional[List[str]] = None,
        start: str = "2024-01-01",
    ):
        symbols = symbols or self.DEFAULT_SYMBOLS
        super().__init__(symbols)
        from mars_lite.data.synthetic import (
            generate_market, build_ohlcv, build_orderflow,
            build_funding, build_derivatives,
        )

        rng = np.random.default_rng(seed)
        n_minutes = n_days * 1440
        returns, latent = generate_market(
            rng, len(symbols), n_minutes, alpha, alpha_strength,
        )

        self._klines: Dict[str, pd.DataFrame] = {}
        self._orderflow: Dict[str, pd.DataFrame] = {}
        self._funding: Dict[str, pd.DataFrame] = {}
        self._derivatives: Dict[str, pd.DataFrame] = {}

        for i, sym in enumerate(symbols):
            price = self.START_PRICES.get(sym, float(rng.uniform(1, 1000)))
            base_volume = float(rng.uniform(200, 2000))
            kdf = build_ohlcv(rng, returns[:, i], price, base_volume, start)
            self._klines[sym] = kdf
            self._orderflow[sym] = build_orderflow(rng, kdf, latent[:, i], alpha)
            self._funding[sym] = build_funding(rng, latent[:, i], start, n_days, alpha)
            self._derivatives[sym] = build_derivatives(rng, kdf, latent[:, i], alpha)

    def _slice(self, df: pd.DataFrame, start, end) -> pd.DataFrame:
        if df.empty:
            return df
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_klines(self, symbol, timeframe="1h", start=None, end=None):
        df = self._klines[symbol]
        if timeframe != "1m":
            df = resample_ohlcv(df, timeframe)
        return self._slice(df, start, end)

    def load_orderflow(self, symbol, start=None, end=None):
        return self._slice(self._orderflow[symbol], start, end)

    def load_derivatives(self, symbol, start=None, end=None):
        return self._slice(self._derivatives[symbol], start, end)

    def load_funding(self, symbol, start=None, end=None):
        return self._slice(self._funding[symbol], start, end)


class CsvSource(DataSource):
    """
    CSVディレクトリソース（fetch_futures.py / generate_sample_data.py 出力）

    レイアウト:
        {data_dir}/{SYMBOL}/1m/YYYY-MM-DD.csv
        {data_dir}/{SYMBOL}/orderflow_1m/YYYY-MM-DD.csv
        {data_dir}/{SYMBOL}/funding/funding.csv
        {data_dir}/{SYMBOL}/derivatives/derivatives.csv
    """

    def __init__(self, data_dir, symbols: List[str]):
        super().__init__(symbols)
        self.data_dir = Path(data_dir)
        self._kline_cache: Dict[str, pd.DataFrame] = {}

    def _load_daily_dir(self, symbol: str, name: str) -> pd.DataFrame:
        d = self.data_dir / symbol / name
        if not d.exists():
            return pd.DataFrame()
        files = sorted(d.glob("*.csv"))
        if not files:
            return pd.DataFrame()
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if "timestamp" in df.columns:
            # ミリ秒epoch or ISO文字列の両対応
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df

    def _1m_klines(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._kline_cache:
            self._kline_cache[symbol] = self._load_daily_dir(symbol, "1m")
        return self._kline_cache[symbol]

    def load_klines(self, symbol, timeframe="1h", start=None, end=None):
        df = self._1m_klines(symbol)
        if df.empty:
            return df
        if timeframe != "1m":
            df = resample_ohlcv(df, timeframe)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_orderflow(self, symbol, start=None, end=None):
        df = self._load_daily_dir(symbol, "orderflow_1m")
        if df.empty:
            return super().load_orderflow(symbol, start, end)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_derivatives(self, symbol, start=None, end=None):
        f = self.data_dir / symbol / "derivatives" / "derivatives.csv"
        if not f.exists():
            return super().load_derivatives(symbol, start, end)
        df = pd.read_csv(f)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_funding(self, symbol, start=None, end=None):
        f = self.data_dir / symbol / "funding" / "funding.csv"
        if not f.exists():
            return super().load_funding(symbol, start, end)
        df = pd.read_csv(f)
        if pd.api.types.is_numeric_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)


_HL_INFO_URL = "https://api.hyperliquid.xyz/info"
_HL_INTERVAL_MAP = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
_HL_SUFFIXES = ("USDT", "USDC", "PERP")


class HyperliquidSource(DataSource):
    """
    Hyperliquid実データソース（公開info API・認証不要）

    上位足（15m/1h/4h/1d）をネイティブ取得し、CSVキャッシュに保存する。
    OI/L/S比率/清算の履歴はHyperliquidでは取得不可（現在値のみ）のため、
    Binance派生データ（derivatives.py::fetch_binance_derivatives）や
    HLネイティブスナップショット（scripts/collect_hl_snapshots.py）で
    別途キャッシュされたCSVを読む。無ければ空フレーム（ゼロ埋め）。
    """

    def __init__(
        self,
        symbols: List[str],
        days: int = 180,
        cache_dir: str = "./data/hyperliquid",
        end: Optional[str] = None,
        symbol_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(symbols)
        self.days = days
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.end = pd.Timestamp(end) if end else pd.Timestamp.now().floor("h")
        self.symbol_map = symbol_map or {}
        self._kline_cache: Dict[str, pd.DataFrame] = {}

    def _coin(self, symbol: str) -> str:
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]
        s = symbol.upper()
        for suf in _HL_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                return s[: -len(suf)]
        return s

    def _cache_path(self, coin: str, interval: str) -> Path:
        return self.cache_dir / f"{coin}_{interval}.csv"

    def _fetch_candles(self, coin: str, interval: str) -> pd.DataFrame:
        start_ms = int((self.end - pd.Timedelta(days=self.days)).timestamp() * 1000)
        end_ms = int(self.end.timestamp() * 1000)
        rows = []
        cursor = start_ms
        for _ in range(200):
            resp = requests.post(_HL_INFO_URL, json={
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": interval,
                        "startTime": cursor, "endTime": end_ms},
            }, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            rows.extend(data)
            last_t = int(data[-1]["t"])
            step_ms = TF_TO_MINUTES[interval] * 60_000
            if last_t <= cursor:
                break
            cursor = last_t + step_ms
            if cursor >= end_ms:
                break
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        out = pd.DataFrame({
            "timestamp": pd.to_datetime(df["t"].astype("int64"), unit="ms"),
            "open": df["o"].astype(float), "high": df["h"].astype(float),
            "low": df["l"].astype(float), "close": df["c"].astype(float),
            "volume": df["v"].astype(float),
        })
        return out.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    def _load_candles(self, coin: str, interval: str) -> pd.DataFrame:
        cache_key = f"{coin}_{interval}"
        if cache_key in self._kline_cache:
            return self._kline_cache[cache_key]
        path = self._cache_path(coin, interval)
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            df = self._fetch_candles(coin, interval)
            if not df.empty:
                df.to_csv(path, index=False)
        self._kline_cache[cache_key] = df
        return df

    def load_klines(self, symbol, timeframe="1h", start=None, end=None):
        coin = self._coin(symbol)
        interval = _HL_INTERVAL_MAP.get(timeframe, timeframe)
        df = self._load_candles(coin, interval)
        if df.empty:
            return df
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_funding(self, symbol, start=None, end=None):
        coin = self._coin(symbol)
        cache_key = f"{coin}_funding"
        path = self.cache_dir / f"{coin}_funding.csv"
        if cache_key in self._kline_cache:
            df = self._kline_cache[cache_key]
        elif path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            self._kline_cache[cache_key] = df
        else:
            start_ms = int((self.end - pd.Timedelta(days=self.days)).timestamp() * 1000)
            end_ms = int(self.end.timestamp() * 1000)
            rows = []
            cursor = start_ms
            for _ in range(200):
                resp = requests.post(_HL_INFO_URL, json={
                    "type": "fundingHistory",
                    "coin": coin, "startTime": cursor, "endTime": end_ms,
                }, timeout=20)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                rows.extend(data)
                last_t = int(data[-1]["time"])
                if last_t <= cursor:
                    break
                cursor = last_t + 1
                if cursor >= end_ms:
                    break
            if rows:
                df = pd.DataFrame({
                    "timestamp": pd.to_datetime([int(r["time"]) for r in rows], unit="ms"),
                    "funding_rate": [float(r["fundingRate"]) for r in rows],
                }).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
                df.to_csv(path, index=False)
            else:
                df = pd.DataFrame(columns=["timestamp", "funding_rate"])
            self._kline_cache[cache_key] = df
        if df.empty:
            return df
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_orderflow(self, symbol, start=None, end=None):
        coin = self._coin(symbol)
        path = self.cache_dir / f"{coin}_orderflow_1m.csv"
        if not path.exists():
            return super().load_orderflow(symbol, start, end)
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_derivatives(self, symbol, start=None, end=None):
        coin = self._coin(symbol)
        path = self.cache_dir / f"{coin}_derivatives.csv"
        if not path.exists():
            return super().load_derivatives(symbol, start, end)
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)


def create_source(name: str, symbols: List[str], **kwargs) -> DataSource:
    """名前からDataSourceを構築するファクトリ"""
    if name == "synthetic":
        return SyntheticSource(symbols=symbols, **kwargs)
    if name == "csv":
        data_dir = kwargs.pop("data_dir", "./data")
        return CsvSource(data_dir, symbols, **kwargs)
    if name == "hyperliquid":
        return HyperliquidSource(symbols, **kwargs)
    if name == "postgres":
        raise NotImplementedError("postgres source not implemented in this checkout")
    raise ValueError(f"unknown source: {name}")
