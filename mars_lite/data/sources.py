"""
データソースモジュール

DataSource: 全ソース共通の抽象インターフェース。
FeaturePipeline はこれだけに依存し、実データ（CSV/Hyperliquid/Postgres）と
合成データ（SyntheticSource）を透過的に扱う。

load_orderflow / load_derivatives / load_funding はデータが無い銘柄・
ソースでは空DataFrameを返してよい（feature_pipelineがゼロ埋めする）。
"""

import hashlib
import os
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
        self,
        symbol: str,
        timeframe: str = "1h",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """OHLCV。列: timestamp, open, high, low, close, volume"""
        raise NotImplementedError

    def load_orderflow(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """1分オーダーフロー集計。列: timestamp, buy_volume, sell_volume,
        trade_count, avg_trade_size, volume_imbalance。デフォルトは空。"""
        return pd.DataFrame(
            columns=[
                "timestamp",
                "buy_volume",
                "sell_volume",
                "trade_count",
                "avg_trade_size",
                "volume_imbalance",
            ]
        )

    def load_derivatives(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """デリバティブ指標。列: timestamp, open_interest, ls_ratio, liq_notional"""
        return pd.DataFrame(
            columns=[
                "timestamp",
                "open_interest",
                "ls_ratio",
                "liq_notional",
            ]
        )

    def load_funding(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
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
        "BTCUSDT",
        "XRPUSDT",
        "SUIUSDT",
        "BNBUSDT",
        "ETHUSDT",
        "PAXGUSDT",
        "ETHBTC",
    ]
    START_PRICES = {
        "BTCUSDT": 40000.0,
        "XRPUSDT": 0.6,
        "SUIUSDT": 1.5,
        "BNBUSDT": 300.0,
        "ETHUSDT": 2500.0,
        "PAXGUSDT": 2000.0,
        "ETHBTC": 0.06,
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
            build_derivatives,
            build_funding,
            build_ohlcv,
            build_orderflow,
            generate_market,
        )

        rng = np.random.default_rng(seed)
        n_minutes = n_days * 1440
        returns, latent = generate_market(
            rng,
            len(symbols),
            n_minutes,
            alpha,
            alpha_strength,
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
            df = (
                df.sort_values("timestamp")
                .drop_duplicates("timestamp")
                .reset_index(drop=True)
            )
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


def _hl_post(payload: dict, max_retries: int = 6) -> list:
    """Hyperliquid info API（429時は指数バックオフでリトライ）"""
    import time

    for attempt in range(max_retries):
        resp = requests.post(_HL_INFO_URL, json=payload, timeout=20)
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 30))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return []


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
            data = _hl_post(
                {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": interval,
                        "startTime": cursor,
                        "endTime": end_ms,
                    },
                }
            )
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
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(rows)
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["t"].astype("int64"), unit="ms"),
                "open": df["o"].astype(float),
                "high": df["h"].astype(float),
                "low": df["l"].astype(float),
                "close": df["c"].astype(float),
                "volume": df["v"].astype(float),
            }
        )
        return (
            out.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

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
                data = _hl_post(
                    {
                        "type": "fundingHistory",
                        "coin": coin,
                        "startTime": cursor,
                        "endTime": end_ms,
                    }
                )
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
                df = (
                    pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(
                                [int(r["time"]) for r in rows], unit="ms"
                            ),
                            "funding_rate": [float(r["fundingRate"]) for r in rows],
                        }
                    )
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
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
            df = super().load_derivatives(symbol, start, end)
        else:
            df = pd.read_csv(path, parse_dates=["timestamp"])

        # HLネイティブのスナップショット（collect_hl_snapshots.py蓄積分）が
        # あれば、重複する期間のopen_interestはネイティブ値で上書きする
        # （Binance代理はあくまで履歴の穴埋め）。ls_ratio/liq_notionalは
        # HLに相当データが無いためBinance代理のまま。
        snap_path = self.cache_dir / "snapshots" / f"{coin}_ctx.csv"
        if snap_path.exists():
            snap = pd.read_csv(snap_path, parse_dates=["timestamp"])
            if not snap.empty:
                snap = snap.sort_values("timestamp")
                if df.empty:
                    df = pd.DataFrame(
                        {
                            "timestamp": snap["timestamp"],
                            "open_interest": snap["open_interest"],
                            "ls_ratio": 1.0,
                            "liq_notional": 0.0,
                        }
                    )
                else:
                    df = df.set_index("timestamp")
                    native_oi = snap.set_index("timestamp")["open_interest"]
                    df = df.reindex(df.index.union(native_oi.index)).sort_index()
                    df.loc[native_oi.index, "open_interest"] = native_oi
                    df = df.ffill().reset_index().rename(columns={"index": "timestamp"})

        if df.empty:
            return df
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)


_BITGET_BASE_URL = "https://api.bitget.com"
_BITGET_INTERVAL_MAP = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1D"}


def _bitget_get(path: str, params: dict, max_retries: int = 6) -> list:
    """Bitget公開API（USDT-M先物、認証不要。429・接続断は指数バックオフでリトライ）"""
    import time

    url = _BITGET_BASE_URL + path
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2**attempt, 30))
            continue
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 30))
            continue
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "00000":
            raise RuntimeError(
                f"bitget api error: {data.get('code')} {data.get('msg')}"
            )
        return data.get("data") or []
    resp.raise_for_status()
    return []


class BitgetSource(DataSource):
    """
    Bitget実データソース（USDT-M先物公開API・認証不要）

    上位足（15m/1h/4h/1d）とfunding rateを取得しCSVキャッシュに保存する。
    candlesエンドポイントは1回のリクエストで最大1000本・start/end間隔90日
    までしか返さないため、endTimeを古い方へずらしながら複数回叩いて
    HyperliquidSource同様にページングする。OI/L-S比率/清算はBitgetでは
    未対応のため空フレーム（feature_pipelineがゼロ埋め）。
    """

    def __init__(
        self,
        symbols: List[str],
        days: int = 180,
        cache_dir: str = "./data/bitget",
        end: Optional[str] = None,
        product_type: str = "usdt-futures",
    ):
        super().__init__(symbols)
        self.days = days
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.end = pd.Timestamp(end) if end else pd.Timestamp.now().floor("h")
        self.product_type = product_type
        self._kline_cache: Dict[str, pd.DataFrame] = {}

    def _fetch_candles(
        self, symbol: str, granularity: str, step_ms: int
    ) -> pd.DataFrame:
        start_ms = int((self.end - pd.Timedelta(days=self.days)).timestamp() * 1000)
        cursor_end = int(self.end.timestamp() * 1000)
        rows: list = []
        prev_oldest = None
        for _ in range(400):
            cursor_start = max(start_ms, cursor_end - 90 * 86_400_000)
            data = _bitget_get(
                "/api/v2/mix/market/candles",
                {
                    "symbol": symbol,
                    "productType": self.product_type,
                    "granularity": granularity,
                    "startTime": str(cursor_start),
                    "endTime": str(cursor_end),
                    "limit": "1000",
                },
            )
            if not data:
                break
            rows = data + rows
            oldest_t = int(data[0][0])
            if prev_oldest is not None and oldest_t >= prev_oldest:
                break  # 進捗なし（これ以上遡れない）
            prev_oldest = oldest_t
            if oldest_t <= start_ms:
                break
            cursor_end = oldest_t - step_ms
            if cursor_end <= start_ms:
                break
        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(
            rows,
            columns=["t", "o", "h", "l", "c", "vbase", "vquote"],
        )
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["t"].astype("int64"), unit="ms"),
                "open": df["o"].astype(float),
                "high": df["h"].astype(float),
                "low": df["l"].astype(float),
                "close": df["c"].astype(float),
                "volume": df["vbase"].astype(float),
            }
        )
        return (
            out.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol}_{interval}.csv"

    def _load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        granularity = _BITGET_INTERVAL_MAP.get(timeframe, timeframe)
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._kline_cache:
            return self._kline_cache[cache_key]
        path = self._cache_path(symbol, timeframe)
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            step_ms = TF_TO_MINUTES[timeframe] * 60_000
            df = self._fetch_candles(symbol, granularity, step_ms)
            if not df.empty:
                df.to_csv(path, index=False)
        self._kline_cache[cache_key] = df
        return df

    def load_klines(self, symbol, timeframe="1h", start=None, end=None):
        df = self._load_candles(symbol, timeframe)
        if df.empty:
            return df
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_funding(self, symbol, start=None, end=None):
        cache_key = f"{symbol}_funding"
        path = self.cache_dir / f"{symbol}_funding.csv"
        if cache_key in self._kline_cache:
            df = self._kline_cache[cache_key]
        elif path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            self._kline_cache[cache_key] = df
        else:
            start_ms = int((self.end - pd.Timedelta(days=self.days)).timestamp() * 1000)
            rows: list = []
            for page in range(1, 400):
                data = _bitget_get(
                    "/api/v2/mix/market/history-fund-rate",
                    {
                        "symbol": symbol,
                        "productType": self.product_type,
                        "pageSize": "100",
                        "pageNo": str(page),
                    },
                )
                if not data:
                    break
                rows.extend(data)
                oldest_t = int(data[-1]["fundingTime"])
                if oldest_t <= start_ms:
                    break
            if rows:
                df = (
                    pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(
                                [int(r["fundingTime"]) for r in rows],
                                unit="ms",
                            ),
                            "funding_rate": [float(r["fundingRate"]) for r in rows],
                        }
                    )
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
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


_OKX_BASE_URL = "https://www.okx.com"
_OKX_BAR_MAP = {"15m": "15m", "1h": "1H", "4h": "4H", "1d": "1Dutc"}
_OKX_SUFFIXES = ("USDT", "USDC")


def _okx_get(path: str, params: dict, max_retries: int = 6) -> list:
    """OKX公開API（USDT建て無期限先物、認証不要。429・接続断は指数バックオフでリトライ）"""
    import time

    url = _OKX_BASE_URL + path
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=20)
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2**attempt, 30))
            continue
        if resp.status_code == 429:
            time.sleep(min(2**attempt, 30))
            continue
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "0":
            raise RuntimeError(f"okx api error: {data.get('code')} {data.get('msg')}")
        return data.get("data") or []
    resp.raise_for_status()
    return []


class OKXSource(DataSource):
    """
    OKX実データソース（USDT建て無期限先物 公開API・認証不要）

    3取引所（Hyperliquid/Bitget/OKX）の中では最も長期の1h足履歴を持つ
    （実測: BTC-USDT-SWAPで1000日以上前まで取得可能）。history-candles
    エンドポイントは`after`（このts「より前」を返す）でシンプルに後方
    ページングできる。OI/L-S比率/清算はOKXでは未対応のため空フレーム
    （feature_pipelineがゼロ埋め）。
    """

    def __init__(
        self,
        symbols: List[str],
        days: int = 180,
        cache_dir: str = "./data/okx",
        end: Optional[str] = None,
    ):
        super().__init__(symbols)
        self.days = days
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.end = pd.Timestamp(end) if end else pd.Timestamp.now().floor("h")
        self._kline_cache: Dict[str, pd.DataFrame] = {}

    def _inst_id(self, symbol: str) -> str:
        s = symbol.upper()
        for suf in _OKX_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                return f"{s[: -len(suf)]}-{suf}-SWAP"
        return f"{s}-USDT-SWAP"

    def _fetch_candles(self, inst_id: str, bar: str) -> pd.DataFrame:
        start_ms = int((self.end - pd.Timedelta(days=self.days)).timestamp() * 1000)
        cursor_after = int(self.end.timestamp() * 1000) + 1
        rows: list = []
        for _ in range(400):
            data = _okx_get(
                "/api/v5/market/history-candles",
                {
                    "instId": inst_id,
                    "bar": bar,
                    "after": str(cursor_after),
                    "limit": "100",
                },
            )
            if not data:
                break
            rows.extend(data)
            oldest_t = int(data[-1][0])
            if oldest_t >= cursor_after:
                break  # 進捗なし（これ以上遡れない）
            cursor_after = oldest_t
            if oldest_t <= start_ms:
                break
        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
        df = pd.DataFrame(
            [r[:6] for r in rows],
            columns=["t", "o", "h", "l", "c", "vol"],
        )
        out = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(df["t"].astype("int64"), unit="ms"),
                "open": df["o"].astype(float),
                "high": df["h"].astype(float),
                "low": df["l"].astype(float),
                "close": df["c"].astype(float),
                "volume": df["vol"].astype(float),
            }
        )
        out = out[out["timestamp"] >= pd.Timestamp(start_ms, unit="ms")]
        return (
            out.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _cache_path(self, inst_id: str, timeframe: str) -> Path:
        return self.cache_dir / f"{inst_id}_{timeframe}.csv"

    def _load_candles(self, symbol: str, timeframe: str) -> pd.DataFrame:
        inst_id = self._inst_id(symbol)
        bar = _OKX_BAR_MAP.get(timeframe, timeframe)
        cache_key = f"{inst_id}_{timeframe}"
        if cache_key in self._kline_cache:
            return self._kline_cache[cache_key]
        path = self._cache_path(inst_id, timeframe)
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
        else:
            df = self._fetch_candles(inst_id, bar)
            if not df.empty:
                df.to_csv(path, index=False)
        self._kline_cache[cache_key] = df
        return df

    def load_klines(self, symbol, timeframe="1h", start=None, end=None):
        df = self._load_candles(symbol, timeframe)
        if df.empty:
            return df
        if start is not None:
            df = df[df["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)

    def load_funding(self, symbol, start=None, end=None):
        inst_id = self._inst_id(symbol)
        cache_key = f"{inst_id}_funding"
        path = self.cache_dir / f"{inst_id}_funding.csv"
        if cache_key in self._kline_cache:
            df = self._kline_cache[cache_key]
        elif path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            self._kline_cache[cache_key] = df
        else:
            start_ms = int((self.end - pd.Timedelta(days=self.days)).timestamp() * 1000)
            cursor_after = int(self.end.timestamp() * 1000) + 1
            rows: list = []
            for _ in range(400):
                data = _okx_get(
                    "/api/v5/public/funding-rate-history",
                    {
                        "instId": inst_id,
                        "after": str(cursor_after),
                        "limit": "100",
                    },
                )
                if not data:
                    break
                rows.extend(data)
                oldest_t = int(data[-1]["fundingTime"])
                if oldest_t >= cursor_after:
                    break
                cursor_after = oldest_t
                if oldest_t <= start_ms:
                    break
            if rows:
                df = (
                    pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(
                                [int(r["fundingTime"]) for r in rows],
                                unit="ms",
                            ),
                            "funding_rate": [float(r["fundingRate"]) for r in rows],
                        }
                    )
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
                df = df[df["timestamp"] >= pd.Timestamp(start_ms, unit="ms")]
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


_DEFAULT_PG_DSN = "postgresql://trade_rl:trade_rl@localhost:5433/trade_rl"


class PostgresSource(DataSource):
    """
    PostgreSQL（rl_ プレフィックステーブル）ソース

    scripts/fetch_futures.py --to postgres 等（mars_lite.data.postgres_store）
    が書き込む rl_klines / rl_funding_rate / rl_orderflow_1m / rl_derivatives
    を読む。各テーブルは (source, symbol, ...) で取得元を区別する
    （例: source="binance"）。rl_klines は1分足のみ保存されるため、
    1h/4h/1d等への変換は resample_ohlcv で行う。

    接続文字列は dsn 引数、無ければ環境変数 PLATFORM_DB_URL、
    どちらも無ければ docker-compose.yml のローカル既定値を使う。
    """

    def __init__(
        self,
        symbols: List[str],
        dsn: Optional[str] = None,
        source: str = "binance",
        derivatives_source: Optional[str] = None,
        orderflow_source: Optional[str] = None,
    ):
        super().__init__(symbols)
        self.dsn = dsn or os.environ.get("PLATFORM_DB_URL", _DEFAULT_PG_DSN)
        self.source = source
        # klines/fundingはネイティブ取得元（例: hyperliquid）、
        # OI/L-S/清算/オーダーフローはBinance代理で埋めるケースが多いため
        # 別のsourceラベルを指定できるようにする（省略時はsourceと同じ）
        self.derivatives_source = derivatives_source or source
        self.orderflow_source = orderflow_source or source

    # 生データ（klines/orderflow/derivatives/funding）はDB上の履歴が確定済みで
    # 事後に書き換わらない（fetch_futures.py等は追記のみ）ため、SQL文＋paramsを
    # キーにディスクキャッシュしてよい。ボラティリティ系指標などfeature_pipeline側の
    # 計算（_z/ichimoku/vol_ratio等）はこのキャッシュの対象外＝毎回この生データから
    # 再計算される（安く、コード変更に追従できる）。
    # 注意: start/end省略（DB全期間取得）のクエリをキャッシュした後にDBへ新しい
    # 日付を追加投入した場合、このキャッシュは古い（狭い）範囲のまま返る。
    # 再取得後に反映したい場合はキャッシュディレクトリを削除するか
    # PG_SOURCE_CACHE_DIR="" で無効化して実行すること。
    _cache_dir_env = "PG_SOURCE_CACHE_DIR"
    _default_cache_dir = "./data/.pg_cache"

    def _query(self, sql: str, params: list) -> pd.DataFrame:
        import psycopg

        cache_dir = os.environ.get(self._cache_dir_env, self._default_cache_dir)
        cache_path = None
        if cache_dir:
            key = hashlib.sha1(f"{self.dsn}|{sql}|{params}".encode("utf-8")).hexdigest()
            cache_path = Path(cache_dir) / f"{key}.parquet"
            if cache_path.exists():
                return pd.read_parquet(cache_path)

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(
                None
            )

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
        return df

    def _time_clauses(self, start, end):
        clauses, params = [], []
        if start is not None:
            clauses.append("timestamp >= %s")
            params.append(pd.Timestamp(start))
        if end is not None:
            clauses.append("timestamp <= %s")
            params.append(pd.Timestamp(end))
        return clauses, params

    def load_klines(self, symbol, timeframe="1h", start=None, end=None):
        """
        rl_klinesはtimeframe列を持つため、まず要求TFそのものを探す
        （Hyperliquidネイティブ取得は15m/1h/4h/1dを直接保存）。
        無ければ1m足から resample_ohlcv でリサンプルする
        （Binance取得は1mのみ保存）。
        """
        clauses, params = self._time_clauses(start, end)
        where = " AND ".join(["source = %s", "symbol = %s", "timeframe = %s"] + clauses)
        sql = (
            f"SELECT timestamp, open, high, low, close, volume FROM rl_klines "
            f"WHERE {where} ORDER BY timestamp"
        )
        df = self._query(sql, [self.source, symbol, timeframe] + params)
        if not df.empty:
            return df.reset_index(drop=True)

        if timeframe == "1m":
            return df
        df = self._query(sql, [self.source, symbol, "1m"] + params)
        if df.empty:
            return df
        return resample_ohlcv(df, timeframe).reset_index(drop=True)

    def load_orderflow(self, symbol, start=None, end=None):
        clauses, params = self._time_clauses(start, end)
        where = " AND ".join(["source = %s", "symbol = %s"] + clauses)
        sql = (
            f"SELECT timestamp, buy_volume, sell_volume, trade_count, "
            f"avg_trade_size, volume_imbalance FROM rl_orderflow_1m "
            f"WHERE {where} ORDER BY timestamp"
        )
        df = self._query(sql, [self.orderflow_source, symbol] + params)
        if df.empty:
            return super().load_orderflow(symbol, start, end)
        return df.reset_index(drop=True)

    def load_derivatives(self, symbol, start=None, end=None):
        clauses, params = self._time_clauses(start, end)
        where = " AND ".join(["source = %s", "symbol = %s"] + clauses)
        sql = (
            f"SELECT timestamp, open_interest, ls_ratio, liq_notional "
            f"FROM rl_derivatives WHERE {where} ORDER BY timestamp"
        )
        df = self._query(sql, [self.derivatives_source, symbol] + params)
        if df.empty:
            return super().load_derivatives(symbol, start, end)
        return df.reset_index(drop=True)

    def load_funding(self, symbol, start=None, end=None):
        clauses, params = self._time_clauses(start, end)
        where = " AND ".join(["source = %s", "symbol = %s"] + clauses)
        sql = (
            f"SELECT timestamp, funding_rate FROM rl_funding_rate "
            f"WHERE {where} ORDER BY timestamp"
        )
        df = self._query(sql, [self.source, symbol] + params)
        if df.empty:
            return super().load_funding(symbol, start, end)
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
    if name == "bitget":
        return BitgetSource(symbols, **kwargs)
    if name == "okx":
        return OKXSource(symbols, **kwargs)
    if name == "postgres":
        return PostgresSource(symbols, **kwargs)
    raise ValueError(f"unknown source: {name}")
