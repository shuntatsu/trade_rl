"""
特徴量パイプラインモジュール

DataSource から銘柄×多時間軸の特徴量テンソルを構築する。

設計:
- 意思決定は基準TF（デフォルト1h）ごと。観測には複数TF（15m/1h/4h/1d）の
  特徴ブロックを持つ
- 各TFの特徴は「そのTF上で」計算し、**確定済みバーのみ**を merge_asof で
  意思決定時刻に整列（look-ahead防止）
- 全特徴はローリングz-scoreまたは有界変換で正規化（クリップ±5）

出力 FeatureSet:
    features        (n_bars, n_symbols, n_features)
    global_features (n_bars, n_global)
    close / open_next (n_bars, n_symbols)  環境のPnL・執行計算用
    funding_rate    (n_bars, n_symbols)    そのバーで授受されるfunding率合計
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mars_lite.data.sources import DataSource
from mars_lite.data.data_utils import TF_TO_MINUTES
from mars_lite.data.volatility import calc_garman_klass
from mars_lite.features.indicators import calc_rsi, calc_adx, calc_rci

CLIP = 5.0


def _ns(x):
    """datetime系列をdatetime64[ns]に統一（merge_asofのdtype不一致回避）"""
    return pd.to_datetime(x).astype("datetime64[ns]")

# TFブロックの特徴（各TF上で計算）
TF_BLOCK_FEATURES = ["ret_z1", "ret_z5", "ret_z20", "vol_ratio", "rsi", "bb_pos", "adx"]
# 基準TFのみの追加特徴
BASE_FEATURES = [
    "vol_anom", "rci",
    "of_imbalance", "of_count_z", "of_size_z",
    "funding_bps", "funding_cum_bps", "time_to_funding",
    "btc_rel_z", "ret_rank",
]
GLOBAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "btc_vol_regime"]


@dataclass
class FeatureSet:
    """環境・評価・推論が共有する特徴量データセット"""
    symbols: List[str]
    timestamps: np.ndarray            # (n_bars,) 基準TFバーの開始時刻
    features: np.ndarray              # (n_bars, n_symbols, n_features)
    global_features: np.ndarray       # (n_bars, n_global)
    close: np.ndarray                 # (n_bars, n_symbols)
    open_next: np.ndarray             # (n_bars, n_symbols) 次バー始値
    funding_rate: np.ndarray          # (n_bars, n_symbols)
    feature_names: List[str] = field(default_factory=list)
    global_feature_names: List[str] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return len(self.timestamps)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_features(self) -> int:
        return self.features.shape[2]

    def slice(self, start_idx: int, end_idx: int) -> "FeatureSet":
        """バー範囲でスライスした新しいFeatureSetを返す"""
        return FeatureSet(
            symbols=self.symbols,
            timestamps=self.timestamps[start_idx:end_idx],
            features=self.features[start_idx:end_idx],
            global_features=self.global_features[start_idx:end_idx],
            close=self.close[start_idx:end_idx],
            open_next=self.open_next[start_idx:end_idx],
            funding_rate=self.funding_rate[start_idx:end_idx],
            feature_names=self.feature_names,
            global_feature_names=self.global_feature_names,
        )


def _z(series: pd.Series, window: int = 100, min_periods: int = 20) -> pd.Series:
    """ローリングz-score（過去窓のみ・クリップ±CLIP・NaN→0）"""
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-CLIP, CLIP).fillna(0.0)


def _tf_block(df: pd.DataFrame) -> pd.DataFrame:
    """単一TFのOHLCVからTFブロック特徴を計算"""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    log_ret = np.log(close / close.shift(1))

    out["ret_z1"] = _z(log_ret)
    out["ret_z5"] = _z(log_ret.rolling(5, min_periods=5).sum())
    out["ret_z20"] = _z(log_ret.rolling(20, min_periods=20).sum())

    vol = calc_garman_klass(df)
    vol_short = vol.rolling(20, min_periods=5).mean()
    vol_long = vol.rolling(100, min_periods=20).mean()
    out["vol_ratio"] = np.log(
        (vol_short / vol_long.replace(0, np.nan)).clip(0.1, 10)
    ).fillna(0.0)

    out["rsi"] = ((calc_rsi(close, 14) - 50) / 25).clip(-CLIP, CLIP).fillna(0.0)

    ma20 = close.rolling(20, min_periods=20).mean()
    sd20 = close.rolling(20, min_periods=20).std()
    out["bb_pos"] = ((close - ma20) / (2 * sd20.replace(0, np.nan))).clip(-CLIP, CLIP).fillna(0.0)

    out["adx"] = ((calc_adx(df["high"], df["low"], close, 14) - 25) / 25).clip(-CLIP, CLIP).fillna(0.0)

    return out


class FeaturePipeline:
    """DataSource → FeatureSet 変換器"""

    def __init__(
        self,
        symbols: List[str],
        base_timeframe: str = "1h",
        timeframes: Optional[List[str]] = None,
        z_window: int = 100,
    ):
        self.symbols = list(symbols)
        self.base_tf = base_timeframe
        self.timeframes = timeframes or ["15m", "1h", "4h", "1d"]
        if self.base_tf not in self.timeframes:
            self.timeframes = [self.base_tf] + self.timeframes
        self.z_window = z_window

        self.feature_names = (
            [f"{tf}_{f}" for tf in self.timeframes for f in TF_BLOCK_FEATURES]
            + BASE_FEATURES
        )
        self.global_feature_names = list(GLOBAL_FEATURES)

    # ---- 内部処理 ----

    def _base_frame(self, source: DataSource, symbol: str, start, end) -> pd.DataFrame:
        """基準TFのOHLCV＋基準TF特徴（インデックス=バー開始時刻）"""
        df = source.load_klines(symbol, self.base_tf, start, end)
        if df.empty:
            raise ValueError(f"No klines for {symbol} ({self.base_tf})")
        df = df.set_index("timestamp")

        base = pd.DataFrame(index=df.index)
        base["open"] = df["open"]
        base["close"] = df["close"]
        base["log_ret"] = np.log(df["close"] / df["close"].shift(1))

        # 出来高異常度
        vol_ma = df["volume"].rolling(self.z_window, min_periods=20).mean()
        base["vol_anom"] = np.log(
            (df["volume"] / vol_ma.replace(0, np.nan)).clip(0.05, 20)
        ).fillna(0.0)
        base["rci"] = (calc_rci(df["close"], 9) / 100).clip(-1, 1).fillna(0.0)
        return base

    def _align_tf_features(
        self, source: DataSource, symbol: str, tf: str,
        decision_times: pd.DatetimeIndex, start, end,
    ) -> pd.DataFrame:
        """
        TFブロック特徴を意思決定時刻に整列

        バー開始時刻 + TF長 = バー確定時刻。確定時刻 <= 意思決定時刻 の
        最新バーを merge_asof(backward) で対応付ける。
        """
        df = source.load_klines(symbol, tf, start, end)
        if df.empty:
            return pd.DataFrame(
                0.0, index=decision_times,
                columns=[f"{tf}_{f}" for f in TF_BLOCK_FEATURES],
            )
        feats = _tf_block(df.set_index("timestamp")).reset_index()
        feats["complete_time"] = _ns(feats["timestamp"] + pd.Timedelta(minutes=TF_TO_MINUTES[tf]))
        feats = feats.drop(columns=["timestamp"]).sort_values("complete_time")

        aligned = pd.merge_asof(
            pd.DataFrame({"decision_time": _ns(decision_times)}),
            feats, left_on="decision_time", right_on="complete_time",
            direction="backward",
        ).drop(columns=["complete_time"]).set_index("decision_time")
        aligned.columns = [f"{tf}_{c}" for c in aligned.columns]
        return aligned.fillna(0.0)

    def _orderflow_features(
        self, source: DataSource, symbol: str,
        bar_index: pd.DatetimeIndex, start, end,
    ) -> pd.DataFrame:
        """オーダーフロー1m集計 → 基準TFに再集計して特徴化"""
        cols = ["of_imbalance", "of_count_z", "of_size_z"]
        of = source.load_orderflow(symbol, start, end)
        if of.empty:
            return pd.DataFrame(0.0, index=bar_index, columns=cols)

        of = of.set_index("timestamp")
        freq = f"{TF_TO_MINUTES[self.base_tf]}min"
        agg = of.resample(freq).agg({
            "buy_volume": "sum", "sell_volume": "sum",
            "trade_count": "sum", "avg_trade_size": "mean",
        })
        total = agg["buy_volume"] + agg["sell_volume"]

        out = pd.DataFrame(index=agg.index)
        out["of_imbalance"] = np.where(
            total > 0, (agg["buy_volume"] - agg["sell_volume"]) / total, 0.0
        )
        out["of_count_z"] = _z(agg["trade_count"], self.z_window)
        out["of_size_z"] = _z(agg["avg_trade_size"], self.z_window)
        return out.reindex(bar_index).fillna(0.0)

    def _funding_features(
        self, source: DataSource, symbol: str,
        bar_index: pd.DatetimeIndex, start, end,
    ):
        """funding特徴と、バーごとの授受funding率配列を計算"""
        cols = ["funding_bps", "funding_cum_bps", "time_to_funding"]
        fr = source.load_funding(symbol, start, end)

        bar_minutes = TF_TO_MINUTES[self.base_tf]
        hours = pd.Series(bar_index.hour, index=bar_index).astype(float)
        # 次の8時間funding境界（0/8/16 UTC）までの残時間（0〜1に正規化）
        time_to = ((8 - (hours % 8)) % 8) / 8.0

        if fr.empty:
            feats = pd.DataFrame(0.0, index=bar_index, columns=cols)
            feats["time_to_funding"] = time_to
            per_bar = pd.Series(0.0, index=bar_index)
            return feats, per_bar

        fr = fr.set_index("timestamp")["funding_rate"]

        # 意思決定時刻時点で判明している最新funding率（バー確定時刻で整列）
        decision_times = bar_index + pd.Timedelta(minutes=bar_minutes)
        fr_df = fr.reset_index().rename(columns={"timestamp": "ft"})
        fr_df["ft"] = _ns(fr_df["ft"])
        latest = pd.merge_asof(
            pd.DataFrame({"t": _ns(decision_times)}),
            fr_df, left_on="t", right_on="ft", direction="backward",
        )["funding_rate"].to_numpy()

        feats = pd.DataFrame(index=bar_index)
        feats["funding_bps"] = np.clip(np.nan_to_num(latest) * 1e4, -CLIP, CLIP)
        cum = fr.rolling("24h").sum()
        cum_df = cum.reset_index().rename(columns={"timestamp": "ft"})
        cum_df["ft"] = _ns(cum_df["ft"])
        cum_aligned = pd.merge_asof(
            pd.DataFrame({"t": _ns(decision_times)}),
            cum_df, left_on="t", right_on="ft", direction="backward",
        )["funding_rate"].to_numpy()
        feats["funding_cum_bps"] = np.clip(np.nan_to_num(cum_aligned) * 1e4, -CLIP, CLIP)
        feats["time_to_funding"] = time_to.to_numpy()

        # このバー内（(T, T+bar]）に発生するfunding授受の合計
        events = fr.copy()
        bucket = (events.index - pd.Timedelta(seconds=1)).floor(f"{bar_minutes}min")
        per_bar = events.groupby(bucket).sum().reindex(bar_index).fillna(0.0)
        return feats, per_bar

    # ---- 公開API ----

    def build(
        self, source: DataSource,
        start: Optional[str] = None, end: Optional[str] = None,
    ) -> FeatureSet:
        """DataSourceからFeatureSetを構築"""
        bar_minutes = TF_TO_MINUTES[self.base_tf]

        # 基準フレーム（銘柄ごと）→ 共通タイムスタンプでinner join
        bases = {s: self._base_frame(source, s, start, end) for s in self.symbols}
        common = None
        for s, b in bases.items():
            common = b.index if common is None else common.intersection(b.index)
        common = common.sort_values()
        if len(common) < 50:
            raise ValueError(f"Too few common bars across symbols: {len(common)}")

        bases = {s: b.reindex(common) for s, b in bases.items()}
        decision_times = common + pd.Timedelta(minutes=bar_minutes)

        # クロスセクション用の全銘柄リターン行列
        ret_mat = pd.DataFrame({s: bases[s]["log_ret"] for s in self.symbols})
        ret_24h = ret_mat.rolling(max(1, 1440 // bar_minutes), min_periods=1).sum()
        rank = ret_24h.rank(axis=1, pct=True) - 0.5  # [-0.5, 0.5]

        btc_col = next((s for s in self.symbols if s.upper().startswith("BTC")), self.symbols[0])
        btc_ret = ret_mat[btc_col]

        n_bars, n_sym = len(common), len(self.symbols)
        features = np.zeros((n_bars, n_sym, len(self.feature_names)), dtype=np.float32)
        funding_arr = np.zeros((n_bars, n_sym), dtype=np.float32)
        close_arr = np.zeros((n_bars, n_sym), dtype=np.float64)
        open_next_arr = np.zeros((n_bars, n_sym), dtype=np.float64)

        for i, sym in enumerate(self.symbols):
            blocks = []
            for tf in self.timeframes:
                blk = self._align_tf_features(source, sym, tf, decision_times, start, end)
                blk.index = common
                blocks.append(blk)

            base_feats = pd.DataFrame(index=common)
            base_feats["vol_anom"] = bases[sym]["vol_anom"]
            base_feats["rci"] = bases[sym]["rci"]

            of = self._orderflow_features(source, sym, common, start, end)
            fund_feats, per_bar_funding = self._funding_features(source, sym, common, start, end)
            rel = _z(bases[sym]["log_ret"] - btc_ret, self.z_window)

            base_feats = pd.concat([base_feats, of, fund_feats], axis=1)
            base_feats["btc_rel_z"] = rel
            base_feats["ret_rank"] = rank[sym]

            full = pd.concat(blocks + [base_feats], axis=1)[self.feature_names]
            features[:, i, :] = np.nan_to_num(
                full.to_numpy(dtype=np.float32), posinf=CLIP, neginf=-CLIP
            )
            funding_arr[:, i] = per_bar_funding.to_numpy(dtype=np.float32)
            close_arr[:, i] = bases[sym]["close"].to_numpy()
            open_arr = bases[sym]["open"].to_numpy()
            open_next_arr[:, i] = np.concatenate([open_arr[1:], open_arr[-1:]])

        # グローバル特徴
        hours = common.hour.to_numpy()
        dows = common.dayofweek.to_numpy()
        btc_vol = _z(
            calc_garman_klass(
                pd.DataFrame({
                    "open": bases[btc_col]["open"], "close": bases[btc_col]["close"],
                    "high": bases[btc_col]["close"], "low": bases[btc_col]["open"],
                }).assign(
                    high=lambda d: d[["open", "close"]].max(axis=1) * 1.0001,
                    low=lambda d: d[["open", "close"]].min(axis=1) * 0.9999,
                )
            ).rolling(24, min_periods=4).mean(),
            self.z_window,
        )
        global_features = np.stack([
            np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
            np.sin(2 * np.pi * dows / 7), np.cos(2 * np.pi * dows / 7),
            btc_vol.to_numpy(),
        ], axis=1).astype(np.float32)

        return FeatureSet(
            symbols=self.symbols,
            timestamps=common.to_numpy(),
            features=features,
            global_features=np.nan_to_num(global_features),
            close=close_arr,
            open_next=open_next_arr,
            funding_rate=funding_arr,
            feature_names=self.feature_names,
            global_feature_names=self.global_feature_names,
        )
