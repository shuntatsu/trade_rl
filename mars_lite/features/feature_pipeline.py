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
from typing import List, Optional

import numpy as np
import pandas as pd

from mars_lite.data.data_utils import TF_TO_MINUTES
from mars_lite.data.sources import DataSource
from mars_lite.data.volatility import calc_garman_klass
from mars_lite.features.ichimoku import ichimoku_features
from mars_lite.features.indicators import calc_adx, calc_cci, calc_rci, calc_rsi

CLIP = 5.0


def _ns(x):
    """datetime系列をdatetime64[ns]に統一（merge_asofのdtype不一致回避）"""
    return pd.to_datetime(x).astype("datetime64[ns]")


# TFブロックの特徴（各TF上で計算）
# cci_divergence: 価格とCCIのレンジ内相対位置の乖離（全時間軸で計算）
TF_BLOCK_FEATURES = [
    "ret_z1",
    "ret_z5",
    "ret_z20",
    "vol_ratio",
    "rsi",
    "bb_pos",
    "adx",
    "cci_divergence",
]
# サポート/レジスタンス特徴（1h/4h/1d上で計算。Nバー・ドンチャンレンジ基準）
# 「平均からの乖離」であるret_z/bb_posとは異なり、過去の高値・安値という
# 水平レベルの記憶（ブレイクアウト接近度・レベルの鮮度）を持たせる
SR_BLOCK_FEATURES = [
    "dist_high",
    "dist_low",
    "range_pos",
    "bars_since_high",
    "bars_since_low",
    "rsi_divergence",
]
SR_WINDOW = 55
# 一目均衡表の追加TFブロック（4h/1d）。1h版は BASE_FEATURES に10種フルセットで
# 含まれるため、4h/1dは主要5種に絞って追加する
ICHI_TF_FEATURES = [
    "ichi_pos",
    "ichi_tk_cross",
    "ichi_price_kijun",
    "ichi_future_pos",
    "ichi_future_bull",
]
# 基準TFのみの追加特徴
# ret_z48/ret_z168: 多時間軸実験の知見（多スケール情報はTFブロックの積み増しより
# 基準TF上の多ホライズン特徴が効率的に運ぶ）に基づく長期ホライズンリターン
# oi_*/ls_*/liq_*: デリバティブ指標（建玉残高・ロングショート比率・清算）
BASE_FEATURES = [
    "vol_anom",
    "rci",
    "ret_z48",
    "ret_z168",
    "of_imbalance",
    "of_count_z",
    "of_size_z",
    "funding_bps",
    "funding_cum_bps",
    "time_to_funding",
    "oi_z",
    "oi_change",
    "ls_ratio_z",
    "liq_z",
    "btc_rel_z",
    "ret_rank",
    # 一目均衡表（look-ahead なし: senkou は shift(26) 済み、chikou は除外）
    # 「現在雲」: 26本前の予測が今届いた雲のサポート/レジスタンスとの位置関係
    "ichi_pos",
    "ichi_cloud_thick",
    "ichi_cloud_bull",
    "ichi_tk_cross",
    "ichi_price_kijun",
    "ichi_price_tenkan",
    # 「未来雲予測」: t時点のデータから t+26 の雲構造を予測（look-ahead なし）
    # エージェントが「26本先の相場構造」を学習できる
    "ichi_future_pos",
    "ichi_future_bull",
    "ichi_future_thick",
    "ichi_tk_accel",
]
# Phase C2: クロスセクション正規化特徴
# ターゲットが cs_demean（銘柄間平均を引いた市場中立リターン）のとき、
# 特徴量側も同じクロスセクション軸で正規化しないと「絶対モメンタム」と
# 「相対アルファ」を混在させた状態で回帰することになり fit が散漫になる。
# 最も直接的な効果が期待できる：リターン系・OI・L/S比率・Fundingに絞って追加。
CS_FEATURES = [
    "csz_ret1",
    "csz_ret5",
    "csz_ret20",
    "csz_oi",
    "csz_ls",
    "csz_funding",
]
GLOBAL_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "btc_vol_regime",
    "btc_trend",
]


@dataclass
class FeatureSet:
    """環境・評価・推論が共有する特徴量データセット"""

    symbols: List[str]
    timestamps: np.ndarray  # (n_bars,) 基準TFバーの開始時刻
    features: np.ndarray  # (n_bars, n_symbols, n_features)
    global_features: np.ndarray  # (n_bars, n_global)
    close: np.ndarray  # (n_bars, n_symbols)
    open_next: np.ndarray  # (n_bars, n_symbols) 次バー始値
    funding_rate: np.ndarray  # (n_bars, n_symbols)
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

    def apply_mask(self, mask) -> "FeatureSet":
        """
        特徴マスクを適用した新しいFeatureSetを返す（マスク外はゼロ埋め）

        次元・レイアウトは維持されるため、モデル構造は変わらない。
        学習時に使ったマスクは推論時にも同一適用すること。
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.n_features:
            raise ValueError(
                f"mask length {mask.shape[0]} != n_features {self.n_features}"
            )
        masked = self.features.copy()
        masked[:, :, ~mask] = 0.0
        return FeatureSet(
            symbols=self.symbols,
            timestamps=self.timestamps,
            features=masked,
            global_features=self.global_features,
            close=self.close,
            open_next=self.open_next,
            funding_rate=self.funding_rate,
            feature_names=self.feature_names,
            global_feature_names=self.global_feature_names,
        )

    def gaussian_rank_normalized(
        self, window: int = 250, min_periods: int = 40
    ) -> "FeatureSet":
        """
        各特徴チャネルをローリング・ガウスランク正規化した新FeatureSetを返す。

        目的（汎用性）: 生の特徴は資産・レジームによってスケール・裾の重さが
        大きく異なる。過去 window バー内での順位 → 分位点 → 逆正規CDF で写像
        すると、どの銘柄・どの時期でも各入力チャネルが概ね標準正規 N(0,1) に
        従うようになり、モデルが「特定資産の特定スケール」に過適合しにくくなる。
        外れ値バーの支配も自動的に抑えられる（順位変換のため単調・裾に頑健）。

        因果的: 各時刻 t の値は「t 以前の window バー」内での順位のみで決まる
        （trailing window、未来を参照しない）。price/funding は変換しない
        （損益計算の整合を保つため、入力特徴のみ正規化する）。

        window/min_periods は学習・推論で同一に適用すること（モデルメタデータに
        記録）。ゼロ埋め特徴（crypto固有指標が無い資産など）は分散ゼロなので
        変換後も 0 のまま。
        """
        new_features = _gaussian_rank_transform(self.features, window, min_periods)
        return FeatureSet(
            symbols=self.symbols,
            timestamps=self.timestamps,
            features=new_features,
            global_features=self.global_features,
            close=self.close,
            open_next=self.open_next,
            funding_rate=self.funding_rate,
            feature_names=self.feature_names,
            global_feature_names=self.global_feature_names,
        )

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


def _gaussian_rank_transform(
    features: np.ndarray,
    window: int = 250,
    min_periods: int = 40,
) -> np.ndarray:
    """
    (n_bars, n_symbols, n_features) の各(symbol,feature)チャネルを
    ローリング・ガウスランク正規化する（因果的・ベクトル化）。

    各時刻 t の値を、trailing window 内での順位 rank(1..cnt) から
    分位点 q=(rank-0.5)/cnt を作り、逆正規CDFで N(0,1) に写像する。
    分散ゼロ（定数/ゼロ埋め）チャネルは順位が縮退するため 0 に落ちる。
    """
    from mars_lite.utils.metrics import norm_ppf_array

    n_bars, n_sym, n_feat = features.shape
    out = np.zeros_like(features, dtype=np.float64)
    for i in range(n_sym):
        for j in range(n_feat):
            s = pd.Series(features[:, i, j])
            if s.nunique(dropna=True) <= 1:
                continue  # 定数/ゼロ埋めチャネルは 0 のまま
            rank = s.rolling(window, min_periods=min_periods).rank()
            cnt = s.rolling(window, min_periods=min_periods).count()
            q = (rank - 0.5) / cnt.replace(0, np.nan)
            z = norm_ppf_array(q.to_numpy())
            out[:, i, j] = np.nan_to_num(z, nan=0.0)
    return out


def _cs_z(mat: pd.DataFrame, window: int = 100, min_periods: int = 5) -> pd.DataFrame:
    """
    クロスセクションz-score（バー×銘柄行列を入力）

    各バーtにおいて、「そのバーの全銘柄値の中での各銘柄の相対位置」を測る。

    look-ahead防止のため、正規化パラメータ（mu, sigma）は
    「過去window本のバーのクロスセクション値（全銘柄×window本）の
    プール統計」から推定する。これにより：
    - 未来情報を使わない（各バーは過去分でのみ推定）
    - バー1本だけの銘柄数が少ない問題を回避（窓内の全データをプール）
    - 市場全体のレベルシフト（上昇相場など）を吸収して相対的な位置を返す
    """
    arr = mat.to_numpy(dtype=np.float64)
    n_bars, n_sym = arr.shape
    out = np.zeros_like(arr)
    for t in range(n_bars):
        lo = max(0, t - window + 1)
        if t - lo + 1 < min_periods:
            continue
        # 過去window本のクロスセクション値を全部プール
        pool = arr[lo : t + 1].ravel()  # shape: (window * n_sym,)
        pool = pool[np.isfinite(pool)]
        if len(pool) < min_periods:
            continue
        mu = np.mean(pool)
        sigma = np.std(pool)
        if sigma > 1e-12:
            out[t] = np.clip((arr[t] - mu) / sigma, -CLIP, CLIP)
    return pd.DataFrame(out, index=mat.index, columns=mat.columns)


def _swing_divergence(
    price: pd.Series,
    osc: pd.Series,
    pivot_margin: int = 3,
    min_swing_gap: int = 8,
    memory: int = 3,
) -> pd.Series:
    """
    スイング確認型ダイバージェンス（look-aheadなし、確定ラグつき）

    直近の確定済みスイング高値どうし／安値どうしを比較し、直近memory件の
    判定を「弱気ダイバージェンス優勢(+1)」〜「強気ダイバージェンス優勢(-1)」の
    連続値として返す。次の確定イベントまで値を保持する（1回の局所的なノイズで
    符号が反転しない）。

    スイング確定にはpivot_margin本先のデータが必要（前後を見て初めて
    「あれが山/谷だった」と分かる）ため、確定情報はそのpivot_margin分だけ
    後ろにシフトして使う（一目均衡表のsenkou spanと同じ確定ラグの手法。
    ichimoku.py参照）。min_swing_gap未満の間隔で並ぶスイングは同一スイングの
    延長とみなし、より極端な値に更新するだけで比較対象にしない（微小ノイズの統合）。
    """
    n = len(price)
    idx = price.index
    price_arr = price.to_numpy(dtype=np.float64)
    osc_arr = osc.to_numpy(dtype=np.float64)

    window = 2 * pivot_margin + 1
    is_high_raw = np.zeros(n, dtype=bool)
    is_low_raw = np.zeros(n, dtype=bool)
    if n >= window:
        wp = np.lib.stride_tricks.sliding_window_view(price_arr, window)
        center = price_arr[pivot_margin : n - pivot_margin]
        is_high_raw[pivot_margin : n - pivot_margin] = center == wp.max(axis=1)
        is_low_raw[pivot_margin : n - pivot_margin] = center == wp.min(axis=1)

    def _confirmed_events(is_pivot_raw: np.ndarray) -> list:
        swing_times = np.where(is_pivot_raw)[0]
        confirm_times = swing_times + pivot_margin
        keep = confirm_times < n
        return list(zip(confirm_times[keep].tolist(), swing_times[keep].tolist()))

    def _apply(events: list, is_high: bool) -> np.ndarray:
        out = np.zeros(n, dtype=np.float64)
        history: list = []  # (swing_t, price_val, osc_val) 確定済みスイング
        recent: list = []  # 直近memory件のダイバージェンス判定(0/1)
        for confirm_t, swing_t in events:
            p_val, o_val = price_arr[swing_t], osc_arr[swing_t]
            if history and (swing_t - history[-1][0]) < min_swing_gap:
                # 直前スイングに近すぎる -> 同一スイングの延長として統合(比較しない)
                more_extreme = (
                    p_val > history[-1][1] if is_high else p_val < history[-1][1]
                )
                if more_extreme:
                    history[-1] = (swing_t, p_val, o_val)
                continue
            if history:
                _, prev_p, prev_o = history[-1]
                if is_high:
                    div = 1.0 if (p_val > prev_p and o_val < prev_o) else 0.0
                else:
                    div = 1.0 if (p_val < prev_p and o_val > prev_o) else 0.0
                recent.append(div)
                if len(recent) > memory:
                    recent.pop(0)
                out[confirm_t:] = float(np.mean(recent))
            history.append((swing_t, p_val, o_val))
        return out

    bearish_pressure = _apply(_confirmed_events(is_high_raw), is_high=True)
    bullish_pressure = _apply(_confirmed_events(is_low_raw), is_high=False)
    return pd.Series(bearish_pressure - bullish_pressure, index=idx).clip(-CLIP, CLIP)


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
    out["bb_pos"] = (
        ((close - ma20) / (2 * sd20.replace(0, np.nan))).clip(-CLIP, CLIP).fillna(0.0)
    )

    out["adx"] = (
        ((calc_adx(df["high"], df["low"], close, 14) - 25) / 25)
        .clip(-CLIP, CLIP)
        .fillna(0.0)
    )

    # スイング確認型ダイバージェンス（価格とCCI、全TF共通）。_swing_divergence参照
    cci = calc_cci(df["high"], df["low"], close, period=20)
    out["cci_divergence"] = _swing_divergence(close, cci)

    return out


def _bars_since_extreme(series: pd.Series, window: int, mode: str) -> pd.Series:
    """
    直近window本のローリング窓内で高値/安値をつけてから何本経過したか（0=当該バーが極値）。
    ウォームアップ期間（先頭window-1本）はNaN。過去窓のみ使用のためlook-aheadなし。
    """
    arr = series.to_numpy(dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan)
    if n >= window:
        windows = np.lib.stride_tricks.sliding_window_view(arr, window)
        idx = (
            np.argmax(windows, axis=1) if mode == "high" else np.argmin(windows, axis=1)
        )
        out[window - 1 :] = (window - 1) - idx
    return pd.Series(out, index=series.index)


def _sr_block(df: pd.DataFrame, window: int = SR_WINDOW) -> pd.DataFrame:
    """
    サポート/レジスタンス特徴（単一TFのOHLCから計算、look-aheadなし）

    ret_z/bb_posが「移動平均からの乖離」であるのに対し、これはNバーの
    ドンチャンレンジという「過去の高値・安値の記憶」を直接特徴化する:
      dist_high/dist_low : ブレイクアウトまでの距離（レジスタンス/サポートの近さ）
      range_pos           : レンジ内の相対位置（0=安値張り付き〜1=高値張り付き、-1〜1に変換）
      bars_since_high/low : そのレベルを付けてからの経過本数（鮮度。z-score化して正規化）
      rsi_divergence      : 価格とRSIの背離（下記）
    """
    high, low, close = df["high"], df["low"], df["close"]
    roll_high = high.rolling(window, min_periods=window).max()
    roll_low = low.rolling(window, min_periods=window).min()
    rng = (roll_high - roll_low).replace(0, np.nan)
    safe_close = close.replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out["dist_high"] = _z((roll_high - close) / safe_close)
    out["dist_low"] = _z((close - roll_low) / safe_close)
    price_pos = (close - roll_low) / rng
    out["range_pos"] = _z((price_pos - 0.5) * 2)
    out["bars_since_high"] = _z(_bars_since_extreme(high, window, "high") / window)
    out["bars_since_low"] = _z(_bars_since_extreme(low, window, "low") / window)

    # スイング確認型ダイバージェンス（価格とRSI）。_swing_divergence参照
    out["rsi_divergence"] = _swing_divergence(close, calc_rsi(close, 14))
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

        # サポレジ特徴を持たせるTF（15mはTFブロックのz1/5/20で短期は十分カバー済みのため除外）
        self.sr_timeframes = [tf for tf in self.timeframes if tf in ("1h", "4h", "1d")]
        # 一目均衡表の4h/1d拡張（1h版はBASE_FEATURESにフルセットで既存）
        self.ichi_extra_timeframes = [
            tf for tf in ("4h", "1d") if tf in self.timeframes
        ]

        self.feature_names = (
            [f"{tf}_{f}" for tf in self.timeframes for f in TF_BLOCK_FEATURES]
            + [f"{tf}_{f}" for tf in self.sr_timeframes for f in SR_BLOCK_FEATURES]
            + [
                f"{tf}_{f}"
                for tf in self.ichi_extra_timeframes
                for f in ICHI_TF_FEATURES
            ]
            + BASE_FEATURES
            + CS_FEATURES  # Phase C2: クロスセクション特徴を末尾に追加
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

        # 長期ホライズンの累積リターンz（2日・1週間）
        base["ret_z48"] = _z(
            base["log_ret"].rolling(48, min_periods=48).sum(), self.z_window
        )
        base["ret_z168"] = _z(
            base["log_ret"].rolling(168, min_periods=168).sum(), max(self.z_window, 200)
        )
        return base

    def _align_tf_features(
        self,
        source: DataSource,
        symbol: str,
        tf: str,
        decision_times: pd.DatetimeIndex,
        start,
        end,
    ) -> pd.DataFrame:
        """TFブロック特徴（ret_z/rsi/bb_pos等）を意思決定時刻に整列"""
        return self._align_generic_tf_features(
            source, symbol, tf, decision_times, start, end, _tf_block, TF_BLOCK_FEATURES
        )

    def _ichi_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        """4h/1d用: 一目均衡表10特徴のうち主要5特徴に絞る"""
        return ichimoku_features(
            df["high"], df["low"], df["close"], z_window=self.z_window
        )[ICHI_TF_FEATURES]

    def _align_generic_tf_features(
        self,
        source: DataSource,
        symbol: str,
        tf: str,
        decision_times: pd.DatetimeIndex,
        start,
        end,
        compute_fn,
        col_names: List[str],
    ) -> pd.DataFrame:
        """
        任意の compute_fn(df) -> DataFrame[col_names] を、そのTF上で計算した上で
        意思決定時刻に整列する。バー開始時刻 + TF長 = バー確定時刻。
        確定時刻 <= 意思決定時刻の最新バーを merge_asof(backward) で対応付ける
        （look-ahead防止）。
        """
        df = source.load_klines(symbol, tf, start, end)
        if df.empty:
            return pd.DataFrame(
                0.0, index=decision_times, columns=[f"{tf}_{c}" for c in col_names]
            )
        feats = compute_fn(df.set_index("timestamp")).reset_index()
        feats["complete_time"] = _ns(
            feats["timestamp"] + pd.Timedelta(minutes=TF_TO_MINUTES[tf])
        )
        feats = feats.drop(columns=["timestamp"]).sort_values("complete_time")

        aligned = (
            pd.merge_asof(
                pd.DataFrame({"decision_time": _ns(decision_times)}),
                feats,
                left_on="decision_time",
                right_on="complete_time",
                direction="backward",
            )
            .drop(columns=["complete_time"])
            .set_index("decision_time")
        )
        aligned.columns = [f"{tf}_{c}" for c in aligned.columns]
        return aligned.fillna(0.0)

    def _orderflow_features(
        self,
        source: DataSource,
        symbol: str,
        bar_index: pd.DatetimeIndex,
        start,
        end,
    ) -> pd.DataFrame:
        """オーダーフロー1m集計 → 基準TFに再集計して特徴化"""
        cols = ["of_imbalance", "of_count_z", "of_size_z"]
        of = source.load_orderflow(symbol, start, end)
        if of.empty:
            return pd.DataFrame(0.0, index=bar_index, columns=cols)

        of = of.set_index("timestamp")
        freq = f"{TF_TO_MINUTES[self.base_tf]}min"
        agg = of.resample(freq).agg(
            {
                "buy_volume": "sum",
                "sell_volume": "sum",
                "trade_count": "sum",
                "avg_trade_size": "mean",
            }
        )
        total = agg["buy_volume"] + agg["sell_volume"]

        out = pd.DataFrame(index=agg.index)
        out["of_imbalance"] = np.where(
            total > 0, (agg["buy_volume"] - agg["sell_volume"]) / total, 0.0
        )
        out["of_count_z"] = _z(agg["trade_count"], self.z_window)
        out["of_size_z"] = _z(agg["avg_trade_size"], self.z_window)
        return out.reindex(bar_index).fillna(0.0)

    def _derivative_features(
        self,
        source: DataSource,
        symbol: str,
        bar_index: pd.DatetimeIndex,
        start,
        end,
    ) -> pd.DataFrame:
        """デリバティブ指標（OI・L/S比率・清算）を基準TFに整列して特徴化"""
        cols = ["oi_z", "oi_change", "ls_ratio_z", "liq_z"]
        d = source.load_derivatives(symbol, start, end)
        if d.empty:
            return pd.DataFrame(0.0, index=bar_index, columns=cols)

        d = d.set_index("timestamp").sort_index()
        freq = f"{TF_TO_MINUTES[self.base_tf]}min"
        # 基準TFにリサンプル（最終値/合計）
        agg = pd.DataFrame(index=None)
        oi = d["open_interest"].resample(freq).last()
        ls = d["ls_ratio"].resample(freq).last()
        liq = d["liq_notional"].resample(freq).sum()

        out = pd.DataFrame(index=oi.index)
        out["oi_z"] = _z(np.log(oi.clip(lower=1e-9)), self.z_window)
        out["oi_change"] = _z(np.log(oi.clip(lower=1e-9)).diff(), self.z_window)
        out["ls_ratio_z"] = _z(np.log(ls.clip(lower=1e-9)), self.z_window)
        out["liq_z"] = _z(np.log1p(liq.clip(lower=0)), self.z_window)
        return out.reindex(bar_index).fillna(0.0)

    def _funding_features(
        self,
        source: DataSource,
        symbol: str,
        bar_index: pd.DatetimeIndex,
        start,
        end,
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
            fr_df,
            left_on="t",
            right_on="ft",
            direction="backward",
        )["funding_rate"].to_numpy()

        feats = pd.DataFrame(index=bar_index)
        feats["funding_bps"] = np.clip(np.nan_to_num(latest) * 1e4, -CLIP, CLIP)
        cum = fr.rolling("24h").sum()
        cum_df = cum.reset_index().rename(columns={"timestamp": "ft"})
        cum_df["ft"] = _ns(cum_df["ft"])
        cum_aligned = pd.merge_asof(
            pd.DataFrame({"t": _ns(decision_times)}),
            cum_df,
            left_on="t",
            right_on="ft",
            direction="backward",
        )["funding_rate"].to_numpy()
        feats["funding_cum_bps"] = np.clip(
            np.nan_to_num(cum_aligned) * 1e4, -CLIP, CLIP
        )
        feats["time_to_funding"] = time_to.to_numpy()

        # このバー内（(T, T+bar]）に発生するfunding授受の合計
        events = fr.copy()
        bucket = (events.index - pd.Timedelta(seconds=1)).floor(f"{bar_minutes}min")
        per_bar = events.groupby(bucket).sum().reindex(bar_index).fillna(0.0)
        return feats, per_bar

    # ---- 公開API ----

    def build(
        self,
        source: DataSource,
        start: Optional[str] = None,
        end: Optional[str] = None,
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

        btc_col = next(
            (s for s in self.symbols if s.upper().startswith("BTC")), self.symbols[0]
        )
        btc_ret = ret_mat[btc_col]

        # ---------- Phase C2: クロスセクション特徴量の事前計算 ----------
        # ターゲットがcs_demean（市場中立リターン）のとき、特徴量側も
        # 同じクロスセクション軸で正規化することで回帰の fit を改善する。
        # ローリングウィンドウ内の全銘柄×全時刻の値を用いて正規化するため、
        # look-ahead-free かつ単一銘柄の時系列ドリフトに依存しない。
        ret1_mat = pd.DataFrame(
            {
                s: np.log(bases[s]["close"] / bases[s]["close"].shift(1))
                for s in self.symbols
            }
        )
        ret5_mat = pd.DataFrame(
            {s: ret1_mat[s].rolling(5, min_periods=5).sum() for s in self.symbols}
        )
        ret20_mat = pd.DataFrame(
            {s: ret1_mat[s].rolling(20, min_periods=20).sum() for s in self.symbols}
        )
        cs_ret1 = _cs_z(ret1_mat, self.z_window)
        cs_ret5 = _cs_z(ret5_mat, self.z_window)
        cs_ret20 = _cs_z(ret20_mat, self.z_window)
        # OI・L/S ratio・Fundingのクロスセクション行列（取得可能な場合のみ）
        # 取得できない銘柄は NaN のままにし、後でゼロ埋めする
        cs_oi_dict, cs_ls_dict, cs_fund_dict = {}, {}, {}
        for _s in self.symbols:
            _d = source.load_derivatives(_s, start, end)
            if not _d.empty:
                _d = _d.set_index("timestamp").sort_index()
                freq = f"{bar_minutes}min"
                _oi = _d["open_interest"].resample(freq).last().reindex(common)
                _ls = _d["ls_ratio"].resample(freq).last().reindex(common)
                cs_oi_dict[_s] = np.log(_oi.clip(lower=1e-9))
                cs_ls_dict[_s] = np.log(_ls.clip(lower=1e-9))
            _fr = source.load_funding(_s, start, end)
            if not _fr.empty:
                _fr = _fr.set_index("timestamp")["funding_rate"]
                _decision_times = common + pd.Timedelta(minutes=bar_minutes)
                _fr_df = _fr.reset_index().rename(columns={"timestamp": "ft"})
                _fr_df["ft"] = _ns(_fr_df["ft"])
                _latest = pd.merge_asof(
                    pd.DataFrame({"t": _ns(_decision_times)}),
                    _fr_df,
                    left_on="t",
                    right_on="ft",
                    direction="backward",
                )["funding_rate"].to_numpy()
                cs_fund_dict[_s] = pd.Series(_latest * 1e4, index=common)

        cs_oi_mat = (
            pd.DataFrame(cs_oi_dict, index=common)
            if cs_oi_dict
            else pd.DataFrame(index=common)
        )
        cs_ls_mat = (
            pd.DataFrame(cs_ls_dict, index=common)
            if cs_ls_dict
            else pd.DataFrame(index=common)
        )
        cs_fund_mat = (
            pd.DataFrame(cs_fund_dict, index=common)
            if cs_fund_dict
            else pd.DataFrame(index=common)
        )

        cs_oi_z = (
            _cs_z(cs_oi_mat, self.z_window)
            if not cs_oi_mat.empty
            else pd.DataFrame(0.0, index=common, columns=self.symbols)
        )
        cs_ls_z = (
            _cs_z(cs_ls_mat, self.z_window)
            if not cs_ls_mat.empty
            else pd.DataFrame(0.0, index=common, columns=self.symbols)
        )
        cs_fund_z = (
            _cs_z(cs_fund_mat, self.z_window)
            if not cs_fund_mat.empty
            else pd.DataFrame(0.0, index=common, columns=self.symbols)
        )
        # ---------------------------------------------------------------

        n_bars, n_sym = len(common), len(self.symbols)
        features = np.zeros((n_bars, n_sym, len(self.feature_names)), dtype=np.float32)
        funding_arr = np.zeros((n_bars, n_sym), dtype=np.float32)
        close_arr = np.zeros((n_bars, n_sym), dtype=np.float64)
        open_next_arr = np.zeros((n_bars, n_sym), dtype=np.float64)

        for i, sym in enumerate(self.symbols):
            blocks = []
            for tf in self.timeframes:
                blk = self._align_tf_features(
                    source, sym, tf, decision_times, start, end
                )
                blk.index = common
                blocks.append(blk)

            for tf in self.sr_timeframes:
                blk = self._align_generic_tf_features(
                    source,
                    sym,
                    tf,
                    decision_times,
                    start,
                    end,
                    _sr_block,
                    SR_BLOCK_FEATURES,
                )
                blk.index = common
                blocks.append(blk)

            for tf in self.ichi_extra_timeframes:
                blk = self._align_generic_tf_features(
                    source,
                    sym,
                    tf,
                    decision_times,
                    start,
                    end,
                    self._ichi_subset,
                    ICHI_TF_FEATURES,
                )
                blk.index = common
                blocks.append(blk)

            base_feats = pd.DataFrame(index=common)
            base_feats["vol_anom"] = bases[sym]["vol_anom"]
            base_feats["rci"] = bases[sym]["rci"]
            base_feats["ret_z48"] = bases[sym]["ret_z48"]
            base_feats["ret_z168"] = bases[sym]["ret_z168"]

            of = self._orderflow_features(source, sym, common, start, end)
            fund_feats, per_bar_funding = self._funding_features(
                source, sym, common, start, end
            )
            deriv = self._derivative_features(source, sym, common, start, end)
            rel = _z(bases[sym]["log_ret"] - btc_ret, self.z_window)

            base_feats = pd.concat([base_feats, of, fund_feats, deriv], axis=1)
            base_feats["btc_rel_z"] = rel
            base_feats["ret_rank"] = rank[sym]

            # 一目均衡表特徴量（look-ahead なし: 先行スパンは shift(26) 済み、未来雲は現在データから計算）
            # リクエストごとにロードするのでキャッシュ済み sources[sym] を再利用
            _kl = source.load_klines(sym, self.base_tf, start, end).set_index(
                "timestamp"
            )
            _kl_common = _kl.reindex(common)
            _ichi = ichimoku_features(
                _kl_common["high"],
                _kl_common["low"],
                _kl_common["close"],
                z_window=self.z_window,
            )
            base_feats = pd.concat([base_feats, _ichi], axis=1)

            # Phase C2: クロスセクション特徴を追加
            cs_feats = pd.DataFrame(index=common)
            cs_feats["csz_ret1"] = cs_ret1[sym] if sym in cs_ret1.columns else 0.0
            cs_feats["csz_ret5"] = cs_ret5[sym] if sym in cs_ret5.columns else 0.0
            cs_feats["csz_ret20"] = cs_ret20[sym] if sym in cs_ret20.columns else 0.0
            cs_feats["csz_oi"] = (
                cs_oi_z[sym].fillna(0.0) if sym in cs_oi_z.columns else 0.0
            )
            cs_feats["csz_ls"] = (
                cs_ls_z[sym].fillna(0.0) if sym in cs_ls_z.columns else 0.0
            )
            cs_feats["csz_funding"] = (
                cs_fund_z[sym].fillna(0.0) if sym in cs_fund_z.columns else 0.0
            )

            full = pd.concat(blocks + [base_feats, cs_feats], axis=1)[
                self.feature_names
            ]
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
                pd.DataFrame(
                    {
                        "open": bases[btc_col]["open"],
                        "close": bases[btc_col]["close"],
                        "high": bases[btc_col]["close"],
                        "low": bases[btc_col]["open"],
                    }
                ).assign(
                    high=lambda d: d[["open", "close"]].max(axis=1) * 1.0001,
                    low=lambda d: d[["open", "close"]].min(axis=1) * 0.9999,
                )
            )
            .rolling(24, min_periods=4)
            .mean(),
            self.z_window,
        )
        # BTCトレンド: 24本トレーリングリターンのz-score（過去窓のみ=look-ahead無し）
        # レジーム判定（強気/弱気/レンジ）の基準。観測に含めることで
        # RegimeEnsembleが obs から直接レジームを読める。
        btc_close = bases[btc_col]["close"]
        btc_ret24 = np.log(btc_close / btc_close.shift(24))
        btc_trend = _z(btc_ret24, self.z_window)
        global_features = np.stack(
            [
                np.sin(2 * np.pi * hours / 24),
                np.cos(2 * np.pi * hours / 24),
                np.sin(2 * np.pi * dows / 7),
                np.cos(2 * np.pi * dows / 7),
                btc_vol.to_numpy(),
                btc_trend.to_numpy(),
            ],
            axis=1,
        ).astype(np.float32)

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
