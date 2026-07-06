"""
一目均衡表（Ichimoku Kinko Hyo）指標計算モジュール

【重要: look-ahead バイアスの回避設計】
一目均衡表には構造的に未来を参照する計算が2つある。

1. 先行スパン A・B（Senkou Span）:
   - 本来の定義: 「時刻 t に計算した (tenkan+kijun)/2 を t+26 の位置に描画」
   - 学習で使える形: 「時刻 t の雲 = 26本前（t-26）に計算された先行スパン」
   → 実装: raw_span を .shift(displacement) することで
     「過去の予測が今届いた」として look-ahead なしに変換。

2. 遅行スパン（Chikou Span）:
   - 定義: close[t] を t-26 に描画（= 今の終値を過去に置く）
   - 言い換え: t 時点で遅行スパンが示す位置は close[t+26]
   - 純粋な未来終値なので学習特徴量には一切使用しない。

【学習に有効な特徴量 (6種)】
  ichi_pos        : 価格の雲に対する相対位置 (上/中/下)
  ichi_cloud_thick: 雲の厚さ → サポート・レジスタンスの強度
  ichi_cloud_bull : 雲の色 (+1=上昇雲, -1=下降雲)
  ichi_tk_cross   : 転換線-基準線乖離 → ゴールデン/デッドクロス強度
  ichi_price_kijun : 価格-基準線乖離 → 中期トレンドの強さ
  ichi_price_tenkan: 価格-転換線乖離 → 短期過熱/過冷
"""

import numpy as np
import pandas as pd


# ============================================================
# コア計算
# ============================================================

def _mid(series_high: pd.Series, series_low: pd.Series, period: int) -> pd.Series:
    """(n本高値最大 + n本安値最小) / 2  ── look-ahead なし（過去窓のみ）"""
    return (
        series_high.rolling(period, min_periods=period).max()
        + series_low.rolling(period, min_periods=period).min()
    ) / 2.0


def calc_ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    一目均衡表の全ラインを計算する（look-ahead 安全版）

    Returns:
        DataFrame with columns:
            tenkan     : 転換線（9本中値）
            kijun      : 基準線（26本中値）
            senkou_a   : 先行スパンA  ← .shift(26) 済み = look-ahead なし
            senkou_b   : 先行スパンB  ← .shift(26) 済み = look-ahead なし
            chikou     : 遅行スパン   ← 未来情報。学習特徴量では使わないこと

    look-ahead 安全性の詳細:
        senkou_a[t] = (tenkan[t-26] + kijun[t-26]) / 2
        senkou_b[t] = mid_52[t-26]
        → t 時点で参照できるのは t-26 以前の情報のみ → 安全
    """
    tenkan = _mid(high, low, tenkan_period)
    kijun  = _mid(high, low, kijun_period)

    # 先行スパン: 26本前に計算された値を現在時刻にシフト
    # 本来「26本先に描画」 → 「26本遅延で参照」に変換することでリークなし
    raw_span_a = (tenkan + kijun) / 2.0
    raw_span_b = _mid(high, low, senkou_b_period)
    senkou_a = raw_span_a.shift(displacement)
    senkou_b = raw_span_b.shift(displacement)

    # 遅行スパン: 現在終値を26本後ろへ = 未来の終値が含まれる（学習禁止）
    chikou = close.shift(-displacement)

    return pd.DataFrame({
        "tenkan":   tenkan,
        "kijun":    kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou":   chikou,
    }, index=close.index)


# ============================================================
# RL 学習用特徴量への変換
# ============================================================

def ichimoku_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
    clip: float = 5.0,
) -> pd.DataFrame:
    """
    一目均衡表から RL 学習用に数値化した6特徴量を返す。

    遅行スパン（chikou）は look-ahead のため含めない。
    すべての特徴量は [-clip, clip] にクリップし、NaN は 0.0 で埋める。

    特徴量:
        ichi_pos          価格の雲に対する相対位置 [-clip, clip]
                          +  : 雲の上 (= 上昇バイアス、サポートゾーン上)
                          0  : 雲の中 (= 膠着状態、突破待ち)
                          -  : 雲の下 (= 下落バイアス、レジスタンスゾーン下)

        ichi_cloud_thick  雲の厚さ (senkou_a と senkou_b の絶対差 / close)
                          大 = 強いサポート/レジスタンス → 価格が通過しにくい
                          小 = 薄い雲 → ブレイクアウトが起きやすい

        ichi_cloud_bull   雲の色
                          +1 = 上昇雲 (senkou_a > senkou_b = bullish kumo)
                          -1 = 下降雲 (senkou_a < senkou_b = bearish kumo)

        ichi_tk_cross     転換線と基準線の乖離 / close
                          + = 転換線が基準線を上回る (ゴールデンクロス方向)
                          - = 転換線が基準線を下回る (デッドクロス方向)

        ichi_price_kijun  (close - 基準線) / close
                          + = 価格が基準線より上 (中期上昇トレンド)
                          - = 価格が基準線より下 (中期下落トレンド)

        ichi_price_tenkan (close - 転換線) / close
                          + = 価格が転換線より上 (短期強気)
                          - = 価格が転換線より下 (短期弱気)

    Caution:
        先頭 (kijun_period + displacement) 本分は NaN → 0 で埋まる。
        FeaturePipeline の z_window に比べて短いので問題ない
        (kijun=26 + disp=26 = 52本。z_window デフォルト=100本以内)。
    """
    ichi = calc_ichimoku(
        high, low, close,
        tenkan_period, kijun_period, senkou_b_period, displacement,
    )
    tenkan  = ichi["tenkan"]
    kijun   = ichi["kijun"]
    span_a  = ichi["senkou_a"]
    span_b  = ichi["senkou_b"]

    # 雲の上限・下限（A と B のどちらが大きいかで変わる）
    both    = pd.concat([span_a, span_b], axis=1)
    kumo_top = both.max(axis=1)
    kumo_bot = both.min(axis=1)

    safe_close = close.replace(0, np.nan)

    # ---- ichi_pos: 価格の雲に対する相対位置 ----
    # 雲の上: (close - kumo_top) / close > 0
    # 雲の下: (close - kumo_bot) / close < 0
    # 雲の中: 0
    above = (close - kumo_top) / safe_close
    below = (close - kumo_bot) / safe_close
    pos_raw = np.where(
        above > 0, above,
        np.where(below < 0, below, 0.0),
    )
    ichi_pos = pd.Series(
        np.clip(pos_raw, -clip, clip), index=close.index,
    ).fillna(0.0)

    # ---- ichi_cloud_thick: 雲の厚さ ----
    thickness_pct = (span_a - span_b).abs() / safe_close.clip(lower=1e-12)
    # log1p で安定化、×100 でスケール調整 (0.1% 差 → log1p(0.1) ≈ 0.1)
    ichi_cloud_thick = (
        np.log1p(thickness_pct.clip(lower=0) * 100)
        .clip(0, clip)
        .fillna(0.0)
    )

    # ---- ichi_cloud_bull: 雲の色 ----
    ichi_cloud_bull = np.sign(span_a - span_b).fillna(0.0)

    # ---- ichi_tk_cross: 転換線-基準線乖離 ----
    ichi_tk_cross = (
        ((tenkan - kijun) / safe_close)
        .clip(-clip, clip)
        .fillna(0.0)
    )

    # ---- ichi_price_kijun: 価格-基準線乖離 ----
    ichi_price_kijun = (
        ((close - kijun) / safe_close)
        .clip(-clip, clip)
        .fillna(0.0)
    )

    # ---- ichi_price_tenkan: 価格-転換線乖離 ----
    ichi_price_tenkan = (
        ((close - tenkan) / safe_close)
        .clip(-clip, clip)
        .fillna(0.0)
    )

    return pd.DataFrame({
        "ichi_pos":           ichi_pos,
        "ichi_cloud_thick":   ichi_cloud_thick,
        "ichi_cloud_bull":    ichi_cloud_bull,
        "ichi_tk_cross":      ichi_tk_cross,
        "ichi_price_kijun":   ichi_price_kijun,
        "ichi_price_tenkan":  ichi_price_tenkan,
    }, index=close.index)
