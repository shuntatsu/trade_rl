"""
一目均衡表（Ichimoku Kinko Hyo）指標計算モジュール

【look-ahead バイアスの回避設計】

1. 先行スパン A・B（現在雲）:
   - 本来の定義: 「時刻 t の計算結果を t+26 の位置に描画」
   - look-ahead 安全化: raw_span を .shift(displacement) することで
     「26本前に計算された予測が今届いた」として参照 → look-ahead なし

2. 遅行スパン（Chikou Span）:
   - close[t+26] を t 時点に置く = 純粋な未来情報 → 学習特徴量から完全除外

【予測的特徴量（Future Cloud）】

一目均衡表の最も強力な洞察：
  「今時刻 t での tenkan[t] と kijun[t] から、t+26 の雲が"予測"できる」

  future_span_a[t] = (tenkan[t] + kijun[t]) / 2   ← これが t+26 に出現する雲A
  future_span_b[t] = mid_52[t]                     ← これが t+26 に出現する雲B

これは look-ahead ではない。なぜなら tenkan[t], kijun[t], mid_52[t] は
すべて時刻 t までのデータから計算されているから。
この「未来の雲の予測値」を現在の特徴量として使うことで、
エージェントは「26本先のサポート・レジスタンス構造」を学習できる。

【全特徴量（10種、すべてローリングz-score標準化済み）】

  現在雲との関係（過去26本の予測が届いた雲）:
    ichi_pos          価格の現在雲に対する相対位置
    ichi_cloud_thick  現在雲の厚さ（サポート/レジスタンス強度）
    ichi_cloud_bull   現在雲の色（+1=上昇雲, -1=下降雲）

  転換線・基準線との関係:
    ichi_tk_cross     転換線-基準線乖離（クロス方向と強度）
    ichi_price_kijun  価格-基準線乖離（中期トレンド位置）
    ichi_price_tenkan 価格-転換線乖離（短期過熱/過冷）

  未来雲予測（26本先の雲をt時点で予測 = look-ahead なし）:
    ichi_future_pos   価格の"未来雲"に対する予測的位置
    ichi_future_bull  未来雲の色予測（+1=上昇雲になる, -1=下降雲になる）
    ichi_future_thick 未来雲の厚さ予測（将来の強い壁の有無）
    ichi_tk_accel     転換線と基準線の乖離加速度（トレンド加速/減速）
"""

import numpy as np
import pandas as pd
from typing import Optional

CLIP = 5.0

# ============================================================
# 内部ヘルパー
# ============================================================

def _mid(series_high: pd.Series, series_low: pd.Series, period: int) -> pd.Series:
    """(n本高値最大 + n本安値最小) / 2  ── look-ahead なし（過去窓のみ）"""
    return (
        series_high.rolling(period, min_periods=period).max()
        + series_low.rolling(period, min_periods=period).min()
    ) / 2.0


def _rolling_z(
    series: pd.Series,
    window: int = 100,
    min_periods: int = 20,
    clip: float = CLIP,
) -> pd.Series:
    """
    ローリングz-score（feature_pipeline._z と同一ロジック）。
    過去窓のみ使用 → look-ahead なし。NaN → 0.0 で埋め。
    """
    mean = series.rolling(window, min_periods=min_periods).mean()
    std  = series.rolling(window, min_periods=min_periods).std()
    z    = (series - mean) / std.replace(0, np.nan)
    return z.clip(-clip, clip).fillna(0.0)


# ============================================================
# コア計算
# ============================================================

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
            tenkan         転換線（9本中値）
            kijun          基準線（26本中値）
            senkou_a       先行スパンA ← shift(displacement)済み = look-ahead なし
            senkou_b       先行スパンB ← shift(displacement)済み = look-ahead なし
            future_span_a  t+26の雲Aの予測値（= raw_span_a。現在データから計算）
            future_span_b  t+26の雲Bの予測値（= raw_span_b。現在データから計算）
            chikou         遅行スパン ← 未来終値なので学習特徴量では使わないこと

    look-ahead 安全性:
        senkou_a[t]    = (tenkan[t-26] + kijun[t-26]) / 2  ← 過去のみ依存
        future_span_a[t] = (tenkan[t] + kijun[t]) / 2      ← 過去のみ依存（未来描画だが計算は過去データ）
        chikou[t]      = close[t+26]                        ← 未来データ依存 = 学習禁止
    """
    tenkan = _mid(high, low, tenkan_period)
    kijun  = _mid(high, low, kijun_period)

    # --- 現在雲（26本前に計算した先行スパンが今届いた） ---
    raw_span_a = (tenkan + kijun) / 2.0
    raw_span_b = _mid(high, low, senkou_b_period)
    senkou_a   = raw_span_a.shift(displacement)   # look-ahead なし
    senkou_b   = raw_span_b.shift(displacement)   # look-ahead なし

    # --- 未来雲予測（t時点のデータから t+26 の雲を予測 = look-ahead なし） ---
    # raw_span_a / raw_span_b は現在時刻 t のデータのみで計算されている
    # これが実際に t+26 の位置に描画されることが一目均衡表の設計
    future_span_a = raw_span_a   # = (tenkan[t] + kijun[t]) / 2
    future_span_b = raw_span_b   # = mid_52[t]

    # --- 遅行スパン（未来情報 → 学習には使わない） ---
    chikou = close.shift(-displacement)

    return pd.DataFrame({
        "tenkan":        tenkan,
        "kijun":         kijun,
        "senkou_a":      senkou_a,
        "senkou_b":      senkou_b,
        "future_span_a": future_span_a,
        "future_span_b": future_span_b,
        "chikou":        chikou,
    }, index=close.index)


# ============================================================
# RL 学習用特徴量への変換（10種、z-score標準化済み）
# ============================================================

def ichimoku_features(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
    z_window: int = 100,
    clip: float = CLIP,
) -> pd.DataFrame:
    """
    一目均衡表から RL 学習用に数値化・z-score標準化した10特徴量を返す。

    すべての特徴量は「比率（/close）」→「ローリングz-score」の2段変換で
    他の特徴量（ret_z*, bb_pos 等）と同一スケールに揃える。
    NaN（warmup 期間）は 0.0 で埋める。

    Args:
        z_window:  ローリングz-score の窓長（feature_pipeline の z_window と合わせる）

    現在雲との関係（「26本前の予測が届いた雲」との位置関係）:
        ichi_pos          価格の現在雲に対する相対位置
                          +  雲の上（上昇バイアス），0  雲の中，-  雲の下（下落バイアス）
        ichi_cloud_thick  現在雲の厚さ（強いサポート/レジスタンスの強度）
        ichi_cloud_bull   現在雲の色（+1=上昇雲, -1=下降雲）

    転換線・基準線との関係:
        ichi_tk_cross     転換線-基準線乖離（正=ゴールデンクロス方向）
        ichi_price_kijun  価格-基準線乖離（中期トレンドの位置確認）
        ichi_price_tenkan 価格-転換線乖離（短期過熱/過冷）

    未来雲予測（現在データから t+26 の雲を予測 = look-ahead なし）:
        ichi_future_pos   価格の"26本先の雲"に対する予測的相対位置
                          正=現価格が将来雲の上側 → 26本後もサポートに乗っている見通し
                          負=現価格が将来雲の下側 → 26本後にレジスタンスに阻まれる見通し
        ichi_future_bull  未来雲の色予測（+1=26本後に上昇雲 → 強気継続の構造）
        ichi_future_thick 未来雲の厚さ予測（大=将来に強いサポート/レジスタンス帯あり）
        ichi_tk_accel     転換線と基準線の乖離の変化速度（加速=トレンド加速、減速=モメンタム減衰）

    Note:
        chikou (遅行スパン = close[t+26]) は look-ahead のため含めない。
    """
    ichi = calc_ichimoku(
        high, low, close,
        tenkan_period, kijun_period, senkou_b_period, displacement,
    )
    tenkan    = ichi["tenkan"]
    kijun     = ichi["kijun"]
    span_a    = ichi["senkou_a"]
    span_b    = ichi["senkou_b"]
    f_span_a  = ichi["future_span_a"]
    f_span_b  = ichi["future_span_b"]

    safe_close = close.replace(0, np.nan).clip(lower=1e-12)

    # ================================================================
    # 現在雲との関係（Step 1: 比率計算 → Step 2: ローリングz-score）
    # ================================================================

    # 現在雲の上限・下限
    cur_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cur_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)

    # ichi_pos: 価格の雲に対する相対位置（比率 → z-score）
    above_ratio = (close - cur_top) / safe_close
    below_ratio = (close - cur_bot) / safe_close
    pos_raw = np.where(above_ratio > 0, above_ratio,
                       np.where(below_ratio < 0, below_ratio, 0.0))
    ichi_pos = _rolling_z(
        pd.Series(pos_raw, index=close.index), z_window, clip=clip
    )

    # ichi_cloud_thick: 雲の厚さ（log比率 → z-score）
    thick_ratio = (span_a - span_b).abs() / safe_close
    ichi_cloud_thick = _rolling_z(
        np.log1p(thick_ratio.clip(lower=0) * 100), z_window, clip=clip
    )

    # ichi_cloud_bull: 雲の色（+1/-1/0、連続値として z-score）
    cloud_bull_raw = np.sign(span_a - span_b).fillna(0.0)
    ichi_cloud_bull = _rolling_z(
        pd.Series(cloud_bull_raw, index=close.index), z_window, clip=clip
    )

    # ================================================================
    # 転換線・基準線との関係
    # ================================================================

    # ichi_tk_cross: 転換線-基準線乖離の比率 → z-score
    tk_ratio = (tenkan - kijun) / safe_close
    ichi_tk_cross = _rolling_z(tk_ratio, z_window, clip=clip)

    # ichi_price_kijun: 価格-基準線乖離の比率 → z-score
    pk_ratio = (close - kijun) / safe_close
    ichi_price_kijun = _rolling_z(pk_ratio, z_window, clip=clip)

    # ichi_price_tenkan: 価格-転換線乖離の比率 → z-score
    pt_ratio = (close - tenkan) / safe_close
    ichi_price_tenkan = _rolling_z(pt_ratio, z_window, clip=clip)

    # ================================================================
    # 未来雲予測（t時点のデータから t+26 を予測 = look-ahead なし）
    # ================================================================

    # 未来雲の上限・下限（将来のサポート/レジスタンス構造の予測）
    fut_top = pd.concat([f_span_a, f_span_b], axis=1).max(axis=1)
    fut_bot = pd.concat([f_span_a, f_span_b], axis=1).min(axis=1)

    # ichi_future_pos: 現在価格 vs 未来雲の予測的相対位置
    #   正 → 今の価格が26本後の雲の上 → 上昇構造の継続が見込まれる
    #   負 → 今の価格が26本後の雲の下 → レジスタンスに阻まれる構造
    f_above = (close - fut_top) / safe_close
    f_below = (close - fut_bot) / safe_close
    f_pos_raw = np.where(f_above > 0, f_above,
                         np.where(f_below < 0, f_below, 0.0))
    ichi_future_pos = _rolling_z(
        pd.Series(f_pos_raw, index=close.index), z_window, clip=clip
    )

    # ichi_future_bull: 未来雲の色予測
    #   +1 → 26本後に上昇雲（強気構造の継続）
    #   -1 → 26本後に下降雲（弱気構造の継続）
    f_bull_raw = np.sign(f_span_a - f_span_b).fillna(0.0)
    ichi_future_bull = _rolling_z(
        pd.Series(f_bull_raw, index=close.index), z_window, clip=clip
    )

    # ichi_future_thick: 未来雲の厚さ予測（将来の強い壁の存在）
    f_thick_ratio = (f_span_a - f_span_b).abs() / safe_close
    ichi_future_thick = _rolling_z(
        np.log1p(f_thick_ratio.clip(lower=0) * 100), z_window, clip=clip
    )

    # ichi_tk_accel: 転換線-基準線乖離の1階差分（トレンド加速度）
    #   正 → 転換線が基準線から更に遠ざかる（上昇モメンタム加速）
    #   負 → 転換線が基準線に近づく（モメンタム減衰・クロス警戒）
    tk_accel_raw = tk_ratio.diff()
    ichi_tk_accel = _rolling_z(tk_accel_raw, z_window, clip=clip)

    return pd.DataFrame({
        # 現在雲
        "ichi_pos":           ichi_pos,
        "ichi_cloud_thick":   ichi_cloud_thick,
        "ichi_cloud_bull":    ichi_cloud_bull,
        # 転換線・基準線
        "ichi_tk_cross":      ichi_tk_cross,
        "ichi_price_kijun":   ichi_price_kijun,
        "ichi_price_tenkan":  ichi_price_tenkan,
        # 未来雲予測（現在データから t+26 を予測）
        "ichi_future_pos":    ichi_future_pos,
        "ichi_future_bull":   ichi_future_bull,
        "ichi_future_thick":  ichi_future_thick,
        "ichi_tk_accel":      ichi_tk_accel,
    }, index=close.index)
