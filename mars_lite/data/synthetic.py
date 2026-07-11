"""
合成マーケット生成モジュール（アルファ注入）

1分足の対数リターン列と「潜在ドリフト状態」を生成し、そこから
OHLCV・オーダーフロー・funding・デリバティブ指標を構築する。
SyntheticSource と generate_sample_data.py の両方がここを使う。

アルファの種類:
    none    : 純ランダムウォーク（陰性対照）
    cross   : 銘柄ごとのAR(1)潜在ドリフト（クロスセクショナルにデミーン）。
              過去リターン・OI・L/S比率が将来リターンを予測する（陽性対照）
    meanrev : 直近24時間リターンへの平均回帰ドリフト
    multi   : cross + meanrev の合成
    bull    : 全銘柄共通の持続的な正ドリフト（方向性ベータ）

潜在状態 latent は「1時間あたりの予測可能ドリフト」を表し、
派生データ（OI/L/S/オーダーフロー/funding）の生成にも使う。
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 1440
VOL_PER_MIN = 0.0009  # 1分あたりのノイズ標準偏差（年率~70%相当）
LATENT_HALF_LIFE_MIN = 60 * 30  # 潜在ドリフトの半減期（30時間）


def generate_market(
    rng: np.random.Generator,
    n_symbols: int,
    n_minutes: int,
    alpha: str = "none",
    alpha_strength: float = 0.002,
    alpha_rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    合成リターンと潜在ドリフト状態を生成

    Returns:
        returns: (n_minutes, n_symbols) 1分対数リターン
        latent:  (n_minutes, n_symbols) 時間あたり予測可能ドリフト状態
    """
    if alpha_rng is None:
        alpha_rng = rng
    phi = np.exp(-np.log(2.0) / LATENT_HALF_LIFE_MIN)
    sig_innov = alpha_strength * np.sqrt(1.0 - phi**2)

    latent = np.zeros((n_minutes, n_symbols))
    returns = np.zeros((n_minutes, n_symbols))
    noise = rng.normal(0.0, VOL_PER_MIN, size=(n_minutes, n_symbols))

    l = np.zeros(n_symbols)
    cum24 = np.zeros(n_symbols)  # 直近24hリターンのEMA近似（meanrev用）
    decay24 = 1.0 - 1.0 / MINUTES_PER_DAY

    use_cross = alpha in (
        "cross",
        "multi",
        "concentrated_alpha",
        "concentrated_alpha_crash",
        "persistent_cross",
        "fast_reversal",
        "vol_shock_up",
        "vol_shock_down",
    )
    use_meanrev = alpha in ("meanrev", "multi")
    scale = 0.5 if alpha == "multi" else 1.0

    phi_use = phi
    if alpha == "persistent_cross":
        phi_use = np.exp(-1.0 / (MINUTES_PER_DAY * 4.0))
    elif alpha == "fast_reversal":
        phi_use = -0.5

    for t in range(n_minutes):
        drift = np.zeros(n_symbols)
        if use_cross:
            l = phi_use * l + alpha_rng.normal(0.0, sig_innov, n_symbols)
            l_eff = l.copy()
            if alpha in ("concentrated_alpha", "concentrated_alpha_crash"):
                l_eff[0] *= 3.0
                l_eff[1:] *= 0.1
            l_eff -= l_eff.mean()
            drift += scale * l_eff
        if use_meanrev:
            drift += -scale * alpha_strength * np.tanh(cum24 / 0.02)
        if alpha == "bull":
            drift += alpha_strength

        latent[t] = drift
        noise_t = noise[t].copy()
        if alpha == "vol_shock_up" and t >= n_minutes // 2:
            noise_t *= 3.0
        elif alpha == "vol_shock_down" and t < n_minutes // 2:
            noise_t *= 3.0

        r = drift / 60.0 + noise_t
        if alpha == "concentrated_alpha_crash" and t >= n_minutes // 2:
            r[0] -= 0.0005

        returns[t] = r
        cum24 = decay24 * cum24 + r

    return returns, latent


def build_ohlcv(
    rng: np.random.Generator,
    returns: np.ndarray,
    start_price: float,
    base_volume: float,
    start,
) -> pd.DataFrame:
    """1銘柄の1分足OHLCVを構築（timestamp=バー開始時刻）"""
    n = len(returns)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0.0, VOL_PER_MIN / 2, size=(n, 2)))
    high = np.maximum(open_, close) * (1.0 + wick[:, 0])
    low = np.minimum(open_, close) * (1.0 - wick[:, 1])
    activity = 1.0 + np.abs(returns) / VOL_PER_MIN
    volume = base_volume * activity * rng.lognormal(0.0, 0.3, n)

    ts = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def build_orderflow(
    rng: np.random.Generator,
    kline_df: pd.DataFrame,
    latent: np.ndarray,
    alpha: str = "none",
) -> pd.DataFrame:
    """1分オーダーフロー集計（買い/売り出来高・件数・平均サイズ）"""
    n = len(kline_df)
    vol = kline_df["volume"].to_numpy()
    # 潜在ドリフトが正 → テイカー買い優勢（アルファ有時のみ情報を持つ）
    signal = latent / (np.abs(latent).std() + 1e-12) if alpha != "none" else np.zeros(n)
    imb = np.tanh(0.5 * signal + rng.normal(0.0, 1.0, n))
    buy = vol * (0.5 + 0.25 * imb)
    sell = vol - buy
    count = np.maximum(
        1, rng.poisson(50, n) + (vol / (vol.mean() + 1e-12) * 20).astype(int)
    )
    avg_size = vol / count
    total = buy + sell
    return pd.DataFrame(
        {
            "timestamp": kline_df["timestamp"],
            "buy_volume": buy,
            "sell_volume": sell,
            "trade_count": count,
            "avg_trade_size": avg_size,
            "volume_imbalance": np.where(total > 0, (buy - sell) / total, 0.0),
        }
    )


def build_funding(
    rng: np.random.Generator,
    latent: np.ndarray,
    start,
    days: int,
    alpha: str = "none",
) -> pd.DataFrame:
    """8時間毎のfunding rate（1日3回）"""
    n_events = days * 3
    ts = pd.date_range(start, periods=n_events, freq="8h")
    idx = np.minimum((np.arange(n_events) * 480), len(latent) - 1)
    sig = (
        latent[idx] / (np.abs(latent).std() + 1e-12)
        if alpha != "none"
        else np.zeros(n_events)
    )
    rate = np.clip(1e-4 * sig + rng.normal(1e-5, 5e-5, n_events), -7.5e-4, 7.5e-4)
    return pd.DataFrame({"timestamp": ts, "funding_rate": rate})


def build_derivatives(
    rng: np.random.Generator,
    kline_df: pd.DataFrame,
    latent: np.ndarray,
    alpha: str = "none",
) -> pd.DataFrame:
    """
    1時間毎のデリバティブ指標（OI・L/S比率・清算notional）

    アルファ有時: OIは潜在ドリフトに正相関（スマートマネーの建玉先行）、
    L/S比率は逆相関（個人の逆張りポジショニング＝逆張りシグナル）。
    """
    n_min = len(kline_df)
    hours = n_min // 60
    idx = np.arange(hours) * 60 + 59
    ts = kline_df["timestamp"].iloc[idx].reset_index(drop=True)

    if alpha != "none":
        sig = latent[idx] / (np.abs(latent).std() + 1e-12)
    else:
        sig = np.zeros(hours)
    noise = rng.normal(0.0, 0.3, hours)
    oi = 1e6 * np.exp(np.clip(1.0 * sig + noise, -20, 20))
    ls = np.exp(-0.8 * sig + rng.normal(0.0, 0.4, hours))

    # 清算: 時間内の大きな逆行で発生
    close = kline_df["close"].to_numpy()
    hr_ret = np.abs(np.log(close[idx] / close[np.maximum(idx - 59, 0)]))
    liq = np.maximum(0.0, hr_ret - 0.005) * 5e7 * rng.lognormal(0.0, 0.5, hours)

    return pd.DataFrame(
        {
            "timestamp": ts,
            "open_interest": oi,
            "ls_ratio": ls,
            "liq_notional": liq,
        }
    )
