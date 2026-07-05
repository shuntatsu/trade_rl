"""
評価指標モジュール

Implementation Shortfall、Differential Sharpe Ratio等の計算
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

_EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """標準正規分布のCDF（math.erfのみで実装、scipy不使用）"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    標準正規分布の分位点関数（逆CDF）。Acklamの有理近似（scipy不使用）。
    精度は|誤差| < 1.15e-9 程度で本用途には十分。
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def deflated_sharpe_ratio(
    returns: np.ndarray,
    trial_sharpes: List[float],
    annualization_factor: float = 252,
) -> Dict[str, float]:
    """
    Deflated Sharpe Ratio（Bailey & Lopez de Prado, 2014）

    「何回試行したか」（シード違い・fold違い・パラメータ違いの繰り返し）を
    差し引いた上で、観測されたSharpeが偶然の産物ではない確率(PSR)を返す。
    バックテストを何度も見て一番良い結果を選ぶほど、選ばれたSharpeは運の
    寄与が混じる（selection bias）。この指標はその危険性を定量化する。

    Args:
        returns: 評価対象（採用する1本）の期間別リターン系列
        trial_sharpes: 実際に試した全試行（他シード・他fold等）の年率化Sharpe。
            試行回数Nとその分散から「偶然で得られる最大Sharpeの期待値」を
            推定するために使う。長さ1なら試行数補正なし（sr0=0扱い）。
        annualization_factor: リターンの年率化係数（1h足なら24*365）

    Returns:
        dsr: 0〜1の確率（高いほど本物のスキルらしい。目安: >=0.95）
        sr0_annualized: 試行数補正後の基準Sharpe（これを超えて初めて「勝ち」）
        sr_hat_annualized: 評価対象の実測Sharpe
        n_trials, skew, kurtosis: 内訳
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    n_trials = max(len(trial_sharpes), 1)
    empty = {
        "dsr": 0.0, "sr0_annualized": 0.0, "sr_hat_annualized": 0.0,
        "n_trials": n_trials, "skew": 0.0, "kurtosis": 3.0,
    }
    if n < 4:
        return empty

    mean = float(returns.mean())
    std = float(returns.std(ddof=1))
    if std < 1e-12:
        return empty
    sr_hat = mean / std  # 期間あたり（未年率化）Sharpe

    z = (returns - mean) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # 非超過尖度（正規分布で3.0）

    sr0 = 0.0
    if n_trials > 1:
        # trial_sharpesは年率化済みの値を想定 -> 期間あたりへ変換して分散を推定
        per_period_trials = np.asarray(trial_sharpes, dtype=np.float64) / math.sqrt(annualization_factor)
        sr_var = float(np.var(per_period_trials, ddof=1))
        if sr_var > 1e-12:
            z_a = _norm_ppf(1.0 - 1.0 / n_trials)
            z_b = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
            sr0 = math.sqrt(sr_var) * ((1 - _EULER_MASCHERONI) * z_a + _EULER_MASCHERONI * z_b)

    denom = math.sqrt(max(1e-12, 1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat ** 2))
    psr_stat = (sr_hat - sr0) * math.sqrt(n - 1) / denom
    dsr = _norm_cdf(psr_stat)

    return {
        "dsr": float(dsr),
        "sr0_annualized": float(sr0 * math.sqrt(annualization_factor)),
        "sr_hat_annualized": float(sr_hat * math.sqrt(annualization_factor)),
        "n_trials": int(n_trials),
        "skew": float(skew),
        "kurtosis": float(kurt),
    }


def calc_implementation_shortfall(
    execution_history: pd.DataFrame,
    arrival_price: float,
) -> Dict[str, float]:
    """
    Implementation Shortfall（IS）を計算
    
    到達価格基準のコスト指標。
    
    Args:
        execution_history: 執行履歴（p_base, p_exec, quantity列必須）
        arrival_price: 到達価格（エピソード開始時の価格）
        
    Returns:
        {
            "total_is": 総IS,
            "is_bps": ISをベーシスポイントで表現,
            "permanent_is": 恒久的IS推定,
            "temporary_is": 一時的IS推定,
        }
    """
    if len(execution_history) == 0:
        return {
            "total_is": 0.0,
            "is_bps": 0.0,
            "permanent_is": 0.0,
            "temporary_is": 0.0,
        }
    
    # 各取引のコスト
    # IS = Σ Q_i * (P_arrival - P_exec_i) / P_arrival
    quantities = execution_history["quantity"].values
    exec_prices = execution_history["p_exec"].values
    
    total_quantity = quantities.sum()
    
    # 加重平均執行価格
    vwap = (quantities * exec_prices).sum() / (total_quantity + 1e-8)
    
    # 総IS（金額）
    total_is = total_quantity * (arrival_price - vwap)
    
    # ISをベーシスポイント（bps）で表現
    is_bps = (arrival_price - vwap) / arrival_price * 10000
    
    # 一時的・恒久的の分解（簡易推定）
    base_prices = execution_history["p_base"].values
    temporary_is = (quantities * (base_prices - exec_prices)).sum()
    permanent_is = total_is - temporary_is
    
    return {
        "total_is": float(total_is),
        "is_bps": float(is_bps),
        "vwap": float(vwap),
        "arrival_price": float(arrival_price),
        "permanent_is": float(permanent_is),
        "temporary_is": float(temporary_is),
    }


def calc_differential_sharpe_ratio(
    returns: np.ndarray,
    eta: float = 0.01,
) -> float:
    """
    Differential Sharpe Ratio（DSR）を計算
    
    オンライン学習向けの微分可能なシャープレシオ近似。
    
    Args:
        returns: リターン系列
        eta: 学習率（移動平均の重み）
        
    Returns:
        DSR値
    """
    if len(returns) < 2:
        return 0.0
    
    # 初期推定
    A = returns[0]  # 平均の推定
    B = returns[0] ** 2  # 二乗平均の推定
    
    dsr_values = []
    
    for r in returns[1:]:
        # 微分計算
        delta_A = r - A
        delta_B = r ** 2 - B
        
        denom = (B - A ** 2) ** 1.5
        if abs(denom) > 1e-8:
            dsr = (B * delta_A - 0.5 * A * delta_B) / denom
        else:
            dsr = 0.0
        
        dsr_values.append(dsr)
        
        # 更新
        A = A + eta * delta_A
        B = B + eta * delta_B
    
    return float(np.mean(dsr_values))


def calc_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252,
) -> float:
    """
    標準的なシャープレシオを計算
    
    Args:
        returns: リターン系列
        risk_free_rate: 無リスク金利（年率）
        annualization_factor: 年率化係数（日次なら252）
        
    Returns:
        シャープレシオ
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)
    
    if std_return < 1e-8:
        return 0.0
    
    return float(mean_return / std_return * np.sqrt(annualization_factor))


def calc_execution_metrics(
    execution_history: pd.DataFrame,
    initial_inventory: float,
) -> Dict[str, Any]:
    """
    執行メトリクスを総合計算
    
    Args:
        execution_history: 執行履歴
        initial_inventory: 初期在庫
        
    Returns:
        総合メトリクス辞書
    """
    if len(execution_history) == 0:
        return {
            "n_trades": 0,
            "total_quantity": 0.0,
            "completion_rate": 0.0,
        }
    
    # 基本統計
    n_trades = len(execution_history)
    total_quantity = execution_history["quantity"].sum()
    completion_rate = total_quantity / initial_inventory
    
    # 平均指標
    mean_pov = execution_history["pov"].mean()
    mean_impact = execution_history["impact_pct"].mean()
    mean_spread_cost = execution_history["spread_cost_pct"].mean()
    
    # 時間分布
    steps = execution_history["step"].values
    time_weighted_step = (steps * execution_history["quantity"].values).sum() / total_quantity
    
    # 執行集中度（ハーフィンダール指数的）
    qty_fracs = execution_history["quantity"].values / total_quantity
    concentration = float((qty_fracs ** 2).sum())
    
    return {
        "n_trades": n_trades,
        "total_quantity": float(total_quantity),
        "completion_rate": float(completion_rate),
        "mean_pov": float(mean_pov),
        "mean_impact_pct": float(mean_impact),
        "mean_spread_cost_pct": float(mean_spread_cost),
        "time_weighted_step": float(time_weighted_step),
        "concentration": concentration,
    }
