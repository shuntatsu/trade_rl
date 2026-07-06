"""
評価指標モジュール

Implementation Shortfall、Differential Sharpe Ratio等の計算
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


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
        delta_B = r**2 - B

        denom = (B - A**2) ** 1.5
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
    time_weighted_step = (
        steps * execution_history["quantity"].values
    ).sum() / total_quantity

    # 執行集中度（ハーフィンダール指数的）
    qty_fracs = execution_history["quantity"].values / total_quantity
    concentration = float((qty_fracs**2).sum())

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
