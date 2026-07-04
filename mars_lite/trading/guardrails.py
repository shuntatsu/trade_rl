"""
ライブ運用ガードレールモジュール

推論の入力・出力を検査し、異常時は「フラット化」または「グロス縮小」を
指示する。すべての発動は理由付きで返し、Platform側はそれをUI警報や
発注抑制に使える。

設計方針: 「疑わしきはリスクを落とす」。データ異常・分布逸脱・損失超過は
いずれも方策が信頼できない状況なので、ポジションを縮小/解消する。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class GuardrailConfig:
    max_data_age_hours: float = 2.0        # これ以上古いデータならシグナル無効
    max_daily_loss: float = 0.05           # 日次損失上限（超で解消）
    max_drawdown: float = 0.20             # 最大DD上限（超で解消）
    max_consecutive_losses: int = 12       # 連続負けバー上限（超でグロス半減）
    max_turnover_z: float = 3.0            # 回転率の学習時分布からの逸脱σ
    max_abs_weight: float = 0.5            # 単一銘柄の絶対ウェイト上限


@dataclass
class GuardrailState:
    """運用側が保持する累積状態"""
    day_start_value: float = 1.0
    peak_value: float = 1.0
    consecutive_losses: int = 0
    turnover_mean: float = 0.0             # 学習時の回転率平均
    turnover_std: float = 1.0


@dataclass
class GuardrailResult:
    action: str = "proceed"                # proceed | scale | flatten
    scale: float = 1.0                     # scale時の倍率
    triggered: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "action": self.action, "scale": self.scale,
            "triggered": self.triggered, "warnings": self.warnings,
        }


def evaluate_guardrails(
    weights: np.ndarray,
    portfolio_value: float,
    turnover: float,
    data_age_hours: float,
    features: Optional[np.ndarray] = None,
    state: Optional[GuardrailState] = None,
    config: Optional[GuardrailConfig] = None,
) -> GuardrailResult:
    """
    ガードレールを評価し推奨アクションを返す

    Args:
        weights: 後処理済みの目標ウェイト
        portfolio_value: 現在の資産（初期=1.0基準）
        turnover: 直近の回転率
        data_age_hours: 最終データの経過時間
        features: 現在の特徴量ベクトル（NaN/全ゼロ検知用）
        state: 累積状態
        config: しきい値
    """
    cfg = config or GuardrailConfig()
    st = state or GuardrailState()
    res = GuardrailResult()

    # --- データ健全性（最優先: 異常なら即フラット） ---
    if data_age_hours > cfg.max_data_age_hours:
        res.action = "flatten"
        res.triggered.append(f"stale data ({data_age_hours:.1f}h old)")
        res.scale = 0.0
        return res

    if features is not None:
        if np.isnan(features).any():
            res.action = "flatten"
            res.triggered.append("NaN in features")
            res.scale = 0.0
            return res
        if np.all(features == 0):
            res.action = "flatten"
            res.triggered.append("all-zero features (feed likely down)")
            res.scale = 0.0
            return res

    # --- 損失系（フラット化） ---
    daily_loss = 1.0 - portfolio_value / max(st.day_start_value, 1e-9)
    if daily_loss > cfg.max_daily_loss:
        res.action = "flatten"
        res.triggered.append(f"daily loss {daily_loss:.1%} > {cfg.max_daily_loss:.0%}")
        res.scale = 0.0
        return res

    drawdown = 1.0 - portfolio_value / max(st.peak_value, 1e-9)
    if drawdown > cfg.max_drawdown:
        res.action = "flatten"
        res.triggered.append(f"drawdown {drawdown:.1%} > {cfg.max_drawdown:.0%}")
        res.scale = 0.0
        return res

    # --- 縮小系（グロス半減） ---
    if st.consecutive_losses > cfg.max_consecutive_losses:
        res.action = "scale"
        res.scale = min(res.scale, 0.5)
        res.triggered.append(f"{st.consecutive_losses} consecutive losing bars")

    # 回転率が学習時分布から逸脱＝方策挙動の異常
    if st.turnover_std > 0:
        tz = (turnover - st.turnover_mean) / st.turnover_std
        if tz > cfg.max_turnover_z:
            res.action = "scale"
            res.scale = min(res.scale, 0.5)
            res.triggered.append(f"turnover z-score {tz:.1f} > {cfg.max_turnover_z}")

    # --- 警告のみ（発注は継続） ---
    if np.abs(weights).max() > cfg.max_abs_weight:
        res.warnings.append(
            f"weight cap exceeded ({np.abs(weights).max():.2f} > {cfg.max_abs_weight})"
        )

    return res


def apply_guardrails(weights: np.ndarray, result: GuardrailResult) -> np.ndarray:
    """ガードレール結果をウェイトに適用"""
    if result.action == "flatten":
        return np.zeros_like(weights)
    if result.action == "scale":
        return weights * result.scale
    return weights
