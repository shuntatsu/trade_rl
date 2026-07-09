"""
執行コストモデルモジュール（Almgren-Chriss由来）

配分層が生む目標ウェイト変化を、現実的な執行コストに変換する。
現在の線形コスト（回転率×固定率）は大きなリバランスのマーケットインパクトを
過小評価する。ここでは:

- 手数料・スプレッド: 回転率に線形（避けられない固定コスト）
- マーケットインパクト: **平方根則**（1銘柄あたり impact ∝ |Δw|^1.5、
  Almgren-Chriss の一時的インパクト）。大口ほど不利に効く
- **TWAP分割**: 大きな注文をK個のサブバーに分割すると総インパクトが 1/√K に。
  分割の恩恵をモデル化し、エージェントに「大きく動くなら分割」を学ばせる

これを環境の step 内で使うことで、学習時のコストが実運用の執行コスト構造を
反映する（train/serve一致）。
"""

from dataclasses import dataclass

import numpy as np

# 執行プロファイル: fee_rate/spread_rate/impact_rate の既定セット。
# taker=成行/IOC想定（現行既定）。maker=指値で板に並べる想定で、手数料が
# 下がりスプレッドは払わない（クロスせず受動的に約定するため）。
# 重要な注意: このモデルは未約定リスク・逆選択・機会損失を表現しない。
# maker前提の結果は「約定できた場合の」楽観シナリオとして解釈すること
# （実運用では約定率・キュー位置を別途検証する必要がある）。
FEE_PROFILES: dict = {
    "taker": {"fee_rate": 0.0005, "spread_rate": 0.0002, "impact_rate": 0.0001},
    "maker": {"fee_rate": 0.0002, "spread_rate": 0.0, "impact_rate": 0.0001},
}
FEE_KWARG_KEYS = ("fee_rate", "spread_rate", "impact_rate")


@dataclass
class ExecutionModel:
    """sqrt-impact + TWAP分割の執行コストモデル"""

    fee_rate: float = 0.0005  # taker手数料（回転率に線形）
    spread_rate: float = 0.0002  # スプレッド/2（回転率に線形）
    impact_coef: float = 0.0005  # sqrt-impact係数（|Δw|^1.5に乗算）
    n_slices: int = 1  # TWAP分割数（インパクトを1/√n_slicesに）
    cost_multiplier: float = 1.0  # 感度分析用

    def cost_fraction(self, weight_deltas: np.ndarray) -> float:
        """
        ウェイト変化ベクトルからポートフォリオ価値比の総コストを計算

        Args:
            weight_deltas: 銘柄別の目標ウェイト変化（ポートフォリオ比）

        Returns:
            総コスト（ポートフォリオ価値に対する割合）
        """
        d = np.abs(np.asarray(weight_deltas, dtype=np.float64))
        turnover = d.sum()
        linear = turnover * (self.fee_rate + self.spread_rate)
        impact = self.impact_coef * np.sum(d**1.5) / np.sqrt(max(self.n_slices, 1))
        return float((linear + impact) * self.cost_multiplier)

    def with_multiplier(self, m: float) -> "ExecutionModel":
        return ExecutionModel(
            self.fee_rate, self.spread_rate, self.impact_coef, self.n_slices, m
        )


def make_execution_model(
    fee_rate: float = 0.0005,
    spread_rate: float = 0.0002,
    impact_rate: float = 0.0001,
    n_slices: int = 1,
    cost_multiplier: float = 1.0,
) -> ExecutionModel:
    """
    従来の線形パラメータから執行モデルを構築（後方互換の橋渡し）

    impact_rate（線形係数）を平方根則の係数へ概算換算する。典型的な
    リバランスサイズ |Δw|≈0.1 で線形コストと概ね一致し、大口ではより高く、
    小口ではより低くなる。
    """
    # 線形 impact_rate*|d| ≈ impact_coef*|d|^1.5 を |d|=0.1 で合わせる
    impact_coef = impact_rate / (0.1**0.5)  # = impact_rate / 0.316
    return ExecutionModel(
        fee_rate=fee_rate,
        spread_rate=spread_rate,
        impact_coef=impact_coef,
        n_slices=n_slices,
        cost_multiplier=cost_multiplier,
    )
