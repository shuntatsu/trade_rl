"""
MAP-Elites Grid Archive

Quality Diversity アルゴリズムのコアコンポーネント。
行動特性空間をグリッドに分割し、各セルで最高性能の個体を保持する。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Individual:
    """
    進化個体（エージェント）
    """

    # モデルパス（PPO の .zip ファイル）
    model_path: Optional[str] = None

    # Fitness（品質指標）
    fitness: float = -np.inf

    # Behavior Descriptor（行動特性）
    # - long_bias: エピソード平均ポジション [-1, 1]
    # - vol_exposure: ポジションサイズの標準偏差 [0, 1]
    long_bias: float = 0.0
    vol_exposure: float = 0.0

    # ハイパーパラメータ
    hyperparams: Dict[str, Any] = None

    # メタデータ
    generation: int = 0
    training_steps: int = 0

    def __post_init__(self):
        if self.hyperparams is None:
            self.hyperparams = {}


class GridArchive:
    """
    MAP-Elites Grid Archive

    行動特性空間を離散化し、各セルで最高適応度の個体を保持。
    """

    def __init__(
        self,
        long_bias_bins: int = 5,
        vol_exposure_bins: int = 5,
        long_bias_range: Tuple[float, float] = (-1.0, 1.0),
        vol_exposure_range: Tuple[float, float] = (0.0, 1.0),
    ):
        """
        Args:
            long_bias_bins: Long Bias 軸の分割数
            vol_exposure_bins: Vol Exposure 軸の分割数
            long_bias_range: Long Bias の範囲
            vol_exposure_range: Vol Exposure の範囲
        """
        self.long_bias_bins = long_bias_bins
        self.vol_exposure_bins = vol_exposure_bins
        self.long_bias_range = long_bias_range
        self.vol_exposure_range = vol_exposure_range

        # Grid: (long_bias_bins, vol_exposure_bins)
        # 各セルは Individual または None
        self.grid = np.empty((long_bias_bins, vol_exposure_bins), dtype=object)

        # 統計
        self.total_insertions = 0
        self.successful_insertions = 0

    def get_cell_index(self, long_bias: float, vol_exposure: float) -> Tuple[int, int]:
        """
        Behavior Descriptor からグリッドインデックスを計算

        Returns:
            (i, j): グリッドインデックス
        """
        # Clamp
        lb = np.clip(long_bias, self.long_bias_range[0], self.long_bias_range[1])
        ve = np.clip(
            vol_exposure, self.vol_exposure_range[0], self.vol_exposure_range[1]
        )

        # Normalize to [0, 1]
        lb_norm = (lb - self.long_bias_range[0]) / (
            self.long_bias_range[1] - self.long_bias_range[0]
        )
        ve_norm = (ve - self.vol_exposure_range[0]) / (
            self.vol_exposure_range[1] - self.vol_exposure_range[0]
        )

        # Discretize
        i = int(lb_norm * self.long_bias_bins)
        j = int(ve_norm * self.vol_exposure_bins)

        # Edge case: 1.0 → last bin
        i = min(i, self.long_bias_bins - 1)
        j = min(j, self.vol_exposure_bins - 1)

        return (i, j)

    def add(self, individual: Individual) -> bool:
        """
        個体をアーカイブに追加

        Returns:
            True if added/replaced, False if rejected
        """
        self.total_insertions += 1

        i, j = self.get_cell_index(individual.long_bias, individual.vol_exposure)

        current = self.grid[i, j]

        # セルが空 or 現在の個体より高適応度
        if current is None or individual.fitness > current.fitness:
            self.grid[i, j] = individual
            self.successful_insertions += 1
            return True

        return False

    def get_all_individuals(self) -> List[Individual]:
        """アーカイブ内の全個体を取得"""
        individuals = []
        for i in range(self.long_bias_bins):
            for j in range(self.vol_exposure_bins):
                ind = self.grid[i, j]
                if ind is not None:
                    individuals.append(ind)
        return individuals

    def get_coverage(self) -> float:
        """グリッドカバレッジ（占有率）を計算"""
        filled = sum(1 for ind in self.grid.flat if ind is not None)
        total = self.long_bias_bins * self.vol_exposure_bins
        return filled / total

    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        individuals = self.get_all_individuals()

        if not individuals:
            return {
                "coverage": 0.0,
                "num_individuals": 0,
                "max_fitness": -np.inf,
                "mean_fitness": -np.inf,
                "total_insertions": self.total_insertions,
                "successful_insertions": self.successful_insertions,
            }

        fitnesses = [ind.fitness for ind in individuals]

        return {
            "coverage": self.get_coverage(),
            "num_individuals": len(individuals),
            "max_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "total_insertions": self.total_insertions,
            "successful_insertions": self.successful_insertions,
        }

    def save(self, path: str):
        """アーカイブを保存"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # メタデータを保存
        metadata = {
            "long_bias_bins": self.long_bias_bins,
            "vol_exposure_bins": self.vol_exposure_bins,
            "long_bias_range": self.long_bias_range,
            "vol_exposure_range": self.vol_exposure_range,
            "stats": self.get_stats(),
        }

        with open(path / "archive_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # 各個体の情報を保存
        individuals_data = []
        for i in range(self.long_bias_bins):
            for j in range(self.vol_exposure_bins):
                ind = self.grid[i, j]
                if ind is not None:
                    individuals_data.append(
                        {
                            "cell": (i, j),
                            "fitness": ind.fitness,
                            "long_bias": ind.long_bias,
                            "vol_exposure": ind.vol_exposure,
                            "hyperparams": ind.hyperparams,
                            "generation": ind.generation,
                            "training_steps": ind.training_steps,
                            "model_path": ind.model_path,
                        }
                    )

        with open(path / "individuals.json", "w") as f:
            json.dump(individuals_data, f, indent=2)

        print(f"[GridArchive] Saved to {path}")
        print(f"  Coverage: {self.get_coverage():.1%}")
        print(f"  Individuals: {len(individuals_data)}")
