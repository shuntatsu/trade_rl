"""
MAP-Elites (Quality Diversity) モジュール

行動特性に基づくグリッドで多様な個体を保存
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..learning.population import Individual


@dataclass
class BehaviorDescriptor:
    """
    行動記述子

    MAP-Elitesのグリッド軸を定義。
    """

    name: str
    min_val: float
    max_val: float
    n_bins: int

    def get_bin(self, value: float) -> int:
        """値をビンにマップ"""
        # クリップ
        value = np.clip(value, self.min_val, self.max_val)

        # ビン計算
        normalized = (value - self.min_val) / (self.max_val - self.min_val + 1e-8)
        bin_idx = int(normalized * self.n_bins)

        # 境界調整
        return min(bin_idx, self.n_bins - 1)


class MAPElitesArchive:
    """
    MAP-Elitesアーカイブ

    行動記述子のグリッドに最良個体を保存。
    Quality Diversity (QD) を実現。
    """

    def __init__(
        self,
        behavior_descriptors: List[BehaviorDescriptor],
        save_dir: Optional[str] = None,
    ):
        """
        Args:
            behavior_descriptors: 行動記述子リスト（グリッド軸）
            save_dir: 保存ディレクトリ
        """
        self.descriptors = behavior_descriptors
        self.save_dir = Path(save_dir) if save_dir else None

        # グリッド形状を計算
        self.grid_shape = tuple(d.n_bins for d in self.descriptors)

        # アーカイブ（None = 空セル）
        self.archive: Dict[Tuple[int, ...], Individual] = {}

        # 統計
        self.n_insertions = 0
        self.n_replacements = 0

    def _get_cell(self, behavior: Dict[str, float]) -> Tuple[int, ...]:
        """行動記述から対応するセルを取得"""
        cell = []
        for desc in self.descriptors:
            if desc.name not in behavior:
                raise ValueError(f"Missing behavior descriptor: {desc.name}")
            cell.append(desc.get_bin(behavior[desc.name]))
        return tuple(cell)

    def add(
        self,
        individual: Individual,
        behavior: Dict[str, float],
        fitness: float,
    ) -> bool:
        """
        個体をアーカイブに追加

        セルが空か、既存個体より優れていれば追加。

        Args:
            individual: 追加する個体
            behavior: 行動記述子の値
            fitness: 評価スコア

        Returns:
            追加されたかどうか
        """
        cell = self._get_cell(behavior)

        # 個体の情報を更新
        individual.behavior_desc = behavior
        individual.fitness = fitness

        # セルが空か確認
        if cell not in self.archive:
            self.archive[cell] = individual
            self.n_insertions += 1
            return True

        # 既存個体と比較
        existing = self.archive[cell]
        if fitness > existing.fitness:
            self.archive[cell] = individual
            self.n_replacements += 1
            return True

        return False

    def get(self, cell: Tuple[int, ...]) -> Optional[Individual]:
        """セルから個体を取得"""
        return self.archive.get(cell)

    def get_by_behavior(self, behavior: Dict[str, float]) -> Optional[Individual]:
        """行動記述から個体を取得"""
        cell = self._get_cell(behavior)
        return self.get(cell)

    def get_nearest(self, behavior: Dict[str, float]) -> Optional[Individual]:
        """
        最も近いセルの個体を取得

        行動記述に最も近い非空セルを見つける。
        """
        if not self.archive:
            return None

        target_cell = self._get_cell(behavior)

        # 完全一致があればそれを返す
        if target_cell in self.archive:
            return self.archive[target_cell]

        # 最近傍を探索
        min_dist = float("inf")
        nearest = None

        for cell, ind in self.archive.items():
            dist = sum((a - b) ** 2 for a, b in zip(cell, target_cell))
            if dist < min_dist:
                min_dist = dist
                nearest = ind

        return nearest

    def get_all(self) -> List[Individual]:
        """全個体を取得"""
        return list(self.archive.values())

    def get_coverage(self) -> float:
        """グリッドカバレッジ（埋まっているセルの割合）"""
        total_cells = 1
        for d in self.descriptors:
            total_cells *= d.n_bins
        return len(self.archive) / total_cells

    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        if not self.archive:
            return {
                "n_cells": 0,
                "coverage": 0.0,
                "mean_fitness": 0.0,
                "max_fitness": 0.0,
                "min_fitness": 0.0,
            }

        fitnesses = [ind.fitness for ind in self.archive.values()]

        return {
            "n_cells": len(self.archive),
            "coverage": self.get_coverage(),
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "min_fitness": np.min(fitnesses),
            "n_insertions": self.n_insertions,
            "n_replacements": self.n_replacements,
        }

    def save(self, filename: str = "map_elites_archive.json"):
        """アーカイブを保存"""
        if self.save_dir is None:
            raise ValueError("save_dir is not set")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # セルキーを文字列に変換
        archive_data = {}
        for cell, ind in self.archive.items():
            key = ",".join(map(str, cell))
            archive_data[key] = ind.to_dict()

        data = {
            "descriptors": [
                {
                    "name": d.name,
                    "min_val": d.min_val,
                    "max_val": d.max_val,
                    "n_bins": d.n_bins,
                }
                for d in self.descriptors
            ],
            "archive": archive_data,
            "statistics": self.get_statistics(),
        }

        with open(self.save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filename: str = "map_elites_archive.json"):
        """アーカイブを読み込み"""
        if self.save_dir is None:
            raise ValueError("save_dir is not set")

        with open(self.save_dir / filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # アーカイブを復元
        self.archive = {}
        for key, ind_data in data["archive"].items():
            cell = tuple(map(int, key.split(",")))
            self.archive[cell] = Individual.from_dict(ind_data)


def compute_behavior_descriptors(
    execution_history: "pd.DataFrame",
    sigma_threshold_quantile: float = 0.75,
) -> Dict[str, float]:
    """
    執行履歴から行動記述子を計算

    Args:
        execution_history: 執行履歴DataFrame
        sigma_threshold_quantile: 高ボラ判定の分位点

    Returns:
        {
            "aggressiveness": 平均participation rate,
            "volatility_tolerance": 高ボラ区間での相対執行量,
        }
    """

    if len(execution_history) == 0:
        return {
            "aggressiveness": 0.0,
            "volatility_tolerance": 0.0,
        }

    # Aggressiveness: 平均participation rate
    aggressiveness = execution_history["pov"].mean()

    # Volatility Tolerance: 高ボラ区間での執行量比率
    # （sigmaがしきい値以上の区間での平均執行量 / 全体平均執行量）
    if "sigma" in execution_history.columns:
        sigma = execution_history["sigma"]
    else:
        # sigmaがない場合はpovで代替
        sigma = execution_history["pov"]

    threshold = sigma.quantile(sigma_threshold_quantile)
    high_vol_mask = sigma >= threshold

    if high_vol_mask.sum() > 0:
        high_vol_qty = execution_history.loc[high_vol_mask, "quantity"].mean()
        mean_qty = execution_history["quantity"].mean()
        volatility_tolerance = high_vol_qty / (mean_qty + 1e-8)
    else:
        volatility_tolerance = 1.0

    return {
        "aggressiveness": float(aggressiveness),
        "volatility_tolerance": float(volatility_tolerance),
    }
