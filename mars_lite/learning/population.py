"""
Population管理モジュール

PBT・MAP-Elites用の複数個体管理
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Individual:
    """
    個体（エージェント＋ハイパーパラメータ）

    Attributes:
        id: 個体ID
        hyperparams: ハイパーパラメータ辞書
        fitness: 評価スコア（高いほど良い）
        model_path: モデル保存パス
        behavior_desc: 行動記述子（MAP-Elites用）
        generation: 世代番号
        parent_id: 親個体ID（コピー元）
    """

    id: str
    hyperparams: Dict[str, Any]
    fitness: float = 0.0
    model_path: Optional[str] = None
    behavior_desc: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "id": self.id,
            "hyperparams": self.hyperparams,
            "fitness": self.fitness,
            "model_path": self.model_path,
            "behavior_desc": self.behavior_desc,
            "generation": self.generation,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Individual":
        """辞書から復元"""
        return cls(**data)


class PopulationManager:
    """
    Population管理クラス

    PBTとMAP-Elitesの両方をサポート。
    個体の生成・評価・淘汰・保存を管理。
    """

    # デフォルトのハイパーパラメータ範囲
    DEFAULT_HYPERPARAM_RANGES = {
        "lr": (1e-5, 1e-3, "log"),  # (min, max, scale)
        "clip_range": (0.1, 0.4, "linear"),
        "lambda_cost": (0.5, 2.0, "linear"),
        "lambda_risk": (1e-4, 1e-2, "log"),
        "time_horizon_bias": (0.5, 2.0, "linear"),
    }

    def __init__(
        self,
        population_size: int = 16,
        hyperparam_ranges: Optional[Dict] = None,
        save_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            population_size: 個体数
            hyperparam_ranges: ハイパーパラメータ範囲
            save_dir: 保存ディレクトリ
            seed: 乱数シード
        """
        self.population_size = population_size
        self.hyperparam_ranges = hyperparam_ranges or self.DEFAULT_HYPERPARAM_RANGES
        self.save_dir = Path(save_dir) if save_dir else None

        self.rng = np.random.default_rng(seed)

        self.population: List[Individual] = []
        self.generation = 0
        self._id_counter = 0

    def _generate_id(self) -> str:
        """ユニークなIDを生成"""
        self._id_counter += 1
        return f"ind_{self.generation}_{self._id_counter}"

    def _sample_hyperparam(self, name: str) -> float:
        """ハイパーパラメータをサンプリング"""
        min_val, max_val, scale = self.hyperparam_ranges[name]

        if scale == "log":
            return float(np.exp(self.rng.uniform(np.log(min_val), np.log(max_val))))
        else:  # linear
            return float(self.rng.uniform(min_val, max_val))

    def _perturb_hyperparam(
        self, value: float, name: str, factor: float = 1.2
    ) -> float:
        """ハイパーパラメータを摂動"""
        min_val, max_val, scale = self.hyperparam_ranges[name]

        # ランダムに増減
        if self.rng.random() < 0.5:
            new_value = value * factor
        else:
            new_value = value / factor

        # 範囲内にクリップ
        return float(np.clip(new_value, min_val, max_val))

    def initialize_population(self) -> List[Individual]:
        """
        ランダムに初期population生成

        Returns:
            生成された個体リスト
        """
        self.population = []

        for _ in range(self.population_size):
            hyperparams = {
                name: self._sample_hyperparam(name)
                for name in self.hyperparam_ranges.keys()
            }

            individual = Individual(
                id=self._generate_id(),
                hyperparams=hyperparams,
                generation=self.generation,
            )
            self.population.append(individual)

        return self.population

    def get_top_k(self, k: int) -> List[Individual]:
        """上位k個体を取得"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:k]

    def get_bottom_k(self, k: int) -> List[Individual]:
        """下位k個体を取得"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        return sorted_pop[:k]

    def pbt_step(
        self,
        exploit_ratio: float = 0.2,
        perturb_factor: float = 1.2,
    ) -> List[Individual]:
        """
        PBT淘汰・増殖ステップ

        下位個体を上位個体からコピーし、ハイパーパラメータを摂動。

        Args:
            exploit_ratio: 淘汰される/ コピー元となる割合
            perturb_factor: 摂動倍率

        Returns:
            更新後のpopulation
        """
        self.generation += 1

        k = max(1, int(len(self.population) * exploit_ratio))
        top_k = self.get_top_k(k)
        bottom_k = self.get_bottom_k(k)

        # 下位個体を上位からコピー＋摂動
        for weak in bottom_k:
            # ランダムに上位個体を選択
            parent = self.rng.choice(top_k)

            # ハイパーパラメータをコピー＋摂動
            new_hyperparams = {}
            for name, value in parent.hyperparams.items():
                new_hyperparams[name] = self._perturb_hyperparam(
                    value, name, perturb_factor
                )

            # 個体を更新
            weak.id = self._generate_id()
            weak.hyperparams = new_hyperparams
            weak.fitness = 0.0  # リセット
            weak.generation = self.generation
            weak.parent_id = parent.id
            weak.model_path = parent.model_path  # モデル重みはコピー

        return self.population

    def save_population(self, filename: str = "population.json"):
        """populationを保存"""
        if self.save_dir is None:
            raise ValueError("save_dir is not set")

        self.save_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "generation": self.generation,
            "population": [ind.to_dict() for ind in self.population],
        }

        with open(self.save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_population(self, filename: str = "population.json"):
        """populationを読み込み"""
        if self.save_dir is None:
            raise ValueError("save_dir is not set")

        with open(self.save_dir / filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.generation = data["generation"]
        self.population = [
            Individual.from_dict(ind_data) for ind_data in data["population"]
        ]
