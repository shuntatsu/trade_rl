"""
Population-Based Training (PBT) Manager

並列学習中の個体集団を管理し、定期的にハイパーパラメータを進化させる。
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid_archive import GridArchive, Individual


class PBTManager:
    """
    Population-Based Training Manager

    - Exploit: 下位パフォーマーが上位パフォーマーの設定を継承
    - Explore: ハイパーパラメータをランダムに摂動
    """

    def __init__(
        self,
        population_size: int = 25,
        exploit_threshold: float = 0.25,  # 下位25%
        explore_factor: float = 0.2,  # ±20%の摂動
        perturb_prob: float = 0.5,  # 各パラメータの摂動確率
        hyperparam_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Args:
            population_size: 集団サイズ
            exploit_threshold: Exploit対象の下位比率
            explore_factor: 摂動の大きさ
            perturb_prob: 各ハイパーパラメータを摂動する確率
            hyperparam_bounds: ハイパーパラメータの範囲
        """
        self.population_size = population_size
        self.exploit_threshold = exploit_threshold
        self.explore_factor = explore_factor
        self.perturb_prob = perturb_prob

        # デフォルトのハイパーパラメータ範囲
        if hyperparam_bounds is None:
            hyperparam_bounds = {
                "learning_rate": (1e-5, 1e-3),
                "gamma": (0.95, 0.999),
                "ent_coef": (0.0, 0.1),
                "clip_range": (0.1, 0.3),
            }
        self.hyperparam_bounds = hyperparam_bounds

        self.generation = 0

    def exploit_and_explore(
        self, population: List[Individual], archive: GridArchive
    ) -> List[Individual]:
        """
        Exploit と Explore を実行

        Args:
            population: 現在の集団
            archive: GridArchive（アーカイブから上位個体を参照）

        Returns:
            更新された集団
        """
        if len(population) < 2:
            return population

        # Fitness でソート
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Exploit: 下位個体を上位個体からコピー
        n_exploit = int(len(sorted_pop) * self.exploit_threshold)

        new_population = []

        for i, ind in enumerate(sorted_pop):
            if i < len(sorted_pop) - n_exploit:
                # 上位: そのまま維持（少しだけ Explore）
                new_ind = copy.deepcopy(ind)
                new_ind.hyperparams = self._explore(ind.hyperparams, mild=True)
                new_population.append(new_ind)
            else:
                # 下位: Exploit（上位からコピー）
                parent_idx = np.random.randint(0, len(sorted_pop) - n_exploit)
                parent = sorted_pop[parent_idx]

                new_ind = copy.deepcopy(ind)
                new_ind.hyperparams = copy.deepcopy(parent.hyperparams)
                new_ind.hyperparams = self._explore(new_ind.hyperparams, mild=False)
                new_ind.model_path = None  # リセット（親のモデルから再学習）
                new_ind.fitness = -np.inf

                new_population.append(new_ind)

        self.generation += 1

        for ind in new_population:
            ind.generation = self.generation

        return new_population

    def _explore(
        self, hyperparams: Dict[str, Any], mild: bool = False
    ) -> Dict[str, Any]:
        """
        ハイパーパラメータを摂動

        Args:
            hyperparams: 現在のハイパーパラメータ
            mild: True ならば摂動を小さく（上位個体用）

        Returns:
            摂動後のハイパーパラメータ
        """
        new_hyperparams = copy.deepcopy(hyperparams)

        factor = self.explore_factor if not mild else self.explore_factor * 0.5

        for key, value in new_hyperparams.items():
            # 摂動するかどうか
            if np.random.rand() > self.perturb_prob:
                continue

            # 範囲を取得
            if key not in self.hyperparam_bounds:
                continue

            low, high = self.hyperparam_bounds[key]

            # 摂動方向（×1.2 or ×0.8）
            if np.random.rand() < 0.5:
                new_value = value * (1.0 + factor)
            else:
                new_value = value * (1.0 - factor)

            # クリップ
            new_value = np.clip(new_value, low, high)
            new_hyperparams[key] = new_value

        return new_hyperparams

    def initialize_population(
        self, base_hyperparams: Dict[str, Any]
    ) -> List[Individual]:
        """
        初期集団を生成

        Args:
            base_hyperparams: ベースとなるハイパーパラメータ

        Returns:
            初期集団
        """
        population = []

        for i in range(self.population_size):
            # ランダムに摂動
            hyperparams = self._sample_hyperparams(base_hyperparams)

            ind = Individual(hyperparams=hyperparams, generation=0, training_steps=0)
            population.append(ind)

        return population

    def _sample_hyperparams(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """ハイパーパラメータをランダムサンプリング"""
        hyperparams = copy.deepcopy(base)

        for key in self.hyperparam_bounds:
            if key in hyperparams:
                low, high = self.hyperparam_bounds[key]
                # Log-uniform for learning_rate
                if key == "learning_rate":
                    hyperparams[key] = np.exp(
                        np.random.uniform(np.log(low), np.log(high))
                    )
                else:
                    hyperparams[key] = np.random.uniform(low, high)

        return hyperparams
