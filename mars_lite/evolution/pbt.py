"""
Population Based Training (PBT) 実行モジュール

定期的な評価と淘汰・増殖サイクルを管理
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..learning.population import PopulationManager


class PBTRunner:
    """
    PBT実行クラス

    学習と評価を交互に行い、定期的にpopulationを更新する。
    """

    def __init__(
        self,
        population_manager: PopulationManager,
        create_env_fn: Callable,
        create_agent_fn: Callable,
        eval_fn: Callable,
        eval_interval: int = 50000,
        exploit_ratio: float = 0.2,
        perturb_factor: float = 1.2,
        save_dir: Optional[str] = None,
    ):
        """
        Args:
            population_manager: Population管理インスタンス
            create_env_fn: 環境生成関数
            create_agent_fn: エージェント生成関数(env, hyperparams) -> agent
            eval_fn: 評価関数(agent, env) -> fitness
            eval_interval: 評価・更新間隔（ステップ数）
            exploit_ratio: 淘汰比率
            perturb_factor: 摂動倍率
            save_dir: 保存ディレクトリ
        """
        self.population_manager = population_manager
        self.create_env_fn = create_env_fn
        self.create_agent_fn = create_agent_fn
        self.eval_fn = eval_fn
        self.eval_interval = eval_interval
        self.exploit_ratio = exploit_ratio
        self.perturb_factor = perturb_factor
        self.save_dir = Path(save_dir) if save_dir else None

        self.agents: Dict[str, Any] = {}  # id -> agent
        self.envs: Dict[str, Any] = {}  # id -> env
        self.total_steps = 0
        self.history: List[Dict] = []

    def initialize(self):
        """
        Populationとエージェントを初期化
        """
        # Populationを初期化
        if not self.population_manager.population:
            self.population_manager.initialize_population()

        # 各個体のエージェントを生成
        for ind in self.population_manager.population:
            env = self.create_env_fn()
            agent = self.create_agent_fn(env, ind.hyperparams)

            self.envs[ind.id] = env
            self.agents[ind.id] = agent

    def train_step(self, steps_per_individual: int = 2048) -> Dict[str, float]:
        """
        全個体を並列に学習（シミュレート）

        Args:
            steps_per_individual: 個体あたりステップ数

        Returns:
            個体IDごとの学習結果
        """
        results = {}

        for ind in self.population_manager.population:
            agent = self.agents[ind.id]

            # 学習実行
            agent.learn(total_timesteps=steps_per_individual, reset_num_timesteps=False)

            self.total_steps += steps_per_individual
            results[ind.id] = {
                "steps": steps_per_individual,
                "total_steps": self.total_steps,
            }

        return results

    def evaluate_population(self) -> Dict[str, float]:
        """
        全個体を評価しfitnessを更新

        Returns:
            個体IDごとのfitness
        """
        fitness_map = {}

        for ind in self.population_manager.population:
            agent = self.agents[ind.id]
            env = self.envs[ind.id]

            # 評価
            fitness = self.eval_fn(agent, env)
            ind.fitness = fitness
            fitness_map[ind.id] = fitness

        return fitness_map

    def evolve(self):
        """
        PBT進化ステップ（淘汰・増殖）
        """
        # 下位個体のIDを記録
        bottom_ids = [
            ind.id
            for ind in self.population_manager.get_bottom_k(
                max(
                    1, int(len(self.population_manager.population) * self.exploit_ratio)
                )
            )
        ]

        # PBT更新
        self.population_manager.pbt_step(
            exploit_ratio=self.exploit_ratio,
            perturb_factor=self.perturb_factor,
        )

        # 更新された個体のエージェントを再生成
        for ind in self.population_manager.population:
            if ind.id not in self.agents:
                # 新しいIDの個体（更新された個体）
                env = self.create_env_fn()

                # 親のモデル重みをコピーして新しいエージェント生成
                if ind.parent_id and ind.parent_id in self.agents:
                    # 注: 実際の実装ではモデル重みのコピーが必要
                    agent = self.create_agent_fn(env, ind.hyperparams)
                else:
                    agent = self.create_agent_fn(env, ind.hyperparams)

                self.envs[ind.id] = env
                self.agents[ind.id] = agent

        # 古いエージェントをクリーンアップ
        current_ids = {ind.id for ind in self.population_manager.population}
        for old_id in list(self.agents.keys()):
            if old_id not in current_ids:
                del self.agents[old_id]
                del self.envs[old_id]

    def run(
        self,
        total_steps: int,
        steps_per_iteration: int = 2048,
        log_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """
        PBT学習ループを実行

        Args:
            total_steps: 総ステップ数
            steps_per_iteration: イテレーションあたりステップ数
            log_fn: ログ関数

        Returns:
            学習履歴
        """
        self.initialize()

        iterations = total_steps // steps_per_iteration
        steps_since_eval = 0

        for i in range(iterations):
            # 学習
            train_results = self.train_step(steps_per_iteration)
            steps_since_eval += steps_per_iteration

            # 評価・進化
            if steps_since_eval >= self.eval_interval:
                fitness_map = self.evaluate_population()

                # ログ記録
                log_entry = {
                    "iteration": i,
                    "total_steps": self.total_steps,
                    "generation": self.population_manager.generation,
                    "fitness_map": fitness_map,
                    "best_fitness": max(fitness_map.values()),
                    "mean_fitness": np.mean(list(fitness_map.values())),
                }
                self.history.append(log_entry)

                if log_fn:
                    log_fn(log_entry)

                # 進化
                self.evolve()
                steps_since_eval = 0

        # 最終保存
        if self.save_dir:
            self.save()

        return self.history

    def save(self):
        """状態を保存"""
        if self.save_dir is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Population保存
        self.population_manager.save_dir = self.save_dir
        self.population_manager.save_population()

        # 履歴保存
        with open(self.save_dir / "history.json", "w", encoding="utf-8") as f:
            # floatをシリアライズ可能に変換
            serializable_history = []
            for entry in self.history:
                s_entry = {}
                for k, v in entry.items():
                    if isinstance(v, dict):
                        s_entry[k] = {kk: float(vv) for kk, vv in v.items()}
                    elif isinstance(v, (np.floating, np.integer)):
                        s_entry[k] = float(v)
                    else:
                        s_entry[k] = v
                serializable_history.append(s_entry)

            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
