"""
Ensemble Inference

複数エージェントの予測を統合して最終的な行動を決定する。
"""

from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import PPO


class EnsemblePolicy:
    """
    Ensemble Inference Policy

    複数エージェントの行動を平均化または投票により統合。
    """

    def __init__(
        self,
        agents: List[PPO],
        weights: Optional[List[float]] = None,
        method: str = "weighted_mean",
    ):
        """
        Args:
            agents: PPO エージェントのリスト
            weights: 各エージェントの重み（None なら均等）
            method: 統合方法
                - "mean": 単純平均
                - "weighted_mean": 重み付き平均
                - "median": 中央値
        """
        self.agents = agents

        if weights is None:
            weights = [1.0 / len(agents)] * len(agents)
        else:
            # 正規化
            total = sum(weights)
            weights = [w / total for w in weights]

        self.weights = weights
        self.method = method

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """
        Ensemble 予測

        Args:
            observation: 観測
            deterministic: 決定的予測かどうか

        Returns:
            action: アンサンブル行動
        """
        actions = []

        for agent in self.agents:
            action, _ = agent.predict(observation, deterministic=deterministic)
            actions.append(action)

        actions = np.array(actions)  # Shape: (n_agents, action_dim)

        if self.method == "mean":
            ensemble_action = np.mean(actions, axis=0)
        elif self.method == "weighted_mean":
            ensemble_action = np.average(actions, axis=0, weights=self.weights)
        elif self.method == "median":
            ensemble_action = np.median(actions, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return ensemble_action


def create_ensemble_from_archive(
    archive, top_k: int = 5, method: str = "weighted_mean"
) -> EnsemblePolicy:
    """
    GridArchive から Top-K エージェントのアンサンブルを作成

    Args:
        archive: GridArchive
        top_k: 上位K個体を選択
        method: 統合方法

    Returns:
        EnsemblePolicy
    """
    individuals = archive.get_all_individuals()

    if len(individuals) == 0:
        raise ValueError("Archive is empty!")

    # Fitness でソート
    individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)

    # Top-K を選択
    top_individuals = individuals[:top_k]

    # Agent をロード
    agents = []
    weights = []

    for ind in top_individuals:
        if ind.model_path is None:
            continue

        # Note: env は推論時に別途用意する必要がある
        # ここでは model_path のみ保存
        agents.append(ind.model_path)
        weights.append(ind.fitness)

    # Fitness を重みとして使用
    # 正規化は EnsemblePolicy 内で実施

    print(f"[Ensemble] Created from top {len(agents)} agents")
    for i, (path, weight) in enumerate(zip(agents, weights)):
        print(f"  {i}: {path} (fitness={weight:.2f})")

    # Note: 実際のロードは使用時に行う
    # ここでは情報を返すのみ
    return {"model_paths": agents, "weights": weights, "method": method}


def load_ensemble_agents(ensemble_info: Dict[str, Any], env) -> EnsemblePolicy:
    """
    Ensemble 情報から実際のエージェントをロード

    Args:
        ensemble_info: create_ensemble_from_archive の戻り値
        env: 環境（PPO.load 用）

    Returns:
        EnsemblePolicy
    """
    agents = []

    for model_path in ensemble_info["model_paths"]:
        agent = PPO.load(model_path, env=env)
        agents.append(agent)

    return EnsemblePolicy(
        agents=agents, weights=ensemble_info["weights"], method=ensemble_info["method"]
    )
