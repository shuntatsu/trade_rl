"""
シードアンサンブルモジュール

金融RLは単一シードの運不運が大きい。複数シードで学習し推奨行動を平均する
ことで分散を低減する。さらにシード間の不一致度でポジションを縮小する
（意見が割れる＝不確実 → グロスを落とす）不確実性スケーリングも提供。

SeedEnsemble は agent.predict と同じインターフェースを持つため、
evaluate_agent_on_slice や推論APIにそのまま差し込める。
"""

from typing import List, Optional, Tuple

import numpy as np


class SeedEnsemble:
    """複数エージェントの行動を平均するアンサンブル方策"""

    def __init__(self, agents: List[object]):
        if not agents:
            raise ValueError("agents must not be empty")
        self.agents = agents
        self.device = getattr(agents[0], "device", "cpu")

    def _all_actions(self, obs: np.ndarray) -> np.ndarray:
        """各エージェントの決定的行動を (n_agents, action_dim) で返す"""
        acts = [
            np.asarray(a.predict(obs, deterministic=True)[0]).flatten()
            for a in self.agents
        ]
        return np.stack(acts, axis=0)

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """平均行動を返す（agent.predict互換）"""
        return self._all_actions(obs).mean(axis=0), None

    def disagreement(self, obs: np.ndarray) -> float:
        """
        シード間の不一致度 [0,1]

        各行動次元の標準偏差の平均を、行動レンジ(2)で正規化した値。
        意見が完全一致で0、大きくばらつくほど1に近づく。
        """
        acts = self._all_actions(obs)
        if len(acts) < 2:
            return 0.0
        return float(np.clip(acts.std(axis=0).mean() / 1.0, 0.0, 1.0))

    def save(self, dir_path) -> None:
        from pathlib import Path

        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        for i, a in enumerate(self.agents):
            a.save(str(d / f"seed_{i}"))

    @classmethod
    def load(cls, dir_path, device: str = "cpu") -> "SeedEnsemble":
        from pathlib import Path

        from stable_baselines3 import PPO

        d = Path(dir_path)
        paths = sorted(d.glob("seed_*.zip"))
        if not paths:
            raise FileNotFoundError(f"No seed_*.zip in {d}")
        return cls([PPO.load(str(p), device=device) for p in paths])


def train_ensemble(
    train_fn,
    fs,
    seeds: Optional[List[int]] = None,
    verbose: int = 0,
) -> SeedEnsemble:
    """
    複数シードで学習してSeedEnsembleを返す

    Args:
        train_fn: (fs, seed) -> agent
        fs: 学習用FeatureSet
        seeds: シードのリスト（デフォルト[0,1,2]）
    """
    seeds = seeds or [0, 1, 2]
    agents = []
    for s in seeds:
        if verbose:
            print(f"[Ensemble] training seed {s}...")
        agents.append(train_fn(fs, s))
    return SeedEnsemble(agents)
