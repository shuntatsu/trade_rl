"""
Behavior Descriptor 計算ユーティリティ

エージェントの評価エピソードから行動特性を抽出する。
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def calculate_behavior_descriptors(
    execution_history: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    実行履歴から Behavior Descriptor を計算

    Args:
        execution_history: MarsLiteEnv.execution_history
            [{step, price, target_pct, position, portfolio_value, ...}, ...]

    Returns:
        {
            "long_bias": float,      # エピソード平均ポジション [-1, 1]
            "vol_exposure": float,   # ポジションサイズの標準偏差 [0, 1]
        }
    """
    if not execution_history:
        return {"long_bias": 0.0, "vol_exposure": 0.0}

    df = pd.DataFrame(execution_history)

    # Long Bias: 平均ポジション
    # position は Units で保存されているため、target_pct を使用
    if "target_pct" in df.columns:
        positions = df["target_pct"].values
    elif "position" in df.columns:
        # position を正規化（-1〜1にスケーリング）
        # 実際には portfolio_value で割る必要があるが、簡易的に max で正規化
        positions = df["position"].values
        max_pos = np.abs(positions).max()
        if max_pos > 0:
            positions = positions / max_pos
        else:
            positions = np.zeros_like(positions)
    else:
        positions = np.zeros(len(df))

    long_bias = float(np.mean(positions))

    # Vol Exposure: ポジションサイズの変動
    # 正規化（0〜1）
    vol_exposure = float(np.std(positions))
    vol_exposure = np.clip(vol_exposure, 0.0, 1.0)

    return {"long_bias": long_bias, "vol_exposure": vol_exposure}


def evaluate_agent_with_descriptors(
    agent, eval_env, n_episodes: int = 3, abort_event=None
) -> Dict[str, Any]:
    """
    エージェントを評価し、Fitness と Behavior Descriptor を計算

    Args:
        agent: PPO エージェント
        eval_env: 評価環境（MarsLiteEnv）
        n_episodes: 評価エピソード数
        abort_event: 停止イベント

    Returns:
        {
            "fitness": float,        # 平均エピソード報酬
            "long_bias": float,
            "vol_exposure": float,
            "episode_rewards": list
        }
    """
    episode_rewards = []
    all_history = []

    for _ in range(n_episodes):
        if abort_event and abort_event.is_set():
            break

        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if abort_event and abort_event.is_set():
                break
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_rewards.append(total_reward)

        # 実行履歴を取得
        if hasattr(eval_env, "execution_history"):
            all_history.extend(eval_env.execution_history)
        elif hasattr(eval_env, "envs") and hasattr(
            eval_env.envs[0], "execution_history"
        ):
            # VecEnv の場合
            all_history.extend(eval_env.envs[0].execution_history)

    # Fitness: 平均報酬
    fitness = float(np.mean(episode_rewards))

    # Behavior Descriptors
    descriptors = calculate_behavior_descriptors(all_history)

    return {
        "fitness": fitness,
        "long_bias": descriptors["long_bias"],
        "vol_exposure": descriptors["vol_exposure"],
        "episode_rewards": episode_rewards,
    }
