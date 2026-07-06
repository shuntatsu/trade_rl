"""
PPOエージェントモジュール

Stable-Baselines3ベースのPPOエージェント管理
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

HAS_SB3 = True

import gymnasium as gym


def create_ppo_agent(
    env: Union[gym.Env, VecEnv],
    learning_rate: Union[float, Callable[[float], float]] = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    policy_kwargs: Optional[Dict] = None,
    verbose: int = 1,
    device: str = "auto",
    seed: Optional[int] = None,
) -> "PPO":
    """
    PPOエージェントを生成

    Args:
        env: Gymnasium環境
        learning_rate: 学習率
        n_steps: 更新あたりステップ数
        batch_size: ミニバッチサイズ
        n_epochs: 更新あたりエポック数
        gamma: 割引率
        gae_lambda: GAEパラメータ
        clip_range: PPOクリップ範囲
        ent_coef: エントロピー係数
        vf_coef: 価値関数係数
        max_grad_norm: 最大勾配ノルム
        policy_kwargs: ポリシーネットワーク設定
        verbose: ログ出力レベル
        device: デバイス（"cpu", "cuda", "auto"）
        seed: 乱数シード

    Returns:
        PPOエージェント
    """
    if not HAS_SB3:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        )

    # デフォルトのポリシー設定
    if policy_kwargs is None:
        policy_kwargs = {
            "net_arch": dict(pi=[128, 128], vf=[128, 128]),
        }

    agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        device=device,
        seed=seed,
    )

    return agent


def train_agent(
    agent: "PPO",
    total_timesteps: int,
    callbacks: Optional[List[BaseCallback]] = None,
    log_interval: int = 10,
    eval_env: Optional[gym.Env] = None,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    save_path: Optional[str] = None,
) -> "PPO":
    """
    エージェントを学習

    Args:
        agent: PPOエージェント
        total_timesteps: 総学習ステップ数
        callbacks: コールバックリスト
        log_interval: ログ出力間隔
        eval_env: 評価用環境
        eval_freq: 評価頻度
        n_eval_episodes: 評価エピソード数
        save_path: モデル保存パス

    Returns:
        学習済みエージェント
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required.")

    callback_list = callbacks or []

    # 評価コールバック追加
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callback_list.append(eval_callback)

    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback_list if callback_list else None,
        log_interval=log_interval,
    )

    # 最終モデル保存
    if save_path:
        agent.save(Path(save_path) / "final_model")

    return agent


def evaluate_agent(
    agent: "PPO",
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    エージェントを評価

    Args:
        agent: 学習済みエージェント
        env: 評価環境
        n_episodes: 評価エピソード数
        deterministic: 決定的行動を使用

    Returns:
        評価結果（mean_reward, std_reward, execution_stats等）
    """
    episode_rewards = []
    episode_lengths = []
    execution_stats = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            length += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(length)

        # 執行履歴を取得（MarsLiteEnvの場合）
        if hasattr(env, "get_execution_history"):
            exec_hist = env.get_execution_history()
            if len(exec_hist) > 0:
                execution_stats.append(
                    {
                        "n_trades": len(exec_hist),
                        "mean_pov": exec_hist["pov"].mean(),
                        "total_quantity": exec_hist["quantity"].sum(),
                        "mean_impact": exec_hist["impact_pct"].mean(),
                    }
                )

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards,
        "execution_stats": execution_stats,
    }


def load_agent(path: str, env: gym.Env) -> "PPO":
    """
    保存済みエージェントを読み込み

    Args:
        path: モデルパス
        env: 環境（ポリシー検証用）

    Returns:
        PPOエージェント
    """
    if not HAS_SB3:
        raise ImportError("stable-baselines3 is required.")

    return PPO.load(path, env=env)
