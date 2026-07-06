"""
評価スクリプト

学習済みエージェントを評価し、様式化された挙動を確認
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mars_lite.data.preprocessing import preprocess_ohlcv
from mars_lite.env.mars_lite_env import MarsLiteEnv

from mars_lite.learning.agent import load_agent
from mars_lite.utils.config import MarsLiteConfig, create_env_kwargs
from mars_lite.utils.metrics import (
    calc_execution_metrics,
)


def create_sample_data(n_bars: int = 10000) -> pd.DataFrame:
    """サンプルデータ生成"""
    np.random.seed(123)  # 学習と異なるシード

    returns = np.random.randn(n_bars) * 0.002
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.003)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.003)
    open_ = low + (high - low) * np.random.rand(n_bars)

    tod = np.arange(n_bars) % 1440
    base_volume = 1000 * (1 + 0.5 * np.cos(2 * np.pi * tod / 1440))
    volume = base_volume * np.random.exponential(1, n_bars)

    timestamps = pd.date_range("2024-02-01", periods=n_bars, freq="1min")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return preprocess_ohlcv(df)


def plot_stylized_behavior(execution_history: pd.DataFrame, output_dir: Path):
    """
    様式化された挙動のプロット

    1. 時刻別執行量（U字型パターン）
    2. ボラティリティ vs 執行量
    3. 残時間 vs 執行量
    4. 執行量分布
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 時刻別執行量
    ax1 = axes[0, 0]
    steps = execution_history["step"]
    quantities = execution_history["quantity"]
    ax1.bar(steps, quantities, alpha=0.7)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Execution Quantity")
    ax1.set_title("Time Distribution of Executions")

    # 2. POV分布
    ax2 = axes[0, 1]
    povs = execution_history["pov"]
    ax2.hist(povs, bins=20, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Participation Rate (POV)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("POV Distribution")

    # 3. 累積執行量
    ax3 = axes[1, 0]
    cumulative_qty = quantities.cumsum()
    ax3.plot(steps, cumulative_qty, marker="o", markersize=2)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Cumulative Quantity")
    ax3.set_title("Cumulative Execution")

    # 4. インパクト vs 数量
    ax4 = axes[1, 1]
    impacts = execution_history["impact_pct"]
    ax4.scatter(quantities, impacts, alpha=0.6)
    ax4.set_xlabel("Execution Quantity")
    ax4.set_ylabel("Impact (%)")
    ax4.set_title("Quantity vs Impact")

    plt.tight_layout()
    plt.savefig(output_dir / "stylized_behavior.png", dpi=150)
    plt.close()

    print(f"プロットを保存しました: {output_dir / 'stylized_behavior.png'}")


def evaluate_stylized_behavior(
    execution_histories: list,
    initial_inventory: float,
) -> dict:
    """
    様式化された挙動を定量評価

    Returns:
        挙動メトリクス
    """
    all_metrics = []

    for hist in execution_histories:
        if len(hist) == 0:
            continue

        metrics = calc_execution_metrics(hist, initial_inventory)
        all_metrics.append(metrics)

    if not all_metrics:
        return {}

    # 平均化
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[f"mean_{key}"] = float(np.mean(values))
        avg_metrics[f"std_{key}"] = float(np.std(values))

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="MarS Lite 評価スクリプト")
    parser.add_argument("--model", type=str, required=True, help="モデルパス")
    parser.add_argument("--data", type=str, default=None, help="評価データパス")
    parser.add_argument(
        "--output", type=str, default="./eval_output", help="出力ディレクトリ"
    )
    parser.add_argument("--episodes", type=int, default=20, help="評価エピソード数")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設定読み込み
    model_dir = Path(args.model).parent
    config_path = model_dir / "config.json"
    if config_path.exists():
        config = MarsLiteConfig.load(str(config_path))
    else:
        config = MarsLiteConfig()

    print("=" * 60)
    print("MarS Lite 評価")
    print("=" * 60)

    # データ読み込み
    if args.data:
        df = pd.read_csv(args.data)
        df.columns = df.columns.str.lower()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        data = preprocess_ohlcv(df)
    else:
        print("サンプルデータを生成中...")
        data = create_sample_data(n_bars=20000)

    # 環境作成
    env_kwargs = create_env_kwargs(config)
    env = MarsLiteEnv(data=data, **env_kwargs)

    # エージェント読み込み
    print(f"モデル読み込み中: {args.model}")
    agent = load_agent(args.model, env)

    # 評価実行
    print(f"評価中... ({args.episodes} エピソード)")
    execution_histories = []
    episode_rewards = []

    for i in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        execution_histories.append(env.get_execution_history())
        episode_rewards.append(total_reward)

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{args.episodes} 完了")

    # 結果集計
    print("\n" + "=" * 60)
    print("評価結果")
    print("=" * 60)
    print(f"平均報酬: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")

    # 様式化された挙動の評価
    behavior_metrics = evaluate_stylized_behavior(
        execution_histories,
        config.initial_inventory,
    )

    print(f"平均取引回数: {behavior_metrics.get('mean_n_trades', 0):.1f}")
    print(f"平均POV: {behavior_metrics.get('mean_mean_pov', 0):.4f}")
    print(f"平均インパクト: {behavior_metrics.get('mean_mean_impact_pct', 0):.4%}")
    print(f"完了率: {behavior_metrics.get('mean_completion_rate', 0):.2%}")

    # プロット生成（最後のエピソード）
    if execution_histories:
        last_hist = execution_histories[-1]
        if len(last_hist) > 0:
            plot_stylized_behavior(last_hist, output_dir)

    # 結果保存
    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "n_episodes": args.episodes,
        **behavior_metrics,
    }

    with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n結果を保存しました: {output_dir}")


if __name__ == "__main__":
    main()
