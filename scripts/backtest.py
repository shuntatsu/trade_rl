"""
バックテストスクリプト

学習済みモデルでテストデータを評価し、パフォーマンス指標を計算。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from mars_lite.data.preprocessing import preprocess_ohlcv
from mars_lite.data.multi_timeframe_loader import MultiTimeframeLoader
from mars_lite.data.data_split import split_temporal_multi_tf
from mars_lite.env.multi_tf_env import MarsLiteMultiTFEnv
from mars_lite.env.mars_lite_env import MarsLiteEnv
from mars_lite.learning.model_manager import ModelManager, get_model_manager
from mars_lite.utils.config import MarsLiteConfig, create_env_kwargs


def calculate_metrics(
    episode_rewards: List[float],
    episode_lengths: List[int],
    trade_history: List[Dict],
) -> Dict[str, Any]:
    """
    パフォーマンス指標を計算
    
    Args:
        episode_rewards: エピソード報酬リスト
        episode_lengths: エピソード長リスト
        trade_history: 取引履歴
        
    Returns:
        指標辞書
    """
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    
    # 基本統計
    metrics = {
        "n_episodes": len(episode_rewards),
        "mean_reward": float(np.mean(rewards)) if len(rewards) > 0 else 0.0,
        "std_reward": float(np.std(rewards)) if len(rewards) > 0 else 0.0,
        "min_reward": float(np.min(rewards)) if len(rewards) > 0 else 0.0,
        "max_reward": float(np.max(rewards)) if len(rewards) > 0 else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if len(lengths) > 0 else 0.0,
    }
    
    # リターン系指標
    if len(rewards) > 1:
        # シャープレシオ（年率換算は省略、単純な mean/std）
        if np.std(rewards) > 0:
            metrics["sharpe_ratio"] = float(np.mean(rewards) / np.std(rewards))
        else:
            metrics["sharpe_ratio"] = 0.0
        
        # 累積リターン
        cumulative = np.cumsum(rewards)
        metrics["total_return"] = float(cumulative[-1])
        
        # 最大ドローダウン
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics["max_drawdown"] = float(np.max(drawdown))
        
        # 勝率（正の報酬の割合）
        wins = np.sum(rewards > 0)
        metrics["win_rate"] = float(wins / len(rewards))
    
    # 取引統計
    if trade_history:
        n_trades = len(trade_history)
        metrics["n_trades"] = n_trades
        
        if n_trades > 0:
            impacts = [t.get("impact_pct", 0) for t in trade_history]
            quantities = [t.get("quantity", 0) for t in trade_history]
            povs = [t.get("pov", 0) for t in trade_history]
            
            metrics["mean_impact"] = float(np.mean(impacts))
            metrics["total_quantity"] = float(np.sum(quantities))
            metrics["mean_pov"] = float(np.mean(povs))
    
    return metrics


def run_backtest(
    agent,
    env,
    n_episodes: int = 10,
    deterministic: bool = True,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    バックテストを実行
    
    Args:
        agent: 学習済みエージェント
        env: テスト環境
        n_episodes: 評価エピソード数
        deterministic: 決定的行動
        verbose: 出力レベル
        
    Returns:
        バックテスト結果
    """
    episode_rewards = []
    episode_lengths = []
    all_trade_history = []
    episode_details = []
    
    for ep in range(n_episodes):
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
        
        # 取引履歴を取得
        if hasattr(env, "get_execution_history"):
            exec_hist = env.get_execution_history()
            if len(exec_hist) > 0:
                trades = exec_hist.to_dict("records")
                all_trade_history.extend(trades)
                
                episode_details.append({
                    "episode": ep + 1,
                    "reward": total_reward,
                    "length": length,
                    "n_trades": len(trades),
                    "total_quantity": exec_hist["quantity"].sum(),
                    "mean_impact": exec_hist["impact_pct"].mean(),
                })
        
        if verbose >= 1:
            print(f"Episode {ep + 1}/{n_episodes}: Reward = {total_reward:.4f}, Length = {length}")
    
    # 指標計算
    metrics = calculate_metrics(episode_rewards, episode_lengths, all_trade_history)
    
    return {
        "metrics": metrics,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_details": episode_details,
        "trade_history": all_trade_history[:100],  # 最初の100件のみ
    }


def main():
    parser = argparse.ArgumentParser(description="MarS Lite バックテスト")
    parser.add_argument("--model", type=str, required=True, help="モデルID or パス")
    parser.add_argument("--data", type=str, required=True, help="データディレクトリ")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="通貨ペア")
    parser.add_argument("--start-date", type=str, default=None, help="開始日")
    parser.add_argument("--end-date", type=str, default=None, help="終了日")
    parser.add_argument("--episodes", type=int, default=10, help="評価エピソード数")
    parser.add_argument("--output", type=str, default="./output/backtest", help="出力ディレクトリ")
    parser.add_argument("--config", type=str, default=None, help="設定ファイル")
    parser.add_argument("--verbose", type=int, default=1, help="出力レベル")
    parser.add_argument("--models-dir", type=str, default="./output/models", help="モデルディレクトリ")
    args = parser.parse_args()
    
    # 設定読み込み
    if args.config:
        config = MarsLiteConfig.load(args.config)
    else:
        config = MarsLiteConfig()
    
    config.symbol = args.symbol
    
    print("=" * 60)
    print("MarS Lite バックテスト")
    print("=" * 60)
    print(f"モデル: {args.model}")
    print(f"通貨ペア: {args.symbol}")
    print(f"エピソード数: {args.episodes}")
    
    # モデル読み込み
    print("\nモデルを読み込み中...")
    manager = get_model_manager(args.models_dir)
    
    # モデルIDかパスかを判定
    if Path(args.model).exists():
        # パス指定
        from stable_baselines3 import PPO
        agent = PPO.load(args.model)
        metadata = None
    else:
        # ID指定
        agent, metadata = manager.load(args.model)
    
    if metadata:
        print(f"  学習ステップ: {metadata.total_timesteps:,}")
        print(f"  平均報酬: {metadata.mean_reward:.4f}")
    
    # データ読み込み
    print("\nデータを読み込み中...")
    loader = MultiTimeframeLoader(
        data_dir=Path(args.data),
        timeframes=list(config.timeframes),
        symbol=args.symbol,
        days=config.data_days,
        preprocess=True,
    )
    
    loader.load_all(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    base_data = loader.get_base_timeframe()
    higher_tf_data = loader.get_higher_timeframes()
    
    print(f"  ベースデータ: {len(base_data):,}バー")
    
    # テストデータを分割（後半20%を使用）
    if config.use_multi_tf and higher_tf_data:
        all_data = {config.base_timeframe: base_data, **higher_tf_data}
        _, _, test_data = split_temporal_multi_tf(
            all_data,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
        )
        test_base = test_data[config.base_timeframe]
        test_higher = {k: v for k, v in test_data.items() if k != config.base_timeframe}
    else:
        n = len(base_data)
        test_base = base_data.iloc[int(n * 0.8):]
        test_higher = {}
    
    print(f"  テストデータ: {len(test_base):,}バー")
    
    # 環境作成
    print("\n環境を作成中...")
    env_kwargs = create_env_kwargs(config)
    
    if config.use_multi_tf and test_higher:
        env = MarsLiteMultiTFEnv(
            data_1m=test_base,
            higher_tf_data=test_higher,
            **env_kwargs
        )
    else:
        single_kwargs = {k: v for k, v in env_kwargs.items() if k != "higher_tf_lookback"}
        env = MarsLiteEnv(data=test_base, **single_kwargs)
    
    # バックテスト実行
    print("\nバックテスト実行中...")
    results = run_backtest(
        agent=agent,
        env=env,
        n_episodes=args.episodes,
        deterministic=True,
        verbose=args.verbose,
    )
    
    # 結果表示
    print("\n" + "=" * 60)
    print("結果")
    print("=" * 60)
    
    m = results["metrics"]
    print(f"平均報酬: {m['mean_reward']:.4f} ± {m['std_reward']:.4f}")
    print(f"シャープレシオ: {m.get('sharpe_ratio', 'N/A')}")
    print(f"最大ドローダウン: {m.get('max_drawdown', 'N/A')}")
    print(f"勝率: {m.get('win_rate', 0) * 100:.1f}%")
    print(f"取引回数: {m.get('n_trades', 0)}")
    print(f"平均インパクト: {m.get('mean_impact', 0) * 100:.4f}%")
    
    # 結果保存
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(args.model).stem or "model"
    result_path = output_dir / f"backtest_{model_name}_{args.symbol}.json"
    
    # NumPy型をPython型に変換
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable = json.loads(
        json.dumps(results, default=convert_numpy)
    )
    
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を保存しました: {result_path}")


if __name__ == "__main__":
    main()
