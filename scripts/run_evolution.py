"""
Evolution Training Script

PBT-MAP-Elites による進化訓練を実行するスタンドアロンスクリプト。
UIを使わずに進化戦略を実行したい場合に使用。

Usage:
    python scripts/run_evolution.py --generations 20 --population 25
"""

import argparse
import sys
from pathlib import Path

# Project root をパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mars_lite.data import MultiSymbolLoader
from mars_lite.data.data_utils import split_nested_by_ratio
from mars_lite.env import MarsLiteTradingEnv
from mars_lite.evolution import EvolutionTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Run Evolution Training (PBT-MAP-Elites)"
    )
    parser.add_argument(
        "--generations", type=int, default=10, help="Number of generations"
    )
    parser.add_argument("--population", type=int, default=25, help="Population size")
    parser.add_argument(
        "--steps-per-gen", type=int, default=10000, help="Training steps per generation"
    )
    parser.add_argument(
        "--grid-bins", type=int, default=5, help="Grid bins (5x5 = 25 cells)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/evolution", help="Output directory"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()

    print("=" * 60)
    print("MarS Lite Evolution Training (PBT-MAP-Elites)")
    print("=" * 60)
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Steps/Gen: {args.steps_per_gen}")
    print(f"Grid: {args.grid_bins}x{args.grid_bins}")
    print("=" * 60)

    # データロード
    print("\n[1/3] Loading data...")
    data_dir = Path(args.data_dir)

    # シンボルをスキャン（上位50）
    try:
        import os

        symbols = [d for d in os.listdir(data_dir) if os.path.isdir(data_dir / d)]
        symbols = [s for s in symbols if s.endswith("USDT")]
        symbols = symbols[:50]

        if not symbols:
            raise FileNotFoundError("No USDT symbols found in data directory")

        print(f"  Found {len(symbols)} symbols")

        loader = MultiSymbolLoader(
            data_dir=data_dir,
            symbols=symbols,
            timeframes=["1m", "15m", "1h", "4h", "1d"],
            days=3650,
            preprocess=True,
        )
        data_dict = loader.load_all(limit_days=None)

        # Train/Val 分割（比率ベース: 70/15/15）
        train_dict, val_dict, _ = split_nested_by_ratio(data_dict)

        print(f"  Train: {len(train_dict)} symbols")
        print(f"  Val: {len(val_dict)} symbols")

    except Exception as e:
        print(f"  Data load failed: {e}")
        print("  Using dummy data...")

        import numpy as np
        import pandas as pd

        from mars_lite.data import preprocess_ohlcv

        n_samples = 20000
        base = 40000 + np.random.randn(n_samples).cumsum()
        spread_noise = np.abs(np.random.randn(n_samples)) * 10
        dummy_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=n_samples, freq="1min"
                ),
                "open": base,
                "high": base + spread_noise,
                "low": base - spread_noise,
                "close": base + np.random.randn(n_samples),
                "volume": np.random.uniform(100, 1000, n_samples),
            }
        )
        dummy_df = preprocess_ohlcv(dummy_df)

        train_dict = {
            "BTCUSDT": {tf: dummy_df.copy() for tf in ["1m", "15m", "1h", "4h", "1d"]}
        }
        val_dict = {
            "BTCUSDT": {tf: dummy_df.copy() for tf in ["1m", "15m", "1h", "4h", "1d"]}
        }

    # 環境作成関数
    def make_train_env():
        env = MarsLiteTradingEnv(
            data_dict=train_dict,
            initial_capital=10000.0,
            max_steps=10080,
            trade_fee=0.0005,
            n_lookback=100,
            timeframes=["1m", "15m", "1h", "4h", "1d"],
        )
        return DummyVecEnv([lambda: Monitor(env)])

    def make_eval_env():
        env = MarsLiteTradingEnv(
            data_dict=val_dict,
            initial_capital=10000.0,
            max_steps=10080,
            trade_fee=0.0005,
            n_lookback=100,
            timeframes=["1m", "15m", "1h", "4h", "1d"],
        )
        return env

    # ベースハイパーパラメータ
    base_hyperparams = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "clip_range": 0.2,
    }

    # Evolution Trainer 作成
    print("\n[2/3] Initializing Evolution Trainer...")
    trainer = EvolutionTrainer(
        make_train_env_fn=make_train_env,
        make_eval_env_fn=make_eval_env,
        base_hyperparams=base_hyperparams,
        population_size=args.population,
        steps_per_generation=args.steps_per_gen,
        output_dir=args.output_dir,
        grid_bins=args.grid_bins,
    )

    # 訓練実行
    print("\n[3/3] Running Evolution Training...")
    trainer.run(n_generations=args.generations)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
