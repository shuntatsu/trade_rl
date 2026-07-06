"""
設定管理モジュール

MarS Lite環境・学習・進化戦略のパラメータ管理
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MarsLiteConfig:
    """
    MarS Lite全体設定

    環境・学習・進化戦略のパラメータを一元管理。
    """

    # === 環境パラメータ ===
    initial_inventory: float = 1000.0
    max_steps: int = 1440
    side: str = "SELL"
    y_impact: float = 0.5
    lambda_risk: float = 0.001
    n_lookback: int = 10
    force_liquidate_threshold: float = 0.01

    # === データ処理パラメータ ===
    vol_window: int = 20
    spread_period: int = 2
    spread_smooth: int = 10

    # === 学習パラメータ（PPO） ===
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_000_000

    # === PBTパラメータ ===
    population_size: int = 16
    eval_interval: int = 50000
    exploit_ratio: float = 0.2
    perturb_factor: float = 1.2

    # === MAP-Elitesパラメータ ===
    aggressiveness_bins: int = 10
    aggressiveness_range: tuple = (0.0, 1.0)
    volatility_tolerance_bins: int = 10
    volatility_tolerance_range: tuple = (0.0, 3.0)

    # === 多時間軸パラメータ ===
    timeframes: tuple = ("1m", "15m", "1h", "4h", "1d")
    base_timeframe: str = "1m"
    higher_tf_lookback: int = 5
    use_multi_tf: bool = True

    # === データ分割パラメータ ===
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_sampling: bool = True

    # === パスとデバイス ===
    data_dir: Optional[str] = None
    save_dir: Optional[str] = None
    device: str = "auto"
    seed: Optional[int] = None
    symbol: str = "BTCUSDT"
    data_days: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            k: v if not isinstance(v, tuple) else list(v)
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarsLiteConfig":
        """辞書から復元"""
        # tupleフィールドを変換
        if "aggressiveness_range" in data:
            data["aggressiveness_range"] = tuple(data["aggressiveness_range"])
        if "volatility_tolerance_range" in data:
            data["volatility_tolerance_range"] = tuple(
                data["volatility_tolerance_range"]
            )
        if "timeframes" in data:
            data["timeframes"] = tuple(data["timeframes"])
        return cls(**data)

    def save(self, path: str):
        """設定を保存"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MarsLiteConfig":
        """設定を読み込み"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# デフォルト設定
default_config = MarsLiteConfig()


def create_ppo_kwargs(config: MarsLiteConfig) -> Dict[str, Any]:
    """PPO用のキーワード引数を生成"""
    return {
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
        "vf_coef": config.vf_coef,
        "max_grad_norm": config.max_grad_norm,
        "device": config.device,
        "seed": config.seed,
    }


def create_env_kwargs(config: MarsLiteConfig) -> Dict[str, Any]:
    """環境用のキーワード引数を生成"""
    return {
        "initial_inventory": config.initial_inventory,
        "max_steps": config.max_steps,
        "side": config.side,
        "y_impact": config.y_impact,
        "lambda_risk": config.lambda_risk,
        "n_lookback": config.n_lookback,
        "force_liquidate_threshold": config.force_liquidate_threshold,
        "higher_tf_lookback": config.higher_tf_lookback,
    }
