"""
学習コールバックモジュール

Stable-Baselines3用のカスタムコールバック。
学習中のメトリクスを収集してWebSocketサーバーへ送信。
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from stable_baselines3.common.callbacks import BaseCallback

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    BaseCallback = object


class MetricsHistory:
    """
    学習メトリクス履歴管理

    UIへのリアルタイム送信とローカル保存の両方に使用。
    """

    def __init__(self, max_history: int = 10000):
        """
        Args:
            max_history: 保持する最大履歴数
        """
        self.max_history = max_history
        self.history: list = []
        self._subscribers: list = []

    def broadcast(self, message: Dict[str, Any]) -> None:
        """履歴に残さず即時配信"""
        for callback in self._subscribers:
            try:
                callback(message)
            except Exception:
                pass

    def add(self, metrics: Dict[str, Any]) -> None:
        """メトリクスを追加"""
        self.history.append(metrics)

        # 履歴が長すぎる場合は古いものを削除
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        # サブスクライバーに通知
        self.broadcast(metrics)

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """メトリクス更新時のコールバックを登録"""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """コールバックを解除"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def get_recent(self, n: int = 100) -> list:
        """最新N件のメトリクスを取得"""
        return self.history[-n:]

    def get_all(self) -> list:
        """全履歴を取得"""
        return self.history.copy()

    def clear(self) -> None:
        """履歴をクリア"""
        self.history = []

    def save(self, path: Path) -> None:
        """履歴をJSONで保存"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """履歴をJSONから読み込み"""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.history = json.load(f)


# グローバルメトリクス履歴インスタンス
_global_metrics_history: Optional[MetricsHistory] = None


def get_metrics_history() -> MetricsHistory:
    """グローバルメトリクス履歴を取得（シングルトン）"""
    global _global_metrics_history
    if _global_metrics_history is None:
        _global_metrics_history = MetricsHistory()
    return _global_metrics_history


class TrainingMetricsCallback(BaseCallback):
    """
    学習メトリクス収集コールバック

    SB3のPPOから以下のメトリクスを収集:
    - policy_loss: ポリシー損失
    - value_loss: 価値関数損失
    - entropy_loss: エントロピー損失
    - approx_kl: KLダイバージェンス近似値
    - clip_fraction: クリップされた割合
    - explained_variance: 説明分散
    - mean_reward: 平均報酬（エピソード終了時）
    """

    def __init__(
        self,
        metrics_history: Optional[MetricsHistory] = None,
        log_freq: int = 1,
        total_timesteps: int = 100000,
        verbose: int = 0,
    ):
        """
        Args:
            metrics_history: メトリクス履歴オブジェクト（Noneならグローバル使用）
            log_freq: ログ出力頻度（更新回数ごと）
            total_timesteps: 総学習ステップ数（進捗計算用）
            verbose: 出力レベル
        """
        super().__init__(verbose)
        self.metrics_history = metrics_history or get_metrics_history()
        self.log_freq = log_freq
        self.total_timesteps = total_timesteps

        # 内部状態
        self._n_updates = 0
        self._episode_rewards = []
        self._win_rates = []
        self._max_drawdowns = []
        self._portfolio_values = []

        # Action Distribution Cache
        self._last_action_dist = {}
        self._last_extra_metrics = {}  # Win Rate, Drawdown, APY, etc.
        self._start_time = None

    def _on_training_start(self) -> None:
        """学習開始時に呼ばれる"""
        self._start_time = time.time()

        # 開始ログ
        start_metrics = {
            "type": "training_start",
            "timestamp": time.time(),
            "total_timesteps": self.total_timesteps,
        }
        self.metrics_history.add(start_metrics)
        print(
            f"[TrainingMetricsCallback] Training started. Total timesteps: {self.total_timesteps}"
        )
        print(
            "[TrainingMetricsCallback] Note: Loss metrics will be calculated after the first rollout (every N steps)."
        )

    def _on_step(self) -> bool:
        """各ステップで呼ばれる"""
        # エピソード報酬を収集 & 可視化データを送信
        if self.locals.get("infos") is not None:
            for i, info in enumerate(self.locals["infos"]):
                # エピソード終了時の報酬収集
                if "episode" in info:
                    self._episode_rewards.append(info["episode"]["r"])

                    if "win_rate" in info:
                        self._win_rates.append(info["win_rate"])
                    if "max_drawdown" in info:
                        self._max_drawdowns.append(info["max_drawdown"])
                    if "portfolio_value" in info:
                        self._portfolio_values.append(info["portfolio_value"])

                    # Action Distribution Cache
                    if "long_pct" in info:
                        self._last_action_dist = {
                            "long_pct": info.get("long_pct", 0),
                            "short_pct": info.get("short_pct", 0),
                            "hold_pct": info.get("hold_pct", 0),
                        }

                    # Extra Metrics Cache (APY, Trade Stats)
                    self._last_extra_metrics = {
                        "apy": info.get("apy", 0.0),
                        "max_trade_profit": info.get("max_trade_profit", 0.0),
                        "max_trade_loss": info.get("max_trade_loss", 0.0),
                        "trades_per_day": info.get("trades_per_day", 0.0),
                        "n_trades": info.get("n_trades", 0),
                        "avg_trade_return": info.get("avg_trade_return", 0.0),
                    }

                # 可視化データの送信（executionがあれば）
                if "execution" in info:
                    exec_data = info["execution"]
                    # タイムスタンプを追加
                    exec_data["type"] = "trading_data"
                    exec_data["timestamp"] = time.time()
                    # 直接送信（履歴には残さない、または別枠で）
                    # ここではMetricsHistory経由で全サブスクライバー（WebSocket含む）に流す
                    # 【最適化】すべてのステップ送ると重すぎるので、売買があった時または定期的に送る
                    is_trade = exec_data.get("side") != "hold"
                    # num_timestepsは全環境の合計ステップかもしれないが、コールバック内では model.num_timesteps を参照すべきか？
                    # self.num_timesteps は BaseCallback のプロパティ
                    if is_trade or (self.num_timesteps % 100 == 0):
                        self.metrics_history.broadcast(exec_data)

        return True

    def _on_rollout_end(self) -> None:
        """ロールアウト終了時に呼ばれる（PPO更新前）"""
        self._n_updates += 1

        if self._n_updates % self.log_freq != 0:
            return

        # 現在のメトリクスを収集
        metrics = self._collect_metrics()
        self.metrics_history.add(metrics)

        # 出力
        if self.verbose >= 1:
            mean_reward = metrics.get("mean_reward")
            policy_loss = metrics.get("policy_loss")
            value_loss = metrics.get("value_loss")

            reward_str = f"{mean_reward:.2f}" if mean_reward is not None else "N/A"
            policy_str = f"{policy_loss:.4f}" if policy_loss is not None else "N/A"
            value_str = f"{value_loss:.4f}" if value_loss is not None else "N/A"

            try:
                print(
                    f"[Step {self.num_timesteps:,}] "
                    f"Reward: {reward_str} | "
                    f"Policy Loss: {policy_str} | "
                    f"Value Loss: {value_str}"
                )
            except Exception as e:
                print(f"[Step {self.num_timesteps}] Logging Error: {e}")

    def _collect_metrics(self) -> Dict[str, Any]:
        """現在のメトリクスを収集"""
        metrics = {
            "type": "training_step",
            "timestamp": time.time(),
            "step": self.num_timesteps,
            "progress": self.num_timesteps / self.total_timesteps * 100,
            "n_updates": self._n_updates,
        }

        # PPOのログからメトリクスを取得
        if hasattr(self.model, "logger") and self.model.logger is not None:
            # SB3 loggerから値を取得
            name_to_value = getattr(self.model.logger, "name_to_value", {})

            def safe_float(val):
                if val is None:
                    return None
                try:
                    return float(val)
                except:
                    return None

            # 主要メトリクス
            metrics["policy_loss"] = safe_float(
                name_to_value.get("train/policy_gradient_loss", None)
            )
            metrics["value_loss"] = safe_float(
                name_to_value.get("train/value_loss", None)
            )
            metrics["entropy_loss"] = safe_float(
                name_to_value.get("train/entropy_loss", None)
            )
            metrics["approx_kl"] = safe_float(
                name_to_value.get("train/approx_kl", None)
            )
            metrics["clip_fraction"] = safe_float(
                name_to_value.get("train/clip_fraction", None)
            )
            metrics["explained_variance"] = safe_float(
                name_to_value.get("train/explained_variance", None)
            )
            metrics["learning_rate"] = safe_float(
                name_to_value.get("train/learning_rate", None)
            )

        # エピソード報酬
        if len(self._episode_rewards) > 0:
            import numpy as np

            metrics["mean_reward"] = float(np.mean(self._episode_rewards[-100:]))
            metrics["std_reward"] = float(np.std(self._episode_rewards[-100:]))
            metrics["n_episodes"] = len(self._episode_rewards)

            if len(self._win_rates) > 0:
                metrics["mean_win_rate"] = float(np.mean(self._win_rates[-100:]))
                metrics["mean_max_drawdown"] = float(
                    np.mean(self._max_drawdowns[-100:])
                )

            if len(self._portfolio_values) > 0:
                metrics["mean_portfolio_value"] = float(
                    np.mean(self._portfolio_values[-100:])
                )

        # 経過時間
        if self._start_time:
            elapsed = time.time() - self._start_time
            metrics["elapsed_time"] = elapsed
            if self.num_timesteps > 0:
                metrics["fps"] = self.num_timesteps / elapsed

        # Action Distribution
        if self._last_action_dist:
            metrics.update(self._last_action_dist)

        if self._last_extra_metrics:
            metrics.update(self._last_extra_metrics)

        return metrics

    def _on_training_end(self) -> None:
        """学習終了時に呼ばれる"""
        end_metrics = {
            "type": "training_end",
            "timestamp": time.time(),
            "total_steps": self.num_timesteps,
            "total_updates": self._n_updates,
            "total_episodes": len(self._episode_rewards),
        }

        if self._start_time:
            end_metrics["total_time"] = time.time() - self._start_time

        if len(self._episode_rewards) > 0:
            import numpy as np

            end_metrics["final_mean_reward"] = float(
                np.mean(self._episode_rewards[-100:])
            )

        self.metrics_history.add(end_metrics)


class CheckpointCallback(BaseCallback):
    """
    定期チェックポイント保存コールバック

    指定間隔でモデルを自動保存。
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "./checkpoints",
        name_prefix: str = "model",
        save_replay_buffer: bool = False,
        verbose: int = 0,
    ):
        """
        Args:
            save_freq: 保存頻度（ステップ数）
            save_path: 保存ディレクトリ
            name_prefix: ファイル名プレフィックス
            save_replay_buffer: リプレイバッファも保存するか
            verbose: 出力レベル
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer

    def _init_callback(self) -> None:
        """コールバック初期化"""
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """各ステップで呼ばれる"""
        if self.num_timesteps % self.save_freq == 0:
            self._save_checkpoint()
        return True

    def _save_checkpoint(self) -> None:
        """チェックポイントを保存"""
        path = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
        self.model.save(str(path))

        if self.verbose >= 1:
            print(f"Checkpoint saved: {path}")

        # メタデータも保存
        meta_path = (
            self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps_meta.json"
        )
        metadata = {
            "timesteps": self.num_timesteps,
            "timestamp": time.time(),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
