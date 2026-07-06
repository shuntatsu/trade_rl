"""
学習マネージャーモジュール

学習プロセスのライフサイクル管理を提供。
バックグラウンドスレッドで学習を実行し、開始/停止を制御。
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Callable
from pathlib import Path

import numpy as np

# データディレクトリはリポジトリルート基準で解決（CWD非依存）
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    """dataディレクトリを解決（CWD直下 → リポジトリルートの順）"""
    cwd_data = Path("data")
    if cwd_data.exists():
        return cwd_data
    return _REPO_ROOT / "data"


class TrainingStatus(Enum):
    """学習状態"""
    IDLE = "idle"           # 待機中
    STARTING = "starting"   # 開始処理中
    RUNNING = "running"     # 学習中
    STOPPING = "stopping"   # 停止処理中
    COMPLETED = "completed" # 完了
    ERROR = "error"         # エラー


class TrainingStoppedError(Exception):
    """学習が中断されたことを示す例外"""
    pass


@dataclass
class TrainingConfig:
    """
    学習設定（portfolioモード専用）

    フロントエンドから送信される学習パラメータ。
    詳細なハイパーパラメータはmars_lite.config.RunConfig（Phase 1）に順次移管予定。
    """
    total_timesteps: int = 100000
    learning_rate: float = 5e-5

    # portfolioモード用データソース: "csv" / "postgres" / "synthetic"
    data_source: str = "csv"

    # 出力設定
    output_dir: str = "./output"

    # Hybrid Processing
    num_envs: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """辞書から設定を生成"""
        # dataのキーとクラスのフィールドをマージ
        # 未知のキーがあっても無視する安全な実装
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "data_source": self.data_source,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "output_dir": self.output_dir,
            "num_envs": self.num_envs,
        }



# 共通のStopCallback
from stable_baselines3.common.callbacks import BaseCallback

class StopCallback(BaseCallback):
    """
    停止イベントを監視し、セットされたら学習を中断するコールバック
    """
    def __init__(self, stop_event):
        super().__init__()
        self.stop_event = stop_event
    
    def _on_step(self):
        # Raise exception to force immediate stop
        if self.stop_event.is_set():
            raise TrainingStoppedError("Training stopped by user")
        return True


class TrainingManager:
    """
    学習プロセスマネージャー
    
    学習の開始・停止・状態監視を管理。
    シングルトンパターンで実装。
    """
    
    _instance: Optional["TrainingManager"] = None
    
    def __new__(cls) -> "TrainingManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._status = TrainingStatus.IDLE
        self._config: Optional[TrainingConfig] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._error_message: Optional[str] = None
        self._start_time: Optional[float] = None
        self._current_step: int = 0
        self._agent = None
        
        self._initialized = True
    
    @property
    def status(self) -> TrainingStatus:
        """現在の学習状態を取得"""
        return self._status
    
    @property
    def config(self) -> Optional[TrainingConfig]:
        """現在の学習設定を取得"""
        return self._config
    
    @property
    def is_running(self) -> bool:
        """学習中かどうか"""
        return self._status in (TrainingStatus.STARTING, TrainingStatus.RUNNING)
    
    def get_status_info(self) -> Dict[str, Any]:
        """
        状態情報を取得
        
        Returns:
            状態情報の辞書
        """
        info = {
            "status": self._status.value,
            "is_running": self.is_running,
        }
        
        if self._config:
            info["config"] = self._config.to_dict()
        
        if self._start_time:
            info["elapsed_time"] = time.time() - self._start_time
            info["current_step"] = self._current_step
            if self._config:
                info["progress"] = self._current_step / self._config.total_timesteps * 100
        
        if self._error_message:
            info["error"] = self._error_message
        
        return info
    
    def start(self, config: Optional[TrainingConfig] = None) -> Dict[str, Any]:
        """
        学習を開始
        
        Args:
            config: 学習設定（Noneならデフォルト）
            
        Returns:
            開始結果
        """
        if self.is_running:
            return {
                "success": False,
                "error": "Training is already running",
                "status": self._status.value,
            }
        
        self._config = config or TrainingConfig()
        self._status = TrainingStatus.STARTING
        self._stop_event.clear()
        self._error_message = None
        self._current_step = 0
        
        # バックグラウンドスレッドで学習開始
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()
        
        return {
            "success": True,
            "message": "Training started",
            "config": self._config.to_dict(),
        }
    
    def stop(self) -> Dict[str, Any]:
        """
        学習を停止
        
        Returns:
            停止結果
        """
        if not self.is_running:
            return {
                "success": False,
                "error": "Training is not running",
                "status": self._status.value,
            }
        
        self._status = TrainingStatus.STOPPING
        self._stop_event.set()
        
        return {
            "success": True,
            "message": "Stopping training...",
        }

    def save_checkpoint(self, name: str = None) -> Dict[str, Any]:
        """
        手動でチェックポイントを保存
        """
        if not self._agent:
            return {
                "success": False,
                "error": "No active agent to save (training might not be running or initialized)."
            }
        
        try:
            timestamp = int(time.time())
            if not name:
                name = f"manual_checkpoint_{timestamp}"
            
            # ディレクトリ確保
            save_dir = Path(self._config.output_dir) / "checkpoints"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = save_dir / name
            self._agent.save(str(save_path))
            
            print(f"[TrainingManager] Manual checkpoint saved: {save_path}")
            return {
                "success": True,
                "message": f"Saved checkpoint to {save_path}",
                "path": str(save_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_training(self) -> None:
        """
        学習を実行（内部メソッド、別スレッドで実行される）

        現在サポートされるのはportfolioモードのみ。
        """
        try:
            self._start_time = time.time()
            self._status = TrainingStatus.RUNNING
            print("[TrainingManager] Training started...")
            print(f"[TrainingManager] Config: {self._config.to_dict()}")

            config = self._config
            output_path = Path(config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            print("[TrainingManager] Portfolio Training Mode")
            self._run_portfolio_training(config, output_path)

        except Exception as e:
            self._status = TrainingStatus.ERROR
            self._error_message = str(e)
            import traceback
            traceback.print_exc()

        finally:
            self._thread = None

    def _run_portfolio_training(self, config: TrainingConfig, output_path: Path) -> None:
        """
        ポートフォリオ配分RLの学習を実行（mode="portfolio"）

        データはdata_sourceに従い、csvならdata/配下の全USDT銘柄
        （orderflow/fundingがあれば自動で特徴に取り込み）。
        メトリクスは既存のTrainingMetricsCallback経由でUIへ配信される。
        """
        try:
            from stable_baselines3.common.callbacks import CallbackList
            from mars_lite.data.sources import create_source, SyntheticSource
            from mars_lite.features.feature_pipeline import FeaturePipeline
            from mars_lite.features.signal_check import run_signal_check
            from mars_lite.learning.training_callback import (
                TrainingMetricsCallback, get_metrics_history,
            )
            from mars_lite.learning.baselines import run_all_baselines
            from mars_lite.eval.walk_forward import evaluate_agent_on_slice

            history = get_metrics_history()

            def log(msg: str):
                print(f"[PortfolioTraining] {msg}")
                history.add({"type": "log", "message": msg, "timestamp": time.time()})

            # ---- データソース ----
            if config.data_source == "synthetic":
                source = SyntheticSource(n_days=90, alpha="cross")
                symbols = source.symbols
            else:
                data_dir = resolve_data_dir()
                symbols = sorted(
                    d.name for d in data_dir.iterdir()
                    if d.is_dir() and (d / "1m").is_dir()
                ) if data_dir.exists() else []
                if not symbols:
                    log("No CSV data found. Falling back to synthetic (alpha=cross).")
                    source = SyntheticSource(n_days=90, alpha="cross")
                    symbols = source.symbols
                elif config.data_source == "postgres":
                    source = create_source("postgres", symbols)
                else:
                    source = create_source("csv", symbols, data_dir=data_dir)

            log(f"Building features for {len(symbols)} symbols...")
            fs = FeaturePipeline(symbols).build(source)
            log(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols x {fs.n_features} features")

            # ---- ゲート1（記録のみ、学習は続行） ----
            ic = run_signal_check(fs)
            log(ic.summary())

            split = int(fs.n_bars * 0.8)
            train_fs = fs.slice(0, split)
            test_fs = fs.slice(min(split + 24, fs.n_bars - 10), fs.n_bars)


            # ---- 学習 ----
            from mars_lite.learning.trainer import train_ppo

            metrics_cb = TrainingMetricsCallback(
                total_timesteps=config.total_timesteps, log_freq=1, verbose=1,
            )
            callbacks = CallbackList([StopCallback(self._stop_event), metrics_cb])

            # 推奨後処理器（学習・運用で同一に適用）
            from mars_lite.trading.post_processor import make_default_processor
            pp = make_default_processor()
            ekw = {"post_processor": pp}

            log(f"Training PPO for {config.total_timesteps:,} steps...")
            agent = train_ppo(
                fs=train_fs,
                timesteps=config.total_timesteps,
                seed=0,
                n_envs=max(config.num_envs, 1),
                learning_rate=config.learning_rate,
                callbacks=callbacks,
                **ekw,
            )

            # ---- OOS評価 + ベースライン比較 ----
            agent_res = evaluate_agent_on_slice(agent, test_fs, **ekw)
            agent_res.pop("equity_curve", None)
            baselines = {k: v.to_dict() for k, v in run_all_baselines(test_fs).items()}
            log(f"OOS: agent return={agent_res['total_return']:+.2%} "
                f"sharpe={agent_res['sharpe']:.2f} | "
                f"B&H={baselines['equal_weight_bh']['total_return']:+.2%} "
                f"momentum={baselines['cross_momentum']['total_return']:+.2%}")

            # ---- 保存 ----
            from mars_lite.serving.model_store import save_bundle, ModelMetadata
            models_dir = output_path / "models"
            save_bundle(models_dir, "portfolio_model", agent, ModelMetadata(
                symbols=symbols,
                post_processor=pp.cfg.to_dict(),
                metrics={
                    "signal_gate": ic.to_dict(),
                    "oos_agent": agent_res,
                    "oos_baselines": baselines,
                },
            ))

            if self._stop_event.is_set():
                self._status = TrainingStatus.IDLE
                log("Portfolio training stopped by user.")
            else:
                self._status = TrainingStatus.COMPLETED
                log("Portfolio training completed.")

        except TrainingStoppedError:
            self._status = TrainingStatus.IDLE
            print("[PortfolioTraining] Stopped by user.")
        except Exception as e:
            if self._stop_event.is_set():
                self._status = TrainingStatus.IDLE
            else:
                self._status = TrainingStatus.ERROR
                self._error_message = str(e)
                import traceback
                traceback.print_exc()


# グローバルインスタンス
_training_manager: Optional[TrainingManager] = None


def get_training_manager() -> TrainingManager:
    """TrainingManagerのシングルトンインスタンスを取得"""
    global _training_manager
    if _training_manager is None:
        _training_manager = TrainingManager()
    return _training_manager
