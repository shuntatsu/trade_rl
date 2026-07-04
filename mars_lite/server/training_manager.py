"""
学習マネージャーモジュール

学習プロセスのライフサイクル管理を提供。
バックグラウンドスレッドで学習を実行し、開始/停止を制御。
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Callable
from pathlib import Path

import numpy as np
from mars_lite.learning.training_callback import get_metrics_history

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
    学習設定
    
    フロントエンドから送信される学習パラメータ。
    """
    # 基本設定
    total_timesteps: int = 100000
    learning_rate: float = 5e-5
    min_learning_rate: float = 1.5e-5
    
    # PPO設定
    n_steps: int = 4096
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    
    # 環境設定
    symbol: str = "BTCUSDT"
    interval: str = "1h"

    initial_inventory: float = 100.0  # Quantity or USDT value (depends on use_usdt)
    use_usdt: bool = False  # If True, initial_inventory is treated as USDT value
    max_steps: int = 100
    
    # Physics params
    trade_fee: float = 0.0
    y_impact: float = 0.5
    lambda_risk: float = 0.001

    # Reward Params
    use_dsr: bool = False
    reward_scale: float = 0.1
    dsr_warmup_steps: int = 0 # 0 means disabled (or immediate if use_dsr=True). -1 disabled? Let's say > 0 enables auto switch.
    # Logic: if dsr_warmup_steps > 0: start with use_dsr=False, then switch to True after steps.
    
    # モデル設定
    policy_layers: list = field(default_factory=lambda: [128, 128])
    value_layers: list = field(default_factory=lambda: [128, 128])
    
    # 学習モード: "trading"（単一銘柄トレード） / "portfolio"（多銘柄配分）
    mode: str = "trading"
    # portfolioモード用データソース: "csv" / "postgres" / "synthetic"
    data_source: str = "csv"

    # Evolution Training 設定
    use_evolution: bool = False          # Evolution Training を使用
    population_size: int = 25            # 集団サイズ
    n_generations: int = 10              # 世代数
    steps_per_generation: int = 10000    # 世代あたりの学習ステップ
    grid_bins: int = 5                   # Grid Archive の分割数 (5x5)
    eval_episodes: int = 3               # 評価エピソード数
    
    # 出力設定
    output_dir: str = "./output"
    checkpoint_freq: int = 10000
    
    # Load Model (Resume Training)
    load_model_path: Optional[str] = None
    
    # Hybrid Processing
    num_envs: int = 1
    device: str = "auto" # "cpu", "cuda", "auto"

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
            "mode": self.mode,
            "data_source": self.data_source,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "symbol": self.symbol,
            "interval": self.interval,

            "initial_inventory": self.initial_inventory,
            "use_usdt": self.use_usdt,
            "max_steps": self.max_steps,
            "policy_layers": self.policy_layers,
            "value_layers": self.value_layers,
            "output_dir": self.output_dir,
            "checkpoint_freq": self.checkpoint_freq,
            "trade_fee": self.trade_fee,
            "y_impact": self.y_impact,
            "lambda_risk": self.lambda_risk,
            "use_evolution": self.use_evolution,
            "population_size": self.population_size,
            "n_generations": self.n_generations,
            "steps_per_generation": self.steps_per_generation,
            "steps_per_generation": self.steps_per_generation,
            "grid_bins": self.grid_bins,
            "eval_episodes": self.eval_episodes,
            "dsr_warmup_steps": self.dsr_warmup_steps,
            "load_model_path": self.load_model_path,
            "num_envs": self.num_envs,
            "device": self.device,
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
        学習を実行（内部メソッド）
        
        別スレッドで実行される。
        """
        try:
            self._start_time = time.time()
            self._status = TrainingStatus.RUNNING
            print("[TrainingManager] Training started...")
            print(f"[TrainingManager] Config: {self._config.to_dict()}")
            
            config = self._config
            output_path = Path(config.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Evolution Mode チェック
            if config.use_evolution:
                print("[TrainingManager] Evolution Training Mode Enabled")
                self._run_evolution_training(config, output_path)
                return

            # Portfolio Mode チェック
            if config.mode == "portfolio":
                print("[TrainingManager] Portfolio Training Mode Enabled")
                self._run_portfolio_training(config, output_path)
                return

            # === 通常の PPO Training ===
            # 必要なモジュールをインポート
            from mars_lite.env.trading_env import MarsLiteTradingEnv
            from mars_lite.learning.agent import create_ppo_agent
            from mars_lite.learning.training_callback import (
                TrainingMetricsCallback,
                CheckpointCallback,
                get_metrics_history
            )
            from mars_lite.learning.curriculum_callback import CurriculumCallback # Import Curriculum Callback
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.monitor import Monitor
            
            # データ読み込み（MultiSymbolLoader使用）
            from mars_lite.data.multi_timeframe_loader import MultiSymbolLoader
            
            data_dir = resolve_data_dir()

            # 利用可能なシンボルをスキャン
            symbols_to_load = []
            if data_dir.exists():
                symbols_to_load = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name.endswith("USDT")]
            
            if not symbols_to_load:
                symbols_to_load = [config.symbol] # フォールバック
            
            # 使用する時間軸
            timeframes = ["1m", "15m", "1h", "4h", "1d"]
            print(f"[TrainingManager] Loading data for {len(symbols_to_load)} symbols: {symbols_to_load}")
            print(f"[TrainingManager] Timeframes: {timeframes}")

            try:
                # ログコールバック
                history = get_metrics_history()
                def log_callback(msg: str):
                    print(f"[TrainingManager] {msg}")
                    history.add({
                        "type": "log",
                        "message": msg,
                        "timestamp": time.time()
                    })

                ms_loader = MultiSymbolLoader(
                    data_dir=data_dir,
                    symbols=symbols_to_load,
                    timeframes=timeframes,
                    days=3650, # 全期間
                    preprocess=True,
                )
                data_dict = ms_loader.load_all(limit_days=None, callback=log_callback)
                
                # データ存在確認
                if not data_dict:
                    raise FileNotFoundError("No data loaded successfully.")
                
                # ===== Train/Val/Test 分割 =====
                from mars_lite.data.data_utils import split_nested_by_ratio, summarize_data_split

                print("[TrainingManager] Splitting data by ratio (70/15/15)...")
                train_dict, val_dict, test_dict = split_nested_by_ratio(data_dict)
                summarize_data_split(train_dict, val_dict, test_dict)
                
                # Train データを使用
                data_dict = train_dict
                
                # データ存在確認（分割後）
                if not data_dict:
                    raise FileNotFoundError("No train data available after split.")
                
                # 統計表示
                first_sym = list(data_dict.keys())[0]
                if "1m" in data_dict[first_sym]:
                     print(f"[TrainingManager] Train data ready. Example ({first_sym}) 1m rows: {len(data_dict[first_sym]['1m'])}")
                
            except (FileNotFoundError, ValueError) as e:
                print(f"[TrainingManager] Data load failed: {e}")
                print("[TrainingManager] Using dummy data for training...")
                
                # ダミーデータ生成（辞書形式）
                import pandas as pd
                from mars_lite.data.preprocessing import preprocess_ohlcv
                n_samples = 20000 # 7日分(10080)以上確保
                base = 40000 + np.random.randn(n_samples).cumsum()
                spread_noise = np.abs(np.random.randn(n_samples)) * 10
                dummy_df = pd.DataFrame({
                    "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1min"),
                    "open": base,
                    "high": base + spread_noise,
                    "low": base - spread_noise,
                    "close": base + np.random.randn(n_samples),
                    "volume": np.random.uniform(100, 1000, n_samples),
                })
                dummy_df = preprocess_ohlcv(dummy_df)

                # 全TF、全Symbolに同じダミーをセット
                data_dict = {}
                val_dict = {} # Dummy Val
                for s in symbols_to_load:
                    data_dict[s] = {tf: dummy_df.copy() for tf in timeframes}
                    val_dict[s] = {tf: dummy_df.copy() for tf in timeframes}

            # 環境作成（Train）
            def make_env():
                env = MarsLiteTradingEnv(
                    data_dict=data_dict,
                    initial_capital=config.initial_inventory, # Use inventory field as capital
                    max_steps=config.max_steps,
                    trade_fee=config.trade_fee,
                    n_lookback=100, # Alpha Trading standard
                    timeframes=timeframes,
                    # Pass Legacy/Other params if needed or leave defaults

                    y_impact=config.y_impact, 
                    lambda_risk=config.lambda_risk,

                    # Curriculum Learning: Start with False if warmup is enabled
                    use_dsr=False if config.dsr_warmup_steps > 0 else config.use_dsr,
                    reward_scale=config.reward_scale,
                )
                return Monitor(env)

            # ベクトル環境 (Parallel / Hybrid)
            from stable_baselines3.common.vec_env import SubprocVecEnv
            
            n_envs = config.num_envs
            if n_envs > 1:
                print(f"[TrainingManager] Using SubprocVecEnv with {n_envs} workers (CPU Parallel)")
                # make_env must be a lambda that creates a new env
                env_fns = [make_env for _ in range(n_envs)]
                env = SubprocVecEnv(env_fns) 
                
                # Note: On Windows, SubprocVecEnv uses 'spawn'. 
                # Ensure make_env and dependencies are pickleable. 
                # We defined make_env inside _run_training which captures 'data_dict' (huge).
                # This might be slow to pickle or fail.
                # Ideally, we should rely on 'fork' (Linux) or avoid capturing huge data.
                # Efficient way: Load data inside the worker? No, that repeats IO.
                # SharedMemory? Complex.
                # For now, let's trust SB3/Python pickling or default to simpler if not explicitly requested?
                # User asked for it. We try. If it crashes, we revert/warn.
            else:
                env = DummyVecEnv([make_env])
            
            # エージェント作成（Mamba Proxy Extractor使用）
            from mars_lite.models.feature_extractor import MambaProxyExtractor
            
            policy_kwargs = {
                "features_extractor_class": MambaProxyExtractor,
                "features_extractor_kwargs": {
                    "n_lookback": 100,
                    "features_dim": 128,
                    "d_model": 64,
                    "n_layers": 2
                },
                "net_arch": dict(pi=[64, 64], vf=[64, 64])
            }
            
            print(f"[TrainingManager] Using MambaProxyExtractor (GRU-based) with n_lookback=100")
            
            # LR Schedule (Rule B)
            def lr_schedule(progress_remaining: float) -> float:
                return max(config.min_learning_rate, config.learning_rate * progress_remaining)

            # Load existing model if requested
            if config.load_model_path:
                load_path = Path(config.load_model_path)
                if load_path.exists():
                    print(f"[TrainingManager] Loading existing model from {load_path}")
                    self._agent = PPO.load(
                        load_path, 
                        env=env,
                        device=config.device,
                    )
                    
                    # Update parameters
                    self._agent.learning_rate = lr_schedule
                    
                    print(f"[TrainingManager] Model loaded successfully.")
                else:
                    print(f"[TrainingManager] Warning: Model path {load_path} does not exist. Starting from scratch.")
                    # Fallback to new model
                    self._agent = PPO(
                        "MlpPolicy",
                        env, 
                        verbose=1,
                        policy_kwargs=policy_kwargs,
                        learning_rate=lr_schedule,
                        n_steps=config.n_steps, 
                        batch_size=config.batch_size,
                        n_epochs=config.n_epochs,
                        gamma=config.gamma,
                        gae_lambda=config.gae_lambda,
                        clip_range=config.clip_range,
                        ent_coef=config.ent_coef,
                        vf_coef=config.vf_coef,
                        device=config.device,
                    )
            else:
                 self._agent = PPO(
                    "MlpPolicy",
                    env, 
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    learning_rate=lr_schedule,
                    n_steps=config.n_steps, 
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    clip_range=config.clip_range,
                    ent_coef=config.ent_coef,
                    vf_coef=config.vf_coef,
                    device=config.device,
                 )
            
            # Validation 環境作成
            def make_val_env():
                val_env = MarsLiteTradingEnv(
                    data_dict=val_dict,
                    initial_capital=config.initial_inventory,
                    max_steps=config.max_steps,
                    trade_fee=config.trade_fee,
                    n_lookback=100,
                    timeframes=timeframes,
 
                    y_impact=config.y_impact, 
                    lambda_risk=config.lambda_risk,
                    use_dsr=config.use_dsr,
                    reward_scale=config.reward_scale,
                )
                return Monitor(val_env)
            
            val_env = DummyVecEnv([make_val_env])
            
            # コールバック設定
            from stable_baselines3.common.callbacks import EvalCallback
            
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path=str(output_path / "models" / "best"),
                log_path=str(output_path / "eval"),
                eval_freq=5000,
                n_eval_episodes=3,
                deterministic=True,
                render=False,
                verbose=1
            )

            # Callbacks
            callbacks = [
                eval_callback,
                TrainingMetricsCallback(
                    total_timesteps=config.total_timesteps,
                    log_freq=1,
                    verbose=1,
                ),
                CheckpointCallback(
                    save_freq=config.checkpoint_freq,
                    save_path=Path(config.output_dir) / "checkpoints",
                    name_prefix="mars_lite_model"
                )
            ]
            
            # Curriculum Callback
            if config.dsr_warmup_steps > 0:
                print(f"[TrainingManager] Curriculum Learning: DSR Warmup for {config.dsr_warmup_steps} steps")
                callbacks.append(CurriculumCallback(
                    dsr_warmup_steps=config.dsr_warmup_steps,
                    metrics_history=get_metrics_history(),
                    verbose=1
                ))
            
            print(f"[TrainingManager] Starting PPO training for {config.total_timesteps} steps...")
            
            # Stop Callback is now global
            # start training log
            print(f"[TrainingManager] Starting PPO training for {config.total_timesteps} steps...")
            
            # Combine callbacks
            from stable_baselines3.common.callbacks import CallbackList
            all_callbacks = CallbackList([StopCallback(self._stop_event)] + callbacks)

            self._agent.learn(
                total_timesteps=config.total_timesteps,
                callback=all_callbacks,
                progress_bar=False  # 進捗はTrainingMetricsCallback経由でUIへ配信
            )
            
            # 最終モデル保存
            final_path = output_path / "models" / "final_model"
            self._agent.save(str(final_path))
            
            if self._stop_event.is_set():
                self._status = TrainingStatus.IDLE
                print("[TrainingManager] Training stopped by user.")
            else:
                self._status = TrainingStatus.COMPLETED
                print("[TrainingManager] Training completed.")
            
        except Exception as e:
            self._status = TrainingStatus.ERROR
            self._error_message = str(e)
            import traceback
            traceback.print_exc()
            with open("crash_log.txt", "w") as f:
                f.write(str(e) + "\n")
                traceback.print_exc(file=f)
        
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
            import sys
            sys.path.insert(0, str(_REPO_ROOT / "scripts"))
            from train_portfolio import train_ppo

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
            models_dir = output_path / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(models_dir / "portfolio_model"))
            import json as _json
            with open(models_dir / "portfolio_model.json", "w", encoding="utf-8") as f:
                _json.dump({
                    "mode": "portfolio",
                    "symbols": symbols,
                    "post_processor": pp.cfg.to_dict(),
                    "signal_gate": ic.to_dict(),
                    "oos_agent": agent_res,
                    "oos_baselines": baselines,
                    "timestamp": time.time(),
                }, f, indent=2, ensure_ascii=False)

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

    def _run_evolution_training(self, config: TrainingConfig, output_path: Path) -> None:
        """
        Evolution Training (PBT-MAP-Elites) を実行
        """
        try:
            from mars_lite.evolution import EvolutionTrainer
            from mars_lite.env.trading_env import MarsLiteTradingEnv
            from mars_lite.data.multi_timeframe_loader import MultiSymbolLoader
            from mars_lite.data.data_utils import split_nested_by_ratio
            from stable_baselines3.common.vec_env import DummyVecEnv
            from stable_baselines3.common.monitor import Monitor
            
            print(f"[EvolutionTrainer] Generations: {config.n_generations}")
            print(f"[EvolutionTrainer] Population: {config.population_size}")
            print(f"[EvolutionTrainer] Steps/Gen: {config.steps_per_generation}")
            print(f"[EvolutionTrainer] Grid: {config.grid_bins}x{config.grid_bins}")
            
            # データロード
            data_dir = Path("data")
            symbols_to_load = []
            if data_dir.exists():
                symbols_to_load = [d.name for d in data_dir.iterdir() if d.is_dir() and d.name.endswith("USDT")]
            
            if not symbols_to_load:
                symbols_to_load = [config.symbol]
            
            symbols_to_load = symbols_to_load[:50]  # 上位50
            timeframes = ["1m", "15m", "1h", "4h", "1d"]
            
            print(f"[EvolutionTrainer] Loading data for {len(symbols_to_load)} symbols...")
            
            try:
                # ログコールバック
                from mars_lite.learning.training_callback import get_metrics_history
                history = get_metrics_history()
                def log_callback(msg: str):
                    print(f"[EvolutionTrainer] {msg}")
                    history.add({
                        "type": "log",
                        "message": msg,
                        "timestamp": time.time()
                    })

                ms_loader = MultiSymbolLoader(
                    data_dir=data_dir,
                    symbols=symbols_to_load,
                    timeframes=timeframes,
                    days=3650,
                    preprocess=True
                )
                data_dict = ms_loader.load_all(limit_days=None, callback=log_callback)
                
                # Train/Val 分割（比率ベース）
                train_dict, val_dict, _ = split_nested_by_ratio(data_dict)
                
                print(f"[EvolutionTrainer] Train: {len(train_dict)} symbols")
                print(f"[EvolutionTrainer] Val: {len(val_dict)} symbols")
                
            except Exception as e:
                print(f"[EvolutionTrainer] Data load failed: {e}")
                print("[EvolutionTrainer] Using dummy data...")
                
                import pandas as pd
                n_samples = 20000
                dummy_df = pd.DataFrame({
                    "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="1min"),
                    "open": 40000 + np.random.randn(n_samples).cumsum(),
                    "high": 40500 + np.random.randn(n_samples).cumsum(),
                    "low": 39500 + np.random.randn(n_samples).cumsum(),
                    "close": 40000 + np.random.randn(n_samples).cumsum(),
                    "volume": np.random.uniform(100, 1000, n_samples),
                    "log_return": np.random.randn(n_samples) * 0.001,
                    "vol_gk": np.full(n_samples, 0.002),
                    "spread_cs": np.full(n_samples, 0.0001),
                    "v_expected": np.full(n_samples, 500.0)
                })
                train_dict = {"BTCUSDT": {tf: dummy_df.copy() for tf in timeframes}}
                val_dict = {"BTCUSDT": {tf: dummy_df.copy() for tf in timeframes}}
            
            # 環境作成関数
            def make_train_env():
                env = MarsLiteTradingEnv(
                    data_dict=train_dict,
                    initial_capital=config.initial_inventory,
                    max_steps=10080,
                    trade_fee=config.trade_fee,
                    n_lookback=100,
                    timeframes=timeframes
                )
                return DummyVecEnv([lambda: Monitor(env)])
            
            def make_eval_env():
                env = MarsLiteTradingEnv(
                    data_dict=val_dict,
                    initial_capital=config.initial_inventory,
                    max_steps=10080,
                    trade_fee=config.trade_fee,
                    n_lookback=100,
                    timeframes=timeframes
                )
                return env
            
            # ベースハイパーパラメータ
            base_hyperparams = {
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "ent_coef": config.ent_coef,
                "clip_range": config.clip_range
            }
            
            # EvolutionTrainer 作成
            trainer = EvolutionTrainer(
                make_train_env_fn=make_train_env,
                make_eval_env_fn=make_eval_env,
                base_hyperparams=base_hyperparams,
                population_size=config.population_size,
                steps_per_generation=config.steps_per_generation,
                eval_episodes=config.eval_episodes,
                output_dir=str(output_path),
                grid_bins=config.grid_bins,
                device=config.device
            )
            
            # コールバック設定
            from mars_lite.learning.training_callback import TrainingMetricsCallback, get_metrics_history
            from stable_baselines3.common.callbacks import BaseCallback
            
            # メトリクス記録
            metrics_callback = TrainingMetricsCallback(
                metrics_history=get_metrics_history(),
                log_freq=1,
                total_timesteps=config.n_generations * config.population_size * config.steps_per_generation,
                verbose=1,
            )
            
            # Reuse the same StopCallback for Evolution Mode
            stop_callback = StopCallback(self._stop_event)
            callbacks = [metrics_callback, stop_callback]
            
            # 実行
            print("[EvolutionTrainer] Starting training...")
            trainer.run(n_generations=config.n_generations, callbacks=callbacks, abort_event=self._stop_event)
            
            print("[EvolutionTrainer] Finalizing...")
            
            if self._stop_event.is_set():
                self._status = TrainingStatus.IDLE
                print("[TrainingManager] Evolution training stopped by user.")
            else:
                self._status = TrainingStatus.COMPLETED
                print("[TrainingManager] Evolution training completed.")
            
        except TrainingStoppedError:
            print("[EvolutionTrainer] Training stopped by user (TrainingStoppedError caught).")
            self._status = TrainingStatus.IDLE

        except Exception as e:
            # Check if it was a manual stop (is_set)
            if self._stop_event.is_set():
                print("[EvolutionTrainer] Training stopped by user (caught in run_evolution).")
                self._status = TrainingStatus.IDLE
            else:
                print(f"[EvolutionTrainer] Error: {e}")
                self._status = TrainingStatus.IDLE # Or ERROR if fatal
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
