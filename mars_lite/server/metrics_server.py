"""
メトリクスサーバーモジュール

FastAPIベースのWebSocket + REST APIサーバー。
学習メトリクスのリアルタイム配信とモデル管理を提供。
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import time
import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None
    WebSocket = None

from mars_lite.learning.training_callback import MetricsHistory, get_metrics_history


class NumpyJSONEncoder(json.JSONEncoder):
    """NumPy型をJSONシリアライズ可能にするエンコーダ"""
    def default(self, obj):
        # NumPy 2.0でnp.float_/np.int_は削除されたため抽象型で判定
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ConnectionManager:
    """
    WebSocket接続管理
    
    複数クライアントへの同時配信をサポート。
    """
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket) -> None:
        """新規接続を受け入れ"""
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket) -> None:
        """接続を切断"""
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """全クライアントにメッセージを送信"""
        if not self.active_connections:
            return
        
        data = json.dumps(message, ensure_ascii=False, cls=NumpyJSONEncoder)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(data)
            except Exception:
                disconnected.add(connection)
        
        # 切断された接続を削除
        self.active_connections -= disconnected
    
    @property
    def connection_count(self) -> int:
        """接続数を取得"""
        return len(self.active_connections)


def create_app(
    metrics_history: Optional[MetricsHistory] = None,
    output_dir: str = "./output",
) -> "FastAPI":
    """
    FastAPIアプリケーションを作成
    
    Args:
        metrics_history: メトリクス履歴（Noneならグローバル使用）
        output_dir: 出力ディレクトリ（モデル保存先等）
        
    Returns:
        FastAPIアプリケーション
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required. "
            "Install with: pip install fastapi uvicorn[standard]"
        )
    
    app = FastAPI(
        title="MarS Lite Training Server",
        description="Real-time training monitoring and model management API",
        version="0.1.0",
    )
    
    # CORS設定（React開発サーバーからのアクセスを許可）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本番では適切に制限
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 状態管理
    manager = ConnectionManager()
    history = metrics_history or get_metrics_history()
    output_path = Path(output_dir)
    
    # メトリクス更新時にWebSocketで配信
    # イベントループ参照を保持
    server_loop: Optional[asyncio.AbstractEventLoop] = None
    
    # メトリクス更新時にWebSocketで配信
    async def on_metrics_update(metrics: Dict[str, Any]) -> None:
        """メトリクス更新時のコールバック"""
        await manager.broadcast(metrics)
    
    # 非同期コールバックを登録するためのラッパー
    def sync_broadcast_wrapper(metrics: Dict[str, Any]) -> None:
        """同期コールバック → 非同期ブロードキャストへの橋渡し (スレッドセーフ)"""
        if server_loop and server_loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(metrics), server_loop)
    
    # 起動時の初期化
    @app.on_event("startup")
    async def startup_event():
        nonlocal server_loop
        server_loop = asyncio.get_running_loop()
        history.subscribe(sync_broadcast_wrapper)
    
    # ==================== WebSocket Endpoints ====================
    
    @app.websocket("/ws/metrics")
    async def websocket_metrics(websocket: WebSocket):
        """
        メトリクスWebSocketエンドポイント
        
        接続すると、以下のメッセージを受信:
        - training_start: 学習開始通知
        - training_step: 各ステップのメトリクス
        - training_end: 学習終了通知
        """
        await manager.connect(websocket)
        
        try:
            # 接続時に現在の履歴を送信
            recent = history.get_recent(100)
            if recent:
                await websocket.send_text(json.dumps({
                    "type": "history",
                    "data": recent,
                }, cls=NumpyJSONEncoder))
            
            # 接続維持（クライアントからのメッセージを待機）
            while True:
                data = await websocket.receive_text()
                
                # pingに応答
                if data == "ping":
                    await websocket.send_text("pong")
                
                # 履歴リクエスト
                elif data == "get_history":
                    await websocket.send_text(json.dumps({
                        "type": "history",
                        "data": history.get_all(),
                    }))
                    
        except (WebSocketDisconnect, ConnectionResetError):
            manager.disconnect(websocket)
        except Exception as e:
            # print(f"WebSocket Error: {e}") # Optional: ignore noisy errors
            manager.disconnect(websocket)
    
    # ==================== REST API Endpoints ====================
    
    # ==================== Metrics & Data Endpoints ====================
    
    @app.get("/api/metrics")
    async def get_metrics(limit: int = 100) -> Dict[str, Any]:
        """
        最新のメトリクスを取得
        
        Args:
            limit: 取得件数（デフォルト100）
            
        Returns:
            metrics: メトリクスリスト
            count: 総メトリクス数
        """
        recent = history.get_recent(limit)
        return {
            "metrics": recent,
            "count": len(history.history),
            "limit": limit,
        }
    
    @app.get("/api/metrics/latest")
    async def get_latest_metrics() -> Dict[str, Any]:
        """
        最新1件のメトリクスを取得
        
        Returns:
            最新メトリクス（なければ空オブジェクト）
        """
        recent = history.get_recent(1)
        return recent[0] if recent else {}

    @app.get("/api/data/available")
    async def get_available_data() -> Dict[str, Any]:
        """
        利用可能なデータセット（通貨ペア・時間足）を取得
        
        Returns:
            available: {symbol: [intervals]} の辞書
        """
        from mars_lite.server.training_manager import resolve_data_dir
        data_dir = resolve_data_dir()
        available = {}
        
        if not data_dir.exists():
            return {"available": {}}
        
        # data/{symbol}/{interval} の構造を走査
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                intervals = []
                for interval_dir in symbol_dir.iterdir():
                    if interval_dir.is_dir():
                        intervals.append(interval_dir.name)
                
                if intervals:
                    available[symbol_dir.name] = sorted(intervals)
        
        return {"available": available}
    
    # ==================== Model Management Endpoints ====================
    
    @app.get("/api/models")
    async def list_models() -> Dict[str, Any]:
        """
        保存済みモデル一覧を取得
        
        Returns:
            models: モデル情報リスト
        """
        models_dir = output_path / "models"
        if not models_dir.exists():
            return {"models": []}
        
        models = []
        for model_path in models_dir.glob("*.zip"):
            meta_path = model_path.with_suffix(".json")
            
            model_info = {
                "id": model_path.stem,
                "path": str(model_path),
                "size_bytes": model_path.stat().st_size,
                "modified_at": model_path.stat().st_mtime,
            }
            
            # メタデータがあれば読み込み
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        model_info["metadata"] = json.load(f)
                except Exception:
                    pass
            
            models.append(model_info)
            
        # Also scan checkpoints
        checkpoints_dir = output_path / "checkpoints"
        if checkpoints_dir.exists():
            for model_path in checkpoints_dir.glob("*.zip"):
                meta_path = model_path.with_suffix(".json")
                
                model_info = {
                    "id": f"checkpoint/{model_path.stem}", # Distinguish checkpoints
                    "name": model_path.stem,
                    "path": str(model_path),
                    "size_bytes": model_path.stat().st_size,
                    "modified_at": model_path.stat().st_mtime,
                    "is_checkpoint": True
                }
                 # メタデータがあれば読み込み
                if meta_path.exists():
                    try:
                        with open(meta_path, "r") as f:
                            model_info["metadata"] = json.load(f)
                    except Exception:
                        pass
                models.append(model_info)
        
        # 更新日時でソート（新しい順）
        models.sort(key=lambda x: x["modified_at"], reverse=True)
        
        return {"models": models}
    
    @app.get("/api/models/{model_id}")
    async def get_model(model_id: str) -> Dict[str, Any]:
        """
        モデル詳細を取得
        
        Args:
            model_id: モデルID
            
        Returns:
            モデル情報
        """
        models_dir = output_path / "models"
        model_path = models_dir / f"{model_id}.zip"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = {
            "id": model_id,
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "modified_at": model_path.stat().st_mtime,
        }
        
        # メタデータ
        meta_path = model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                model_info["metadata"] = json.load(f)
        
        return model_info
    
    @app.delete("/api/models/{model_id}")
    async def delete_model(model_id: str) -> Dict[str, Any]:
        """
        モデルを削除
        
        Args:
            model_id: モデルID
            
        Returns:
            deleted: 削除されたか
        """
        models_dir = output_path / "models"
        model_path = models_dir / f"{model_id}.zip"
        
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_path.unlink()
        
        # メタデータも削除
        meta_path = model_path.with_suffix(".json")
        if meta_path.exists():
            meta_path.unlink()
        
        return {"deleted": True, "id": model_id}

    @app.post("/api/training/save")
    async def save_checkpoint(name: Optional[str] = None) -> Dict[str, Any]:
        """
        手動でチェックポイントを保存
        
        Args:
            name: 保存名（省略時はタイムスタンプ）
            
        Returns:
            保存結果
        """
        from mars_lite.server.training_manager import get_training_manager
        
        manager = get_training_manager()
        result = manager.save_checkpoint(name)
        return result
    
    # ==================== Training Control Endpoints ====================
    
    @app.post("/api/training/start")
    async def start_training(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        学習を開始
        
        Args:
            config: 学習設定（オプション）
            
        Returns:
            開始結果
        """
        from mars_lite.server.training_manager import (
            get_training_manager, 
            TrainingConfig
        )
        
        manager = get_training_manager()
        
        training_config = None
        if config:
            training_config = TrainingConfig.from_dict(config)
        
        result = manager.start(training_config)
        return result
    
    @app.post("/api/training/stop")
    async def stop_training() -> Dict[str, Any]:
        """
        学習を停止
        
        Returns:
            停止結果
        """
        from mars_lite.server.training_manager import get_training_manager
        
        manager = get_training_manager()
        result = manager.stop()
        return result
    
    @app.get("/api/training/status")
    async def get_training_status() -> Dict[str, Any]:
        """
        学習状態を取得
        
        Returns:
            現在の学習状態
        """
        from mars_lite.server.training_manager import get_training_manager
        
        manager = get_training_manager()
        return manager.get_status_info()
    
    @app.get("/api/training/config")
    async def get_training_config() -> Dict[str, Any]:
        """
        デフォルト学習設定を取得
        
        Returns:
            学習設定
        """
        from mars_lite.server.training_manager import TrainingConfig
        
        return TrainingConfig().to_dict()
    
    # ==================== Signal Endpoint (Platform連携) ====================

    @app.get("/api/signal/latest")
    def get_latest_signal() -> Dict[str, Any]:
        """
        ポートフォリオモデルの最新推奨ウェイトを返す（Trade Platform連携用）

        Platform側はこのエンドポイントをポーリングするだけで
        RLエージェントのシグナルをBots画面等に統合できる。

        Returns:
            {
              "timestamp": ...,
              "symbols": [...],
              "weights": {symbol: weight},     # -1〜1、Σ|w|<=1
              "net_exposure": float,
              "model_id": "portfolio_model",
              "data_timestamp": 最終バー時刻
            }
        """
        from stable_baselines3 import PPO
        from mars_lite.server.training_manager import resolve_data_dir
        from mars_lite.data.sources import create_source
        from mars_lite.features.feature_pipeline import FeaturePipeline
        from mars_lite.env.portfolio_env import PortfolioTradingEnv

        model_path = output_path / "models" / "portfolio_model.zip"
        meta_path = output_path / "models" / "portfolio_model.json"
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail="portfolio_model not found. Train with mode='portfolio' first.",
            )

        symbols = None
        pp_cfg = None
        feature_mask = None
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    symbols = meta.get("symbols")
                    pp_cfg = meta.get("post_processor")
                    feature_mask = meta.get("feature_mask")
            except Exception:
                pass

        data_dir = resolve_data_dir()
        available = set(
            d.name for d in data_dir.iterdir()
            if d.is_dir() and (d / "1m").is_dir()
        ) if data_dir.exists() else set()

        if symbols is None:
            symbols = sorted(available)
        else:
            # モデルは学習時の銘柄数・順序に依存するため、全銘柄のデータが必要
            missing = [s for s in symbols if s not in available]
            if missing:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"Model was trained on {symbols} but data is missing for "
                        f"{missing}. Fetch data for all trained symbols "
                        f"(fetch_futures.py) or retrain with available data."
                    ),
                )
        if not symbols:
            raise HTTPException(status_code=404, detail="No data available")

        from mars_lite.trading.post_processor import (
            PortfolioPostProcessor, PostProcessConfig, make_default_processor,
        )

        source = create_source("csv", symbols, data_dir=data_dir)
        fs = FeaturePipeline(symbols).build(source)

        # 学習時と同一の特徴マスクを適用（train/serve一致）
        if feature_mask is not None and len(feature_mask) == fs.n_features:
            fs = fs.apply_mask(np.asarray(feature_mask, dtype=bool))

        # データ鮮度チェック（古いデータでシグナルを出さない）
        import pandas as pd
        last_ts = pd.Timestamp(fs.timestamps[-1])
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_localize(None)
        age_hours = (pd.Timestamp.utcnow().tz_localize(None) - last_ts).total_seconds() / 3600
        # 合成/過去データでの検証を妨げないため、鮮度は警告フラグとして返す（拒否はしない）
        stale = age_hours > 2.0

        env = PortfolioTradingEnv(fs, episode_bars=10)
        # 最終バーの観測でフラット状態からの推奨ウェイトを推論
        obs, _ = env.reset(options={"start_idx": max(fs.n_bars - 3, 0)})
        agent = PPO.load(str(model_path), device="cpu")
        action, _ = agent.predict(obs, deterministic=True)
        raw_weights = PortfolioTradingEnv.project_weights(
            np.asarray(action, dtype=np.float64).flatten()
        )

        # 学習時と同一の後処理を適用（train/serve一致）
        if pp_cfg:
            pp = PortfolioPostProcessor(PostProcessConfig(**pp_cfg))
        else:
            pp = make_default_processor()
        recent = env._recent_returns()
        processed, pp_info = pp.process(
            raw_weights=raw_weights,
            prev_weights=np.zeros(len(symbols)),  # 現ポジションはPlatform側が持つ
            recent_returns=recent,
        )

        # ガードレール評価（データ鮮度・NaN・過大ウェイト等）
        from mars_lite.trading.guardrails import evaluate_guardrails, apply_guardrails
        guard = evaluate_guardrails(
            weights=processed,
            portfolio_value=1.0,
            turnover=float(np.abs(processed).sum()),
            data_age_hours=age_hours,
            features=fs.features[-1].flatten(),
        )
        final = apply_guardrails(processed, guard)

        return {
            "timestamp": time.time(),
            "symbols": symbols,
            "raw_weights": {s: float(w) for s, w in zip(symbols, raw_weights)},
            "weights": {s: float(w) for s, w in zip(symbols, final)},
            "processed_weights": {s: float(w) for s, w in zip(symbols, processed)},
            "net_exposure": float(final.sum()),
            "gross_exposure": float(np.abs(final).sum()),
            "model_id": "portfolio_model",
            "data_timestamp": str(fs.timestamps[-1]),
            "data_age_hours": round(float(age_hours), 2),
            "stale": bool(stale),
            "guardrail": guard.to_dict(),
            "vol_scale": round(float(pp_info.vol_scale), 3),
            "est_annual_vol": round(float(pp_info.est_port_vol), 3),
        }

    # ==================== Backtest Endpoint ====================

    @app.post("/api/backtest")
    def run_backtest_endpoint(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存済みモデルでバックテストを実行（BacktestPanel用）

        同期defで宣言しFastAPIのthreadpoolで実行する
        （イベントループ／WebSocket配信をブロックしないため）。

        Args:
            body: {model_id, symbol, n_episodes}

        Returns:
            {"metrics": {...}, "episode_rewards": [...]}
        """
        from mars_lite.server.training_manager import get_training_manager, resolve_data_dir

        manager_tm = get_training_manager()
        if manager_tm.is_running:
            raise HTTPException(
                status_code=409, detail="Training is running. Stop it before backtesting."
            )

        model_id = body.get("model_id")
        symbol = body.get("symbol", "BTCUSDT")
        n_episodes = int(body.get("n_episodes", 5))

        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")

        # モデルパス解決（models/ と checkpoints/ の両方に対応）
        if str(model_id).startswith("checkpoint/"):
            model_path = output_path / "checkpoints" / f"{str(model_id).split('/', 1)[1]}.zip"
        else:
            model_path = output_path / "models" / f"{model_id}.zip"
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        from stable_baselines3 import PPO
        from mars_lite.env.trading_env import MarsLiteTradingEnv
        from mars_lite.data.multi_timeframe_loader import MultiTimeframeLoader
        from mars_lite.data.data_utils import split_nested_by_ratio

        # データ読み込み（テストスライス = 後半15%）
        timeframes = ["1m", "15m", "1h", "4h", "1d"]
        data_dir = resolve_data_dir()
        try:
            loader = MultiTimeframeLoader(
                data_dir=data_dir,
                timeframes=timeframes,
                symbol=symbol,
                days=3650,
                preprocess=True,
            )
            tf_data = loader.load_all()
            _, _, test_dict = split_nested_by_ratio({symbol: tf_data})
            if not test_dict:
                test_dict = {symbol: tf_data}
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=f"Data not found for {symbol}: {e}")

        try:
            env = MarsLiteTradingEnv(
                data_dict=test_dict,
                timeframes=timeframes,
                n_lookback=100,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Not enough test data: {e}")

        agent = PPO.load(str(model_path), device="cpu")

        episode_rewards: List[float] = []
        final_infos: List[Dict[str, Any]] = []
        for _ in range(max(n_episodes, 1)):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            info: Dict[str, Any] = {}
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
            episode_rewards.append(total_reward)
            final_infos.append(info)

        rewards = np.array(episode_rewards)
        metrics: Dict[str, Any] = {
            "n_episodes": len(episode_rewards),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
        }
        if len(rewards) > 1 and float(np.std(rewards)) > 0:
            metrics["sharpe_ratio"] = float(np.mean(rewards) / np.std(rewards))
        if final_infos:
            metrics["max_drawdown"] = float(
                np.mean([i.get("max_drawdown", 0.0) for i in final_infos])
            )
            metrics["win_rate"] = float(
                np.mean([i.get("win_rate", 0.0) for i in final_infos])
            )
            metrics["n_trades"] = int(
                np.sum([i.get("n_trades", 0) for i in final_infos])
            )
            metrics["mean_portfolio_value"] = float(
                np.mean([i.get("portfolio_value", 0.0) for i in final_infos])
            )

        return {"metrics": metrics, "episode_rewards": episode_rewards}

    @app.post("/api/training/clear")
    async def clear_metrics() -> Dict[str, Any]:
        """
        メトリクス履歴をクリア
        
        Returns:
            cleared: クリアされたか
        """
        history.clear()
        return {"cleared": True}
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    metrics_history: Optional[MetricsHistory] = None,
    output_dir: str = "./output",
) -> None:
    """
    サーバーを起動
    
    Args:
        host: ホストアドレス
        port: ポート番号
        metrics_history: メトリクス履歴
        output_dir: 出力ディレクトリ
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required. "
            "Install with: pip install fastapi uvicorn[standard]"
        )
    
    app = create_app(metrics_history, output_dir)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


async def run_server_async(
    host: str = "0.0.0.0",
    port: int = 8001,
    metrics_history: Optional[MetricsHistory] = None,
    output_dir: str = "./output",
) -> None:
    """
    サーバーを非同期で起動（学習と並行実行用）
    
    Args:
        host: ホストアドレス
        port: ポート番号
        metrics_history: メトリクス履歴
        output_dir: 出力ディレクトリ
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required. "
            "Install with: pip install fastapi uvicorn[standard]"
        )
    
    app = create_app(metrics_history, output_dir)
    config = uvicorn.Config(
        app, 
        host=host, 
        port=port, 
        log_level="warning", 
        access_log=False
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MarS Lite Training Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Starting MarS Lite Training Server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    run_server(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
    )
