"""
シグナルサーバーモジュール

FastAPIベースのREST API。ポートフォリオRLエージェントの最新推奨ウェイトを
Trade Platformに配信する（/api/signal/latest）ことと、保存済みモデルの
一覧・管理を提供する。

学習はCLI（scripts/train_portfolio.py）で行う設計に一本化したため、
このモジュールは学習トリガー・リアルタイムメトリクス配信は持たない
（旧metrics_server.py + training_manager.pyの学習制御機能は削除。
唯一動いていたportfolio学習はCLIから同じ mars_lite.learning.trainer /
mars_lite.pipeline.phases を使って実行できる）。
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    FastAPI = None

# データディレクトリはリポジトリルート基準で解決（CWD非依存）
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    """dataディレクトリを解決（CWD直下 → リポジトリルートの順）"""
    cwd_data = Path("data")
    if cwd_data.exists():
        return cwd_data
    return _REPO_ROOT / "data"


def create_app(output_dir: str = "./output") -> "FastAPI":
    """
    FastAPIアプリケーションを作成

    Args:
        output_dir: モデル保存先ディレクトリ

    Returns:
        FastAPIアプリケーション
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn[standard]"
        )

    app = FastAPI(
        title="Trade RL Signal Server",
        description="Portfolio RL signal serving and model management API",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 本番では適切に制限
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    output_path = Path(output_dir)

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/data/available")
    async def get_available_data() -> Dict[str, Any]:
        """
        利用可能なデータセット（通貨ペア・時間足）を取得

        Returns:
            available: {symbol: [intervals]} の辞書
        """
        data_dir = resolve_data_dir()
        available = {}

        if not data_dir.exists():
            return {"available": {}}

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
        models_dir = output_path
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

            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        model_info["metadata"] = json.load(f)
                except Exception:
                    pass

            models.append(model_info)

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
        models_dir = output_path
        model_path = models_dir / f"{model_id}.zip"

        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        model_info = {
            "id": model_id,
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "modified_at": model_path.stat().st_mtime,
        }

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
        models_dir = output_path
        model_path = models_dir / f"{model_id}.zip"

        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")

        model_path.unlink()

        meta_path = model_path.with_suffix(".json")
        if meta_path.exists():
            meta_path.unlink()

        return {"deleted": True, "id": model_id}

    # ==================== Signal Endpoint (Platform連携) ====================

    @app.get("/api/signal/latest")
    def get_latest_signal(
        prev_weights: Optional[str] = None,
        portfolio_value: float = 1.0,
        peak_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        ポートフォリオモデルの最新推奨ウェイトを返す（Trade Platform連携用）

        Platform側はこのエンドポイントをポーリングするだけで
        RLエージェントのシグナルをBots画面等に統合できる。

        Args:
            prev_weights: 現在の実ポジション（symbols順のカンマ区切り）。
                省略時は無ポジション(stateless)として扱う（従来互換）。
                指定するとEMA平滑・no-tradeバンドが実ポジション基準で
                正しく機能する（train/serveのEMA初期化不一致を解消）。
            portfolio_value / peak_value: ドローダウン応答用。
                省略時はdrawdown=0（デリスキング無効）として扱う（従来互換）。

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
        import time

        from stable_baselines3 import PPO

        from mars_lite.data.sources import create_source
        from mars_lite.env.portfolio_env import PortfolioTradingEnv
        from mars_lite.features.feature_pipeline import FeaturePipeline
        from mars_lite.serving.model_store import load_metadata, model_exists
        from mars_lite.trading.pipeline import (
            DecisionPipeline,
            MarketView,
            PortfolioState,
        )

        models_dir = output_path
        model_path = models_dir / "portfolio_model.zip"
        if not model_exists(models_dir, "portfolio_model"):
            raise HTTPException(
                status_code=404,
                detail="portfolio_model not found. Train with "
                "scripts/train_portfolio.py --phase train first.",
            )

        symbols = None
        pp_cfg = None
        feature_mask = None
        run_config = {}
        meta = load_metadata(models_dir, "portfolio_model")
        if meta is not None:
            symbols = meta.symbols or None
            pp_cfg = meta.post_processor or None
            feature_mask = meta.feature_mask
            run_config = meta.run_config or {}

        data_dir = resolve_data_dir()
        available = (
            set(
                d.name for d in data_dir.iterdir() if d.is_dir() and (d / "1m").is_dir()
            )
            if data_dir.exists()
            else set()
        )

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

        from mars_lite.trading.guardrails import GuardrailConfig
        from mars_lite.trading.post_processor import (
            PortfolioPostProcessor,
            PostProcessConfig,
            make_default_processor,
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
        age_hours = (
            pd.Timestamp.now("UTC").tz_localize(None) - last_ts
        ).total_seconds() / 3600
        # 合成/過去データでの検証を妨げないため、鮮度は警告フラグとして返す（拒否はしない）
        # 閾値はguardrails.GuardrailConfigを単一の正とする
        stale = age_hours > GuardrailConfig().max_data_age_hours

        # 学習時のenv構築引数（htf_gate/obs_risk_state/decision_every等）を
        # run_configから復元する。これを省略すると、学習時と異なる観測形状・
        # HTFゲート無しの環境で推論することになりtrain/serve一致が崩れる
        # （post_processorだけを揃えても不十分）。post_processorは別途
        # meta.post_processorから復元するのでrun_configには含めない。
        env = PortfolioTradingEnv(fs, episode_bars=10, **run_config)
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

        # prev_weights省略時は無ポジション(stateless)として扱う（従来互換）
        if prev_weights:
            try:
                prev_w = np.array(
                    [float(x) for x in prev_weights.split(",")], dtype=np.float64
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="prev_weights must be comma-separated floats",
                )
            if len(prev_w) != len(symbols):
                raise HTTPException(
                    status_code=400,
                    detail=f"prev_weights has {len(prev_w)} values but model has {len(symbols)} symbols",
                )
        else:
            prev_w = np.zeros(len(symbols))

        pv = portfolio_value
        pk = peak_value if peak_value is not None else portfolio_value

        # env.step と同一のDecisionPipeline（射影→後処理→HTFゲート）を通す
        # （実装共有により train/serve 一致が構造で保証される）。
        # HTFゲートのしきい値・スケールもenvと同一のものを使う
        # （学習時 --htf-gate 付きなら env._htf_idx が設定されている）。
        pipeline = DecisionPipeline(
            post_processor=pp,
            htf_threshold=env.htf_threshold,
            htf_neutral_scale=env.htf_neutral_scale,
        )
        state = PortfolioState(weights=prev_w, portfolio_value=pv, peak_value=pk)
        market = MarketView.from_feature_set(
            fs,
            env.t,
            vol_lookback=pp.cfg.vol_lookback,
            htf_idx=env._htf_idx,
        )
        processed, pp_info = pipeline.target_weights(raw_weights, state, market)

        # ガードレール評価（データ鮮度・NaN・過大ウェイト等）
        from mars_lite.trading.guardrails import apply_guardrails, evaluate_guardrails

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

    return app


def run_server(
    host: str = "0.0.0.0", port: int = 8001, output_dir: str = "./output"
) -> None:
    """サーバーを起動"""
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn[standard]"
        )

    app = create_app(output_dir)
    uvicorn.run(app, host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trade RL Signal Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--output-dir", default="./output", help="Output directory")

    args = parser.parse_args()

    print(f"Starting Trade RL Signal Server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")

    run_server(host=args.host, port=args.port, output_dir=args.output_dir)
