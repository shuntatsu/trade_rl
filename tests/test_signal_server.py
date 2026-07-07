"""
mars_lite.server.signal_server のテスト

/api/signal/latest が実際にモデル+データを通して200を返すこと
（以前は存在しないenv._recent_returns()を呼びAttributeErrorで
必ず落ちていた）と、モデル管理・データ一覧エンドポイントを確認する。
"""

import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mars_lite.data.sources import CsvSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.learning.trainer import train_ppo
from mars_lite.serving.model_store import save_bundle, ModelMetadata
from mars_lite.trading.post_processor import make_default_processor
from mars_lite.server.signal_server import create_app

_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trained_model_dir(tmp_path_factory):
    """小さく学習したモデル+メタデータ+対応CSVデータを用意する

    scripts/generate_sample_data.py（実運用と同じ生成コード）でCSVを作り、
    CsvSourceで読み込んで学習する。
    """
    base = tmp_path_factory.mktemp("signal_server_fixture")
    models_dir = base / "output"
    data_dir = base / "data"
    models_dir.mkdir()

    subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "generate_sample_data.py"),
         "--days", "30", "--alpha", "cross", "--seed", "1", "--output", str(data_dir)],
        check=True, cwd=_REPO_ROOT, capture_output=True,
    )

    symbols = sorted(
        d.name for d in data_dir.iterdir() if d.is_dir() and (d / "1m").is_dir()
    )
    source = CsvSource(data_dir, symbols)
    fs = FeaturePipeline(symbols).build(source)
    pp = make_default_processor()
    agent = train_ppo(
        fs=fs, timesteps=1500, seed=0, n_envs=1,
        bc_warmstart=False, post_processor=pp,
    )
    save_bundle(models_dir, "portfolio_model", agent, ModelMetadata(
        symbols=symbols, post_processor=pp.cfg.to_dict(),
    ))

    return models_dir, data_dir


@pytest.fixture
def client(trained_model_dir, monkeypatch):
    models_dir, data_dir = trained_model_dir
    # resolve_data_dirはCWD直下の"data"を優先するため、CWDをdata_dirの親に切替
    monkeypatch.chdir(data_dir.parent)
    app = create_app(output_dir=str(models_dir))
    return TestClient(app)


class TestSignalServer:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_list_models(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["models"]]
        assert "portfolio_model" in ids

    def test_get_model_detail(self, client):
        resp = client.get("/api/models/portfolio_model")
        assert resp.status_code == 200
        assert "metadata" in resp.json()

    def test_get_model_404(self, client):
        resp = client.get("/api/models/does_not_exist")
        assert resp.status_code == 404

    def test_data_available(self, client, trained_model_dir):
        resp = client.get("/api/data/available")
        assert resp.status_code == 200
        _, data_dir = trained_model_dir
        available = resp.json()["available"]
        assert len(available) > 0

    def test_signal_latest_returns_200_not_crash(self, client):
        """以前 env._recent_returns() 呼び出しでAttributeErrorになっていたバグの回帰テスト"""
        resp = client.get("/api/signal/latest")
        assert resp.status_code == 200
        body = resp.json()
        assert "weights" in body
        assert "raw_weights" in body
        assert "processed_weights" in body
        assert "guardrail" in body
        assert set(body["symbols"]) == set(body["weights"].keys())

    def test_signal_latest_with_prev_weights(self, client):
        resp = client.get("/api/signal/latest")
        symbols = resp.json()["symbols"]
        prev = ",".join("0.05" for _ in symbols)
        resp2 = client.get(f"/api/signal/latest?prev_weights={prev}&portfolio_value=0.9&peak_value=1.0")
        assert resp2.status_code == 200

    def test_signal_latest_bad_prev_weights_length(self, client):
        resp = client.get("/api/signal/latest?prev_weights=0.1,0.2")
        assert resp.status_code == 400

    def test_signal_latest_malformed_prev_weights(self, client):
        resp = client.get("/api/signal/latest?prev_weights=abc,def")
        assert resp.status_code == 400

    def test_signal_latest_404_without_model(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        app = create_app(output_dir=str(tmp_path / "empty_output"))
        client = TestClient(app)
        resp = client.get("/api/signal/latest")
        assert resp.status_code == 404
