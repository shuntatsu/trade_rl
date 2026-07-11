from pathlib import Path
from types import SimpleNamespace

from mars_lite.pipeline import production_pipeline
from mars_lite.serving.bundle import load_bundle
from mars_lite.serving.registry import ModelRegistry


class _PostConfig:
    def to_dict(self):
        return {"vol_lookback": 60}


class _PostProcessor:
    cfg = _PostConfig()


class _ObservationSpace:
    shape = (9,)


class _Environment:
    observation_space = _ObservationSpace()


def test_pipeline_registers_complete_candidate_without_activation(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "portfolio_model.zip").write_bytes(b"model")
    (output / "train_report.json").write_text(
        '{"feature_mask":[true],"signal_gate":{"passed":true},"lockbox":null}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        production_pipeline,
        "build_post_processor",
        lambda args, horizon: _PostProcessor(),
    )
    monkeypatch.setattr(
        production_pipeline,
        "build_env_kwargs",
        lambda args, processor, horizon: {
            "post_processor": processor,
            "obs_risk_state": True,
            "disagreement_dr_max": 0.3,
        },
    )
    monkeypatch.setattr(
        production_pipeline,
        "PortfolioTradingEnv",
        lambda feature_set, **kwargs: _Environment(),
    )
    args = SimpleNamespace(
        git_sha="abc123",
        model_version="v1",
        horizon=4,
        ensemble=1,
        feature_norm="none",
        base_timeframe="4h",
        registry_dir=str(tmp_path / "registry"),
    )
    features = SimpleNamespace(
        symbols=["BTCUSDT"],
        feature_names=["ret"],
        global_feature_names=[],
    )
    result = {
        "agent_res": {"total_return": 0.1, "equity_curve": [1.0, 1.1]},
        "gate2": {"passed": True},
    }

    candidate = production_pipeline.build_and_register_candidate(
        args=args,
        output_dir=output,
        feature_set=features,
        train_result=result,
    )

    registry = ModelRegistry(tmp_path / "registry")
    registered = load_bundle(registry.version_dir("v1"))
    run_config = registered.metadata["run_config"]
    assert candidate.is_dir()
    assert registry.list_versions() == ["v1"]
    assert not registry.active_path.exists()
    assert registered.manifest.observation_dim == 9
    assert run_config["obs_risk_state"] is True
    assert run_config["disagreement_dr_max"] == 0.3
    assert run_config["base_timeframe"] == "4h"
    assert run_config["observation_progress_mode"] == "zero"
