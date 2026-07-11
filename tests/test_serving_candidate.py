from pathlib import Path

import pytest

from mars_lite.serving.bundle import load_bundle
from mars_lite.serving.candidate import create_candidate_bundle


def test_create_complete_single_model_candidate(tmp_path: Path) -> None:
    model = tmp_path / "source.zip"
    model.write_bytes(b"model")
    candidate = create_candidate_bundle(
        destination=tmp_path / "candidate",
        model_source=model,
        version="v1",
        git_sha="abc123",
        symbols=("BTCUSDT",),
        feature_names=("ret", "vol"),
        global_feature_names=("market_ret",),
        feature_norm="none",
        feature_mask=(True, False),
        observation_dim=7,
        observation_schema_version=1,
        post_processor={"vol_lookback": 60},
        run_config={"observation_progress_mode": "zero"},
        metrics={"gate2": {"passed": True}},
        guardrails={},
        pre_trade={},
    )
    loaded = load_bundle(candidate)
    assert loaded.version == "v1"
    assert loaded.preprocessing["post_mask_dim"] == 2
    assert loaded.metadata["observation_dim"] == 7
    assert loaded.model_path.name == "model.zip"


def test_episode_progress_model_is_not_serving_compatible(tmp_path: Path) -> None:
    model = tmp_path / "source.zip"
    model.write_bytes(b"model")
    with pytest.raises(ValueError, match="observation_progress_mode"):
        create_candidate_bundle(
            destination=tmp_path / "candidate",
            model_source=model,
            version="v1",
            git_sha="abc123",
            symbols=("BTCUSDT",),
            feature_names=("ret",),
            global_feature_names=(),
            feature_norm="none",
            feature_mask=None,
            observation_dim=5,
            observation_schema_version=1,
            post_processor={},
            run_config={"observation_progress_mode": "episode"},
            metrics={},
            guardrails={},
            pre_trade={},
        )
