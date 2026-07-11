from pathlib import Path

import pytest

from mars_lite.serving.bundle import load_bundle
from mars_lite.serving.candidate import create_candidate_bundle


def _kwargs(tmp_path: Path, **overrides):
    model = tmp_path / "source.zip"
    model.write_bytes(b"model")
    kwargs = {
        "destination": tmp_path / "candidate",
        "model_source": model,
        "version": "v1",
        "git_sha": "a" * 40,
        "symbols": ("BTCUSDT",),
        "feature_names": ("ret", "vol"),
        "global_feature_names": ("market_ret",),
        "feature_norm": "none",
        "feature_mask": (True, False),
        "observation_dim": 7,
        "observation_schema_version": 1,
        "post_processor": {"vol_lookback": 60},
        "run_config": {"observation_progress_mode": "zero"},
        "metrics": {"gate2": {"passed": True}},
        "guardrails": {},
        "pre_trade": {},
    }
    kwargs.update(overrides)
    return kwargs


def _create(tmp_path: Path, **overrides):
    return create_candidate_bundle(**_kwargs(tmp_path, **overrides))


def test_create_complete_single_model_candidate(tmp_path: Path) -> None:
    candidate = _create(tmp_path)
    loaded = load_bundle(candidate)
    assert loaded.version == "v1"
    assert loaded.preprocessing["post_mask_dim"] == 2
    assert loaded.metadata["observation_dim"] == 7
    assert loaded.metadata["model_kind"] == "single"
    assert loaded.model_path.name == "model.zip"


def test_create_complete_ensemble_candidate(tmp_path: Path) -> None:
    ensemble = tmp_path / "source-ensemble"
    ensemble.mkdir()
    (ensemble / "seed_0.zip").write_bytes(b"seed-0")
    (ensemble / "seed_1.zip").write_bytes(b"seed-1")

    candidate = _create(tmp_path, model_source=ensemble)
    loaded = load_bundle(candidate)

    assert loaded.metadata["model_kind"] == "ensemble"
    assert loaded.model_path.name == "ensemble"
    assert sorted(path.name for path in loaded.model_path.glob("seed_*.zip")) == [
        "seed_0.zip",
        "seed_1.zip",
    ]


def test_empty_or_unrecognized_ensemble_source_is_rejected(tmp_path: Path) -> None:
    ensemble = tmp_path / "source-ensemble"
    ensemble.mkdir()
    (ensemble / "model.zip").write_bytes(b"not-a-seed-layout")

    with pytest.raises(ValueError, match=r"seed_\*\.zip"):
        _create(tmp_path, model_source=ensemble)


def test_episode_progress_model_is_not_serving_compatible(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="observation_progress_mode"):
        _create(
            tmp_path,
            feature_names=("ret",),
            feature_mask=None,
            observation_dim=5,
            run_config={"observation_progress_mode": "episode"},
        )


@pytest.mark.parametrize("git_sha", ["abc123", "g" * 40, "a" * 41, ""])
def test_invalid_git_identity_is_rejected(tmp_path: Path, git_sha: str) -> None:
    with pytest.raises(ValueError, match="git_sha"):
        _create(tmp_path, git_sha=git_sha)


@pytest.mark.parametrize("version", ["", " bad", "v1/escape", "x" * 51])
def test_invalid_version_is_rejected(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="version"):
        _create(tmp_path, version=version)


def test_invalid_rank_window_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rank normalization"):
        _create(tmp_path, rank_window=20, rank_min_periods=40)
