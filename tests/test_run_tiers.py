import pytest

from mars_lite.pipeline.training_engine import validate_run_tier


def test_smoke_allows_five_updates() -> None:
    result = validate_run_tier(
        "smoke", timesteps=10_240, n_envs=8, n_steps=256, n_seeds=1
    )
    assert result["updates"] == 5


def test_research_requires_fifty_updates_and_three_seeds() -> None:
    with pytest.raises(ValueError, match="50 updates"):
        validate_run_tier(
            "research", timesteps=10_240, n_envs=8, n_steps=256, n_seeds=3
        )
    with pytest.raises(ValueError, match="3 seeds"):
        validate_run_tier(
            "research", timesteps=102_400, n_envs=8, n_steps=256, n_seeds=1
        )


def test_release_requires_one_hundred_updates_and_five_seeds() -> None:
    result = validate_run_tier(
        "release", timesteps=204_800, n_envs=8, n_steps=256, n_seeds=5
    )
    assert result["updates"] == 100
    assert result["required_seeds"] == 5
