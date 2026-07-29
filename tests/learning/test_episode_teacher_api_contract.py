from __future__ import annotations


def test_episode_teacher_integration_api_is_public() -> None:
    from trade_rl.integrations.sb3_training import _oracle_episode_sampling_config
    from trade_rl.learning.episode_behavior_cloning import behavior_cloning_split
    from trade_rl.learning.episode_teacher_artifact import (
        collect_episode_teacher_rollout,
    )

    assert callable(_oracle_episode_sampling_config)
    assert callable(behavior_cloning_split)
    assert callable(collect_episode_teacher_rollout)
