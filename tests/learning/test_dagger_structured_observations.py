import numpy as np
import pytest

from trade_rl.learning.dagger import DaggerEpisodeRollout

_DATASET = "1" * 64
_ENVIRONMENT = "2" * 64
_ACTION_SPEC = "3" * 64
_TEACHER = "4" * 64


def test_structured_dagger_rollout_copies_and_freezes_each_observation_field() -> None:
    asset = np.asarray([[3.0], [4.0]], dtype=np.float32)
    global_state = np.asarray([[1.0], [2.0]], dtype=np.float32)

    rollout = DaggerEpisodeRollout(
        observations={"global": global_state, "asset": asset},
        teacher_actions=np.asarray([[-0.75], [-0.75]], dtype=np.float32),
        learner_actions=np.asarray([[0.5], [0.5]], dtype=np.float32),
        decision_indices=np.asarray([3, 4], dtype=np.int64),
        dataset_id=_DATASET,
        environment_digest=_ENVIRONMENT,
        action_spec_digest=_ACTION_SPEC,
        teacher_config_digest=_TEACHER,
        start=3,
        stop=6,
        initial_state_mode="cash",
    )

    asset[0, 0] = 99.0
    global_state[0, 0] = 88.0

    assert isinstance(rollout.observations, dict)
    assert tuple(rollout.observations) == ("asset", "global")
    assert rollout.observations["asset"][0, 0] == pytest.approx(3.0)
    assert rollout.observations["global"][0, 0] == pytest.approx(1.0)
    assert rollout.observations["asset"].flags.writeable is False
    assert rollout.observations["global"].flags.writeable is False
    with pytest.raises(ValueError):
        rollout.observations["asset"][0, 0] = 42.0
