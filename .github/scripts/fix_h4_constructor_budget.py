from __future__ import annotations

from pathlib import Path


def replace_once(source: str, old: str, new: str, *, seam: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{seam} changed: expected one exact match, got {count}")
    return source.replace(old, new)


path = Path("trade_rl/rl/environment.py")
source = path.read_text(encoding="utf-8")
source = replace_once(
    source,
    '''        self.observation_space = observation_contract.observation_space
        self._full_observation_space = observation_contract.observation_space
        self._compact_sequence_training_observations = False
        self.action_space = observation_contract.action_space
''',
    '''        self._install_observation_transport(observation_contract.observation_space)
        self.action_space = observation_contract.action_space
''',
    seam="constructor observation transport installation",
)
source = replace_once(
    source,
    '''    def _install_reward_execution_resources(
        self, resources: EnvironmentRewardExecutionResources
    ) -> None:
''',
    '''    def _install_observation_transport(
        self,
        observation_space: gym.spaces.Space[Any],
    ) -> None:
        self.observation_space = observation_space
        self._full_observation_space = observation_space
        self._compact_sequence_training_observations = False

    def _install_reward_execution_resources(
        self, resources: EnvironmentRewardExecutionResources
    ) -> None:
''',
    seam="observation transport helper",
)
path.write_text(source, encoding="utf-8")
