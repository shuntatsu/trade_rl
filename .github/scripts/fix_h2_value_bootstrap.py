from __future__ import annotations

from pathlib import Path


def replace_once(source: str, old: str, new: str, *, seam: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{seam} changed: expected one exact match, got {count}")
    return source.replace(old, new)


path = Path("trade_rl/integrations/cost_critic_ppo.py")
source = path.read_text(encoding="utf-8")
source = replace_once(
    source,
    '''from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
''',
    '''from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.utils import obs_as_tensor
''',
    seam="BaseModel import",
)
source = replace_once(
    source,
    '''    def _predict_cost_values(self, observations: Any) -> torch.Tensor:
        features = self._cost_features(observations)
        return self.cost_critic(features).values

    def collect_rollouts(
''',
    '''    def _predict_cost_values(self, observations: Any) -> torch.Tensor:
        features = self._cost_features(observations)
        return self.cost_critic(features).values

    def _predict_values_with_cost_features(
        self,
        observations: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match SB3 2.3.2 predict_values while reusing its value features."""

        features = BaseModel.extract_features(
            self.policy,
            observations,
            self.policy.vf_features_extractor,
        )
        if not isinstance(features, torch.Tensor):
            raise RuntimeError("policy value feature extraction did not return a tensor")
        latent_value = self.policy.mlp_extractor.forward_critic(features)
        values = self.policy.value_net(latent_value)
        return values, features.detach()

    def collect_rollouts(
''',
    seam="shared value helper",
)
source = replace_once(
    source,
    '''                    with torch.no_grad():
                        terminal_values, terminal_features = (
                            self._run_policy_with_cost_features(
                                lambda: self.policy.predict_values(terminal_obs)
                            )
                        )
                        terminal_value = terminal_values[0]
                        terminal_cost = self.cost_critic(terminal_features).values[0]
''',
    '''                    with torch.no_grad():
                        terminal_values, terminal_features = (
                            self._predict_values_with_cost_features(terminal_obs)
                        )
                        terminal_value = terminal_values[0]
                        terminal_cost = self.cost_critic(terminal_features).values[0]
''',
    seam="terminal value feature sharing",
)
source = replace_once(
    source,
    '''        with torch.no_grad():
            final_observations = obs_as_tensor(cast(Any, new_obs), self.device)
            final_values, final_features = self._run_policy_with_cost_features(
                lambda: self.policy.predict_values(final_observations)
            )
            final_cost_values = self.cost_critic(final_features).values
''',
    '''        with torch.no_grad():
            final_observations = obs_as_tensor(cast(Any, new_obs), self.device)
            final_values, final_features = self._predict_values_with_cost_features(
                final_observations
            )
            final_cost_values = self.cost_critic(final_features).values
''',
    seam="final value feature sharing",
)
path.write_text(source, encoding="utf-8")
