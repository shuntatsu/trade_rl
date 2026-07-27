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
    '''        rng_state = self._torch_rng_state()
        policy_training = self.policy.training
        self.policy.set_training_mode(False)
        self.cost_critic.train(True)
        feature_cache = self._build_cost_feature_cache()
        try:
            for _ in range(self.cost_n_epochs):
''',
    '''        rng_state = self._torch_rng_state()
        policy_training = self.policy.training
        critic_training = self.cost_critic.training
        self.policy.set_training_mode(False)
        self.cost_critic.train(True)
        completed = False
        try:
            feature_cache = self._build_cost_feature_cache()
            for _ in range(self.cost_n_epochs):
''',
    seam="Cost Critic cache failure guard",
)
source = replace_once(
    source,
    '''                    self.cost_update_count += 1
                    losses.append(float(total_loss.detach().cpu()))
        finally:
            self.policy.set_training_mode(policy_training)
            self._restore_torch_rng_state(rng_state)
''',
    '''                    self.cost_update_count += 1
                    losses.append(float(total_loss.detach().cpu()))
            completed = True
        finally:
            if not completed:
                self.cost_critic.train(critic_training)
            self.policy.set_training_mode(policy_training)
            self._restore_torch_rng_state(rng_state)
''',
    seam="Cost Critic failure restoration",
)
path.write_text(source, encoding="utf-8")
