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
    "from collections.abc import Iterable\n",
    "from collections.abc import Callable, Iterable\n",
    seam="callable import",
)
source = replace_once(
    source,
    '''    def _cost_features(self, observations: Any) -> torch.Tensor:
        features = self.policy.extract_features(observations)
        if isinstance(features, tuple):
            features = features[1]
        if not isinstance(features, torch.Tensor):
            raise RuntimeError("policy feature extraction did not return a tensor")
        return features.detach()

    def _predict_cost_values(self, observations: Any) -> torch.Tensor:
        features = self._cost_features(observations)
        return self.cost_critic(features).values
''',
    '''    @staticmethod
    def _select_cost_features(features: object) -> torch.Tensor:
        """Preserve the maintained Cost Critic feature-selection semantics."""

        selected = features[1] if isinstance(features, tuple) else features
        if not isinstance(selected, torch.Tensor):
            raise RuntimeError("policy feature extraction did not return a tensor")
        return selected

    def _cost_features(self, observations: Any) -> torch.Tensor:
        return self._select_cost_features(
            self.policy.extract_features(observations)
        ).detach()

    def _run_policy_with_cost_features(
        self,
        operation: Callable[[], Any],
    ) -> tuple[Any, torch.Tensor]:
        """Run one policy operation and capture its exact Cost Critic features."""

        if not callable(operation):
            raise TypeError("policy operation must be callable")
        policy = self.policy
        original = policy.extract_features
        namespace = getattr(policy, "__dict__", None)
        if isinstance(namespace, dict) and "extract_features" in namespace:
            had_local = True
            local_value = namespace["extract_features"]
        else:
            had_local = False
            local_value = None
        captured: list[torch.Tensor] = []

        def capture(*args: Any, **kwargs: Any) -> Any:
            features = original(*args, **kwargs)
            captured.append(self._select_cost_features(features))
            return features

        policy.extract_features = capture  # type: ignore[method-assign]
        try:
            result = operation()
        finally:
            if had_local:
                policy.extract_features = local_value  # type: ignore[method-assign,assignment]
            else:
                delattr(policy, "extract_features")
        if len(captured) != 1:
            raise RuntimeError(
                "policy operation must extract features exactly once for Cost Critic reuse"
            )
        return result, captured[0].detach()

    def _predict_cost_values(self, observations: Any) -> torch.Tensor:
        features = self._cost_features(observations)
        return self.cost_critic(features).values
''',
    seam="feature capture helpers",
)
source = replace_once(
    source,
    '''            with torch.no_grad():
                obs_tensor = obs_as_tensor(cast(Any, self._last_obs), self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                cost_values = self._predict_cost_values(obs_tensor)
''',
    '''            with torch.no_grad():
                obs_tensor = obs_as_tensor(cast(Any, self._last_obs), self.device)
                policy_output, cost_features = self._run_policy_with_cost_features(
                    lambda: self.policy(obs_tensor)
                )
                actions, values, log_probs = policy_output
                cost_values = self.cost_critic(cost_features).values
''',
    seam="rollout feature reuse",
)
source = replace_once(
    source,
    '''                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        terminal_cost = self._predict_cost_values(terminal_obs)[0]
''',
    '''                    with torch.no_grad():
                        terminal_values, terminal_features = (
                            self._run_policy_with_cost_features(
                                lambda: self.policy.predict_values(terminal_obs)
                            )
                        )
                        terminal_value = terminal_values[0]
                        terminal_cost = self.cost_critic(terminal_features).values[0]
''',
    seam="terminal feature reuse",
)
source = replace_once(
    source,
    '''        with torch.no_grad():
            final_observations = obs_as_tensor(cast(Any, new_obs), self.device)
            final_values = self.policy.predict_values(final_observations)
            final_cost_values = self._predict_cost_values(final_observations)
''',
    '''        with torch.no_grad():
            final_observations = obs_as_tensor(cast(Any, new_obs), self.device)
            final_values, final_features = self._run_policy_with_cost_features(
                lambda: self.policy.predict_values(final_observations)
            )
            final_cost_values = self.cost_critic(final_features).values
''',
    seam="final feature reuse",
)
source = replace_once(
    source,
    '''    def _cost_head_parameters(
''',
    '''    def _build_cost_feature_cache(self) -> torch.Tensor:
        """Materialize one immutable post-PPO feature tensor for this rollout."""

        transition_count = self.n_steps * self.n_envs
        indices = np.arange(transition_count, dtype=np.int64)
        observations = self._rollout_observations(indices)
        policy_training = self.policy.training
        self.policy.set_training_mode(False)
        try:
            with torch.no_grad():
                cache = self._cost_features(observations)
        finally:
            self.policy.set_training_mode(policy_training)
        if cache.ndim != 2 or cache.shape[0] != transition_count:
            raise RuntimeError("Cost Critic feature cache has an invalid rollout shape")
        if cache.device != self.device:
            raise RuntimeError("Cost Critic feature cache is on the wrong device")
        if not bool(torch.isfinite(cache).all()):
            raise RuntimeError("Cost Critic feature cache contains non-finite values")
        return cache.detach()

    def _cached_cost_features(
        self,
        cache: torch.Tensor,
        indices: np.ndarray,
    ) -> torch.Tensor:
        """Select canonical rollout rows from the device-local feature cache."""

        if not isinstance(cache, torch.Tensor) or cache.ndim != 2:
            raise TypeError("Cost Critic feature cache must be a rank-two tensor")
        raw_indices = np.asarray(indices)
        if raw_indices.ndim != 1 or not np.issubdtype(raw_indices.dtype, np.integer):
            raise ValueError("Cost Critic cache indices must be one-dimensional integers")
        normalized = np.asarray(raw_indices, dtype=np.int64)
        if normalized.size == 0:
            raise ValueError("Cost Critic cache indices must not be empty")
        if np.any(normalized < 0) or np.any(normalized >= cache.shape[0]):
            raise ValueError("Cost Critic cache index is outside the rollout")
        tensor_indices = torch.as_tensor(
            normalized,
            dtype=torch.long,
            device=cache.device,
        )
        return cache.index_select(0, tensor_indices)

    def _cost_head_parameters(
''',
    seam="rollout feature cache helpers",
)
source = replace_once(
    source,
    '''    def _build_cost_training_diagnostics(
        self,
    ) -> tuple[
''',
    '''    def _build_cost_training_diagnostics(
        self,
        feature_cache: torch.Tensor,
    ) -> tuple[
''',
    seam="diagnostic cache parameter",
)
source = replace_once(
    source,
    '''        transition_count = self.n_steps * self.n_envs
        indices = np.arange(transition_count, dtype=np.int64)
        observations = self._rollout_observations(indices)
        policy_training = self.policy.training
        critic_training = self.cost_critic.training
        self.policy.set_training_mode(False)
        self.cost_critic.train(False)
        try:
            with torch.no_grad():
                output = self.cost_critic(self._cost_features(observations))
        finally:
            self.policy.set_training_mode(policy_training)
            self.cost_critic.train(critic_training)
''',
    '''        transition_count = self.n_steps * self.n_envs
        indices = np.arange(transition_count, dtype=np.int64)
        critic_training = self.cost_critic.training
        self.cost_critic.train(False)
        try:
            with torch.no_grad():
                output = self.cost_critic(feature_cache)
        finally:
            self.cost_critic.train(critic_training)
''',
    seam="diagnostic feature reuse",
)
source = replace_once(
    source,
    '''        self.policy.set_training_mode(False)
        self.cost_critic.train(True)
        try:
            for _ in range(self.cost_n_epochs):
''',
    '''        self.policy.set_training_mode(False)
        self.cost_critic.train(True)
        feature_cache = self._build_cost_feature_cache()
        try:
            for _ in range(self.cost_n_epochs):
''',
    seam="build one cost feature cache",
)
source = replace_once(
    source,
    '''                    indices = permutation[start : start + batch_size]
                    observations = self._rollout_observations(indices)
                    with torch.no_grad():
                        features = self._cost_features(observations)
                    output = self.cost_critic(features)
''',
    '''                    indices = permutation[start : start + batch_size]
                    features = self._cached_cost_features(feature_cache, indices)
                    output = self.cost_critic(features)
''',
    seam="cost minibatch cache selection",
)
source = replace_once(
    source,
    '''        reports, family, diagnostic_metrics = self._build_cost_training_diagnostics()
''',
    '''        reports, family, diagnostic_metrics = self._build_cost_training_diagnostics(
            feature_cache
        )
''',
    seam="diagnostic cache forwarding",
)
path.write_text(source, encoding="utf-8")
