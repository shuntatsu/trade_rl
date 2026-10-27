"""Explicit v2 verified inference for declared runtime execution scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    encode_portfolio_decision_snapshot,
    portfolio_action_to_proposal,
)
from trade_rl.strategies.rl.portfolio_env import SharedPortfolioGymEnv
from trade_rl.strategies.rl.portfolio_inference_bundle import (
    _load_verified_bundle_policy,
    _predict_portfolio_action,
    _validate_portfolio_snapshot_window,
    _VerifiedPortfolioInferenceBundleInputs,
    _verify_portfolio_inference_bundle_inputs,
)
from trade_rl.strategies.rl.portfolio_runtime_scenario import (
    PortfolioRuntimeExecutionScenarioV2,
)
from trade_rl.strategies.rl.portfolio_scenario_runtime import (
    PortfolioScenarioGymObservationWrapperV2,
    capture_portfolio_scenario_inference_source_digest,
    create_portfolio_scenario_runtime_binding_v2,
)
from trade_rl.strategies.rl.portfolio_transform import (
    PortfolioObservationTransform,
    PortfolioScenarioObservationRuntimeBindingV2,
)


@dataclass(frozen=True, slots=True)
class PortfolioScenarioInferencePolicy:
    """Deterministic PPO prediction bound to one validated v2 runtime scenario."""

    _policy: Any
    _transform: PortfolioObservationTransform
    _action_space: gym.Space[Any]
    runtime_binding: PortfolioScenarioObservationRuntimeBindingV2
    _observation_wrapper: PortfolioScenarioGymObservationWrapperV2
    _target_abs_weights: tuple[float, ...]

    def predict(self, raw_observation: object) -> np.ndarray:
        """Transform a current raw native observation and predict one action."""

        transformed = self._observation_wrapper.observation(raw_observation)
        return _predict_portfolio_action(self._policy, transformed, self._action_space)

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        """Validate a native snapshot and map the loaded policy action to targets."""

        if not isinstance(snapshot, PortfolioDecisionSnapshot):
            raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
        self._observation_wrapper.validate_environment()
        binding = self.runtime_binding
        _validate_portfolio_snapshot_window(
            snapshot,
            symbols=self._transform.symbols,
            feature_names=self._transform.feature_names,
            global_feature_names=self._transform.global_feature_names,
            runtime_start_index=binding.runtime_start_index,
            runtime_agent_stop_index=binding.runtime_agent_stop_index,
            runtime_start_ns=binding.runtime_start_ns,
            native_interval_seconds=binding.native_interval_seconds,
        )
        raw_observation = encode_portfolio_decision_snapshot(snapshot)
        action = self.predict(raw_observation)
        return portfolio_action_to_proposal(action, snapshot, self._target_abs_weights)


def load_portfolio_scenario_inference_bundle(
    root: str | Path,
    *,
    expected_bundle_sha256: str,
    expected_manifest_sha256: str,
    expected_completion_sha256: str,
    expected_training_context_sha256: str,
    expected_checkpoint_sha256: str,
    expected_scenario: PortfolioRuntimeExecutionScenarioV2,
    expected_scenario_sha256: str,
    expected_scenario_inference_source_sha256: str,
    runtime_env: SharedPortfolioGymEnv,
) -> PortfolioScenarioInferencePolicy:
    """Load one pinned checkpoint only after full v2 runtime-scenario checks."""

    if type(expected_scenario) is not PortfolioRuntimeExecutionScenarioV2:
        raise TypeError(
            "expected_scenario must be exactly PortfolioRuntimeExecutionScenarioV2"
        )
    require_sha256(expected_scenario_sha256, field="expected_scenario_sha256")
    require_sha256(
        expected_scenario_inference_source_sha256,
        field="expected_scenario_inference_source_sha256",
    )
    expected_scenario.to_bytes()
    if expected_scenario.digest != expected_scenario_sha256:
        raise ValueError("scenario digest differs from its external expected pin")
    if type(runtime_env) is not SharedPortfolioGymEnv:
        raise TypeError("runtime_env must be exactly SharedPortfolioGymEnv")

    verified = _verify_portfolio_inference_bundle_inputs(
        root,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_completion_sha256=expected_completion_sha256,
        expected_training_context_sha256=expected_training_context_sha256,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    binding = _scenario_binding(
        verified,
        runtime_env,
        expected_scenario,
        expected_scenario_sha256=expected_scenario_sha256,
        expected_scenario_inference_source_sha256=(
            expected_scenario_inference_source_sha256
        ),
        require_unreset=True,
    )
    observation_wrapper = _make_observation_wrapper(
        verified,
        runtime_env,
        expected_scenario,
        binding,
        expected_scenario_sha256=expected_scenario_sha256,
        expected_scenario_inference_source_sha256=(
            expected_scenario_inference_source_sha256
        ),
    )
    policy = _load_verified_bundle_policy(
        verified,
        action_space=runtime_env.action_space,
    )
    return PortfolioScenarioInferencePolicy(
        _policy=policy,
        _transform=verified.transform,
        _action_space=runtime_env.action_space,
        runtime_binding=binding,
        _observation_wrapper=observation_wrapper,
        _target_abs_weights=tuple(runtime_env.target_abs_weights),
    )


def _scenario_binding(
    verified: _VerifiedPortfolioInferenceBundleInputs,
    runtime_env: SharedPortfolioGymEnv,
    scenario: PortfolioRuntimeExecutionScenarioV2,
    *,
    expected_scenario_sha256: str,
    expected_scenario_inference_source_sha256: str,
    require_unreset: bool,
) -> PortfolioScenarioObservationRuntimeBindingV2:
    if capture_portfolio_scenario_inference_source_digest() != (
        expected_scenario_inference_source_sha256
    ):
        raise ValueError("current scenario inference source differs from its pin")
    return create_portfolio_scenario_runtime_binding_v2(
        verified.transform,
        verified.training_bound,
        runtime_env,
        training_context=verified.training_context,
        training_context_sha256=cast(
            str, verified.file_records["training_context.json"]["sha256"]
        ),
        checkpoint_sha256=cast(str, verified.file_records["policy.zip"]["sha256"]),
        scenario=scenario,
        expected_scenario_sha256=expected_scenario_sha256,
        scenario_inference_source_sha256=(expected_scenario_inference_source_sha256),
        require_unreset=require_unreset,
    )


def _make_observation_wrapper(
    verified: _VerifiedPortfolioInferenceBundleInputs,
    runtime_env: SharedPortfolioGymEnv,
    scenario: PortfolioRuntimeExecutionScenarioV2,
    binding: PortfolioScenarioObservationRuntimeBindingV2,
    *,
    expected_scenario_sha256: str,
    expected_scenario_inference_source_sha256: str,
) -> PortfolioScenarioGymObservationWrapperV2:
    return PortfolioScenarioGymObservationWrapperV2(
        runtime_env,
        verified.transform,
        verified.training_bound,
        binding,
        training_context=verified.training_context,
        training_context_sha256=cast(
            str, verified.file_records["training_context.json"]["sha256"]
        ),
        checkpoint_sha256=cast(str, verified.file_records["policy.zip"]["sha256"]),
        scenario=scenario,
        expected_scenario_sha256=expected_scenario_sha256,
        scenario_inference_source_sha256=(expected_scenario_inference_source_sha256),
    )


__all__ = [
    "PortfolioScenarioInferencePolicy",
    "load_portfolio_scenario_inference_bundle",
]
