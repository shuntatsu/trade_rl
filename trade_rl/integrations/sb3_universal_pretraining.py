"""Universal pretraining helpers for the SB3 training coordinator."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_sha256
from trade_rl.integrations.policy_stage_snapshot import write_policy_stage_snapshot
from trade_rl.integrations.universal_critic_warm_start import (
    ConfiguredCriticWarmStart,
    run_configured_critic_warm_start,
)
from trade_rl.learning import SupervisedPolicyDataset
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
from trade_rl.rl.training import ResidualTrainingConfig


def _apply_universal_pretraining_if_configured(
    *,
    hook: Callable[..., Mapping[str, object]] | None,
    policy: Any,
    config: ResidualTrainingConfig,
    behavior_cloning_seed: int,
    member_seed: int,
    output_root: Path,
) -> dict[str, object] | None:
    if hook is None:
        return None
    if config.behavior_cloning_epochs <= 0:
        raise ValueError(
            "Universal pretraining requires behavior cloning to be enabled"
        )
    if config.behavior_cloning_teacher != "oracle":
        raise ValueError(
            "Universal pretraining requires the Oracle behavior cloning teacher"
        )
    if not callable(hook):
        raise TypeError("Universal pretraining hook must be callable")
    write_policy_stage_snapshot(
        policy,
        output_root=output_root,
        stage="random",
        member_seed=member_seed,
    )
    evidence = hook(
        policy=policy,
        config=config,
        behavior_cloning_seed=behavior_cloning_seed,
        member_seed=member_seed,
        output_root=output_root,
    )
    if not isinstance(evidence, Mapping):
        raise TypeError("Universal pretraining hook must return a mapping")
    payload = dict(evidence)
    if payload.get("schema_version") != "universal_pretraining_evidence_v1":
        raise ValueError("Universal pretraining evidence schema mismatch")
    if payload.get("passed") is not True:
        raise RuntimeError("Universal pretraining failed its admission gates")
    for field in ("teacher_artifact_digest", "behavior_cloning_digest"):
        value = payload.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Universal pretraining evidence is missing {field}")
        require_sha256(value, field=field)
    critic_digest = payload.get("critic_warm_start_digest")
    if config.behavior_cloning_critic_warm_start_enabled:
        if not isinstance(critic_digest, str):
            raise ValueError(
                "Universal pretraining evidence is missing critic_warm_start_digest"
            )
        require_sha256(critic_digest, field="critic_warm_start_digest")
    elif critic_digest is not None:
        if not isinstance(critic_digest, str):
            raise ValueError(
                "critic_warm_start_digest must be a SHA-256 string when present"
            )
        require_sha256(critic_digest, field="critic_warm_start_digest")
    bound_payload: dict[str, object] = {
        **payload,
        "behavior_cloning_seed": behavior_cloning_seed,
        "member_seed": member_seed,
        "training_config_digest": content_digest(config.digest_payload()),
    }
    artifact_digest = content_digest(bound_payload)
    resolved = {**bound_payload, "artifact_digest": artifact_digest}
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        output_root / "universal-pretraining.json",
        canonical_json_bytes(resolved),
    )
    return resolved


def _run_behavior_cloning_critic_warm_start_if_enabled(
    *,
    policy: Any,
    teacher_environment: Any,
    teacher_dataset: SupervisedPolicyDataset,
    episode_batch: EpisodeOracleBatch | None,
    episode_split: BehaviorCloningSplit | None,
    config: ResidualTrainingConfig,
    observation_provider: Any | None,
    behavior_cloning_seed: int,
    output_root: Path,
    run_warm_start: Callable[
        ..., ConfiguredCriticWarmStart
    ] = run_configured_critic_warm_start,
) -> ConfiguredCriticWarmStart | None:
    if not config.behavior_cloning_critic_warm_start_enabled:
        return None
    if episode_batch is None or episode_split is None:
        raise RuntimeError(
            "critic warm-start requires Oracle episode evidence and split"
        )
    return run_warm_start(
        policy=policy,
        teacher_environment=teacher_environment,
        teacher_dataset=teacher_dataset,
        episode_batch=episode_batch,
        split=episode_split,
        config=config,
        observation_provider=observation_provider,
        behavior_cloning_seed=behavior_cloning_seed,
        output_root=output_root,
    )
