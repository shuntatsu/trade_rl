from __future__ import annotations

from types import SimpleNamespace

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.universal_architecture import (
    UniversalArchitectureName,
    apply_architecture_to_training_config,
)


def _digest(label: str) -> str:
    return content_digest(label)


def _config() -> ResidualTrainingConfig:
    return apply_architecture_to_training_config(
        ResidualTrainingConfig(
            timesteps=128,
            gamma=1.0,
            seeds=(7, 11),
            n_steps=8,
            batch_size=8,
        ),
        UniversalArchitectureName.U_MEDIUM_DIRECT,
    )


def _manifest(config: ResidualTrainingConfig) -> dict[str, object]:
    config_digest = content_digest(config.digest_payload())
    payload: dict[str, object] = {
        "schema_version": "universal_training_run_v1",
        "architecture_name": UniversalArchitectureName.U_MEDIUM_DIRECT.value,
        "train_symbols": ["AAAUSDT", "BBBUSDT"],
        "catalog_digest": _digest("catalog"),
        "partition_digest": _digest("partition"),
        "split_manifest_digest": _digest("split"),
        "feature_schema_digest": _digest("feature"),
        "statistics_digest": _digest("stats"),
        "instrument_context_schema_digest": _digest("context"),
        "training_contract_digest": _digest("training-contract"),
        "pretraining_artifact_digest": _digest("pretraining"),
        "training_config_digest": config_digest,
        "members": [
            {
                "seed": 7,
                "policy_file": "seed-7/policy.zip",
                "policy_digest": _digest("policy:7"),
                "environment_digest": _digest("environment"),
                "architecture_digest": _digest("architecture"),
                "actual_timesteps": 128,
            },
            {
                "seed": 11,
                "policy_file": "seed-11/policy.zip",
                "policy_digest": _digest("policy:11"),
                "environment_digest": _digest("environment"),
                "architecture_digest": _digest("architecture"),
                "actual_timesteps": 128,
            },
        ],
        "research_success": False,
        "research_success_reason": "sealed zero-shot evidence not evaluated by training runner",
    }
    return {**payload, "run_digest": content_digest(payload)}


def _checkpoint(seed: int, config: ResidualTrainingConfig) -> object:
    architecture_digest = _digest("architecture")
    return SimpleNamespace(
        digest=_digest(f"checkpoint:{seed}"),
        algorithm=config.algorithm,
        seed=seed,
        requested_timestep=config.timesteps,
        observed_timestep=config.timesteps,
        environment_digest=_digest("environment"),
        training_config_digest=content_digest(config.digest_payload()),
        algorithm_identity={
            "schema_version": "sb3_checkpoint_identity_v2",
            "policy": {
                "schema_version": "sb3_policy_identity_v4",
                "policy_architecture_digest": architecture_digest,
            },
            "algorithm": None,
        },
    )


def test_build_universal_stage_a_candidate_uses_final_checkpoint_manifests(
    monkeypatch,
    tmp_path,
) -> None:
    import trade_rl.workflows.universal_stage_a as module
    from trade_rl.workflows.universal_stage_a import (
        build_universal_stage_a_candidate_from_training,
    )

    config = _config()
    checkpoints = {
        7: (_checkpoint(7, config),),
        11: (_checkpoint(11, config),),
    }

    def manifests(root):
        seed = int(root.parent.name.removeprefix("seed-"))
        return checkpoints[seed]

    monkeypatch.setattr(module, "checkpoint_manifests", manifests)

    candidate = build_universal_stage_a_candidate_from_training(
        architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
        training_config=config,
        training_manifest=_manifest(config),
        output_root=tmp_path,
    )

    assert candidate.architecture is UniversalArchitectureName.U_MEDIUM_DIRECT
    assert candidate.stage_a_candidate.candidate_id == "u_medium_direct"
    assert candidate.stage_a_candidate.candidate_config_digest == content_digest(
        config.digest_payload()
    )
    assert candidate.stage_a_candidate.policy_identity == _digest("architecture")
    assert candidate.stage_a_candidate.checkpoint_digests == (
        (7, _digest("checkpoint:7")),
        (11, _digest("checkpoint:11")),
    )
    assert (
        candidate.stage_a_candidate.final_training_completion_digest
        == _manifest(config)["run_digest"]
    )


def test_build_universal_stage_a_candidate_rejects_missing_final_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    import trade_rl.workflows.universal_stage_a as module
    from trade_rl.workflows.universal_stage_a import (
        build_universal_stage_a_candidate_from_training,
    )

    config = _config()
    monkeypatch.setattr(
        module,
        "checkpoint_manifests",
        lambda _root: (),
    )

    with pytest.raises(ValueError, match="final checkpoint"):
        build_universal_stage_a_candidate_from_training(
            architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
            training_config=config,
            training_manifest=_manifest(config),
            output_root=tmp_path,
        )


def test_build_universal_stage_a_candidate_rejects_policy_identity_drift(
    monkeypatch,
    tmp_path,
) -> None:
    import trade_rl.workflows.universal_stage_a as module
    from trade_rl.workflows.universal_stage_a import (
        build_universal_stage_a_candidate_from_training,
    )

    config = _config()
    second = _checkpoint(11, config)
    second.algorithm_identity["policy"]["policy_architecture_digest"] = _digest(
        "wrong-architecture"
    )
    checkpoints = {
        7: (_checkpoint(7, config),),
        11: (second,),
    }

    def manifests(root):
        seed = int(root.parent.name.removeprefix("seed-"))
        return checkpoints[seed]

    monkeypatch.setattr(module, "checkpoint_manifests", manifests)

    with pytest.raises(ValueError, match="policy architecture"):
        build_universal_stage_a_candidate_from_training(
            architecture=UniversalArchitectureName.U_MEDIUM_DIRECT,
            training_config=config,
            training_manifest=_manifest(config),
            output_root=tmp_path,
        )
