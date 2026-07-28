from pathlib import Path

import pytest

from trade_rl.rl.training import PolicyTrainingResult


def _result(**overrides: object) -> PolicyTrainingResult:
    values: dict[str, object] = {
        "checkpoint_path": Path("policy.zip"),
        "actual_timesteps": 8,
        "resolved_device": "cpu",
        "environment_digest": "a" * 64,
        "initial_capital": 1_000.0,
        "action_size": 1,
        "action_names": ("target_weight:BTC",),
        "action_spec_digest": "b" * 64,
        "observation_size": 1,
    }
    values.update(overrides)
    return PolicyTrainingResult(**values)  # type: ignore[arg-type]


def test_policy_training_result_requires_complete_structured_export_identity() -> None:
    with pytest.raises(ValueError, match="structured export identity must be complete"):
        _result(structured_export_manifest_path=Path("structured-export.json"))


def test_policy_training_result_accepts_architecture_identity_without_export() -> None:
    result = _result(architecture_digest="e" * 64)

    assert result.architecture_digest == "e" * 64
    assert result.structured_export_manifest_path is None


def test_policy_training_result_accepts_complete_structured_export_identity() -> None:
    result = _result(
        structured_export_manifest_path=Path("structured-export.json"),
        structured_export_manifest_digest="c" * 64,
        structured_export_model_path=Path("policy.structured.torchscript.pt"),
        structured_export_model_digest="d" * 64,
        architecture_digest="e" * 64,
    )

    assert result.architecture_digest == "e" * 64
