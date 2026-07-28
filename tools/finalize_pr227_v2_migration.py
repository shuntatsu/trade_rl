from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


_LEGACY_ENCODER_BLOCK = re.compile(
    r'(?ms)^(?P<indent>[ \t]*)"asset_set_encoder": (?P<asset>true|false),\n'
    r'(?P<middle>.*?)'
    r'^(?P=indent)"sequence_encoder": (?P<sequence>true|false),\n'
)


def _replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    updated = text.replace(old, new, 1)
    ast.parse(updated, filename=path)
    target.write_text(updated, encoding="utf-8")


def _encoder_name(*, sequence: bool, asset: bool) -> str:
    if sequence and asset:
        raise ValueError("legacy sequence_encoder and asset_set_encoder cannot both be true")
    if sequence:
        return "hierarchical_sequence_v2"
    if asset:
        return "asset_set"
    return "flat_mlp"


def _migrate_encoder_block(match: re.Match[str]) -> str:
    sequence = match.group("sequence") == "true"
    asset = match.group("asset") == "true"
    indent = match.group("indent")
    middle = match.group("middle")
    encoder = _encoder_name(sequence=sequence, asset=asset)
    return f'{indent}"observation_encoder": "{encoder}",\n{middle}'


def _assert_no_legacy_training_fields(value: object, *, path: Path) -> None:
    if isinstance(value, Mapping):
        legacy = {
            "asset_set_encoder",
            "sequence_encoder",
            "sequence_capacity",
            "sequence_attention_heads",
            "sequence_attention_layers",
        }
        overlap = legacy.intersection(value)
        if overlap:
            raise RuntimeError(f"{path}: legacy training fields remain: {sorted(overlap)}")
        if value.get("schema_version") == "training_run_config_v1":
            raise RuntimeError(f"{path}: legacy training_run_config_v1 remains")
        for nested in value.values():
            _assert_no_legacy_training_fields(nested, path=path)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_legacy_training_fields(nested, path=path)


def _migrate_example_json(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original.replace(
        '"schema_version": "training_run_config_v1"',
        '"schema_version": "training_run_config_v2"',
    )
    updated = _LEGACY_ENCODER_BLOCK.sub(_migrate_encoder_block, updated)
    updated = updated.replace(
        '"sequence_capacity":',
        '"sequence_tcn_capacity":',
    )
    updated = re.sub(
        r'(?m)^(?P<indent>[ \t]*)"sequence_attention_heads": (?P<value>[^,]+),$',
        lambda match: (
            f'{match.group("indent")}"sequence_timeframe_attention_heads": '
            f'{match.group("value")},\n'
            f'{match.group("indent")}"sequence_asset_attention_heads": '
            f'{match.group("value")},'
        ),
        updated,
    )
    updated = re.sub(
        r'(?m)^(?P<indent>[ \t]*)"sequence_attention_layers": (?P<value>[^,]+),$',
        lambda match: (
            f'{match.group("indent")}"sequence_timeframe_attention_layers": '
            f'{match.group("value")},\n'
            f'{match.group("indent")}"sequence_asset_attention_layers": '
            f'{match.group("value")},'
        ),
        updated,
    )
    payload = json.loads(updated)
    _assert_no_legacy_training_fields(payload, path=path)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def _migrate_example_configs() -> None:
    changed = 0
    for path in sorted((ROOT / "examples").rglob("*.json")):
        changed += int(_migrate_example_json(path))
    if changed <= 0:
        raise RuntimeError("expected at least one maintained example config to migrate")


def _update_sequence_policy_fixture() -> None:
    _replace_once(
        "tests/architecture/test_architecture_audit_fixes.py",
        '''            "d_model": 16,
            "attention_heads": 4,
            "attention_layers": 1,
            "dropout": 0.0,
''',
        '''            "d_model": 16,
            "timeframe_attention_heads": 4,
            "timeframe_attention_layers": 1,
            "asset_attention_heads": 4,
            "asset_attention_layers": 1,
            "dropout": 0.0,
''',
    )


def _update_constrained_profile_contract() -> None:
    _replace_once(
        "tests/examples/test_constrained_growth_profiles.py",
        '''    assert training.sequence_d_model == 336
    assert training.sequence_attention_heads == 8
    assert training.sequence_attention_layers == 2
    assert training.sequence_compile is True
''',
        '''    assert training.sequence_d_model == 336
    assert training.sequence_timeframe_attention_heads == 8
    assert training.sequence_timeframe_attention_layers == 2
    assert training.sequence_asset_attention_heads == 8
    assert training.sequence_asset_attention_layers == 2
    assert training.sequence_compile is True
''',
    )


def _update_sb3_training_fixtures() -> None:
    _replace_once(
        "tests/integrations/test_sb3_training.py",
        '''def _training_config(*, asset_set_enabled: bool = False) -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=2,
        batch_size=2,
        n_epochs=1,
        observation_encoder=(
            "invalid_legacy_combination"
            if (False) and (asset_set_enabled)
            else "hierarchical_sequence_v2"
            if (False)
            else "asset_set"
            if (asset_set_enabled)
            else "flat_mlp"
        ),
        device="cpu",
    )
''',
        '''def _training_config(
    *, observation_encoder: str = "flat_mlp"
) -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=2,
        gamma=0.99,
        seeds=(0,),
        n_steps=1,
        n_envs=2,
        batch_size=2,
        n_epochs=1,
        observation_encoder=observation_encoder,
        device="cpu",
    )
''',
    )
    _replace_once(
        "tests/integrations/test_sb3_training.py",
        '''    class CheckpointSource:
        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"resume-policy")
''',
        '''    policy_identity = {
        "observation_encoder": "flat_mlp",
        "schema_version": "sb3_policy_identity_v1",
    }

    class CheckpointSource:
        _trade_rl_policy_identity = policy_identity

        def save(self, target: str) -> None:
            Path(target).with_suffix(".zip").write_bytes(b"resume-policy")
''',
    )
    _replace_once(
        "tests/integrations/test_sb3_training.py",
        '''    class FakeResumePPO:
        device = "cpu"
''',
        '''    class FakeResumePPO:
        _trade_rl_policy_identity = policy_identity
        device = "cpu"
''',
    )


def _update_runtime_error_contract() -> None:
    _replace_once(
        "tests/integrations/test_sequence_runtime_acceleration.py",
        '    with pytest.raises(ValueError, match="sequence_compile.*sequence_encoder"):\n',
        '    with pytest.raises(ValueError, match="sequence_compile.*observation_encoder"):\n',
    )
    _replace_once(
        "tests/integrations/test_sequence_runtime_acceleration.py",
        '    with pytest.raises(ValueError, match="sequence_transfer_mode.*sequence_encoder"):\n',
        '    with pytest.raises(ValueError, match="sequence_transfer_mode.*observation_encoder"):\n',
    )


def main() -> None:
    _migrate_example_configs()
    _update_sequence_policy_fixture()
    _update_constrained_profile_contract()
    _update_sb3_training_fixtures()
    _update_runtime_error_contract()


if __name__ == "__main__":
    main()
