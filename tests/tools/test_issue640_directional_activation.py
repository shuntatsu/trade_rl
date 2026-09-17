from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from trade_rl.artifacts import canonical_json_bytes, content_digest


def _load() -> ModuleType:
    path = Path('.github/scripts/issue640_directional_activation.py')
    spec = importlib.util.spec_from_file_location('issue640_directional_activation', path)
    if spec is None or spec.loader is None:
        raise AssertionError('activation helper spec unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol() -> dict[str, object]:
    return {
        'schema': 'directional_development_protocol_v1',
        'source_dataset_id': 'd' * 64,
        'source_artifact_digest': 'a' * 64,
        'source_study_digest': 's' * 64,
        'source_plan_sha256': 'p' * 64,
        'arms': [
            'cash', 'constant_long', 'constant_short', 'trend', 'mean_reversion',
            'ridge24', 'lightgbm24', 'channel_breakout', 'ppo0', 'ppo1', 'ppo2',
            'ppo3', 'ppo4',
        ],
        'required_runtime_packages': {
            'lightgbm': '4.7.0',
            'stable-baselines3': '2.3.2',
            'torch': '2.4.1',
            'scikit-learn': '1.9.0',
        },
        'unused_data_accessed': False,
        'production_eligible': False,
        'provenance': {
            'implementation_digest': '1' * 64,
            'runtime_environment_digest': '2' * 64,
        },
    }


def test_activation_claim_binds_remote_run_protocol_source_and_runtime_not_local_root() -> None:
    module = _load()
    protocol = _protocol()
    claim = module.build_activation_claim(
        protocol,
        workflow_run_id=12345,
        workflow_run_attempt=1,
        implementation_head='a' * 40,
    )
    assert claim['schema_version'] == 'issue640_directional_activation_v1'
    assert claim['issue_number'] == 640
    assert claim['workflow_run_id'] == 12345
    assert claim['workflow_run_attempt'] == 1
    assert claim['implementation_head'] == 'a' * 40
    assert claim['protocol_digest'] == content_digest(protocol)
    assert claim['implementation_digest'] == '1' * 64
    assert claim['runtime_environment_digest'] == '2' * 64
    assert claim['source_dataset_id'] == 'd' * 64
    assert claim['source_artifact_digest'] == 'a' * 64
    assert claim['source_study_digest'] == 's' * 64
    assert claim['required_runtime_packages'] == protocol['required_runtime_packages']
    assert claim['arm_roster'] == protocol['arms']
    assert claim['local_output_root_is_authority'] is False
    assert claim['economic_execution_started'] is False
    assert claim['economic_result_inspected'] is False
    assert claim['unused_data_accessed'] is False
    assert claim['production_eligible'] is False
    assert 'output' not in claim and 'output_root' not in claim
    assert module.activation_claim_bytes(claim) == canonical_json_bytes(claim)


def test_remote_activation_slot_is_global_and_second_claim_fails_closed() -> None:
    module = _load()
    module.validate_remote_slot({'total_count': 0, 'artifacts': []}, state='empty')
    claimed = {
        'total_count': 1,
        'artifacts': [{
            'id': 77,
            'name': module.ACTIVATION_ARTIFACT_NAME,
            'expired': False,
            'workflow_run': {'id': 12345},
        }],
    }
    with pytest.raises(RuntimeError, match='already claimed'):
        module.validate_remote_slot(claimed, state='empty')
    artifact = module.validate_remote_slot(
        claimed, state='claimed', workflow_run_id=12345
    )
    assert artifact['id'] == 77
    with pytest.raises(RuntimeError, match='workflow'):
        module.validate_remote_slot(claimed, state='claimed', workflow_run_id=999)


def test_activation_claim_rejects_runtime_or_identity_drift() -> None:
    module = _load()
    protocol = _protocol()
    protocol['required_runtime_packages'] = {
        'lightgbm': '4.7.0',
        'stable-baselines3': None,
        'torch': '2.4.1',
        'scikit-learn': '1.9.0',
    }
    with pytest.raises(ValueError, match='runtime'):
        module.build_activation_claim(
            protocol,
            workflow_run_id=1,
            workflow_run_attempt=1,
            implementation_head='a' * 40,
        )
    protocol = _protocol()
    with pytest.raises(ValueError, match='attempt'):
        module.build_activation_claim(
            protocol,
            workflow_run_id=1,
            workflow_run_attempt=2,
            implementation_head='a' * 40,
        )
    with pytest.raises(ValueError, match='implementation_head'):
        module.build_activation_claim(
            protocol,
            workflow_run_id=1,
            workflow_run_attempt=1,
            implementation_head='not-a-sha',
        )


def test_validate_claim_requires_same_remote_execution_identity() -> None:
    module = _load()
    protocol = _protocol()
    claim = module.build_activation_claim(
        protocol,
        workflow_run_id=12345,
        workflow_run_attempt=1,
        implementation_head='a' * 40,
    )
    module.validate_activation_claim(
        claim,
        protocol,
        workflow_run_id=12345,
        implementation_head='a' * 40,
    )
    with pytest.raises(ValueError, match='workflow_run_id'):
        module.validate_activation_claim(
            claim,
            protocol,
            workflow_run_id=12346,
            implementation_head='a' * 40,
        )
    changed = dict(protocol)
    changed['source_dataset_id'] = 'e' * 64
    with pytest.raises(ValueError, match='protocol'):
        module.validate_activation_claim(
            claim,
            changed,
            workflow_run_id=12345,
            implementation_head='a' * 40,
        )
