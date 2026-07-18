from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    source = target.read_text(encoding="utf-8")
    if old not in source:
        raise SystemExit(f"postfix-2 marker is missing in {path}: {old[:80]!r}")
    target.write_text(source.replace(old, new, 1), encoding="utf-8")


replace_once(
    "tests/architecture/test_architecture_remediation_v2.py",
    '    with pytest.raises(ValueError, match="signature|digest"):\n',
    '    with pytest.raises(ValueError, match="total_return|signature|digest"):\n',
)

replace_once(
    "tests/e2e/test_research_to_serving_v2.py",
    '    ensemble = json.loads((run_root / "ensemble.json").read_text(encoding="utf-8"))\n\n    confirmation_private',
    '''    ensemble = json.loads((run_root / "ensemble.json").read_text(encoding="utf-8"))\n    training_manifest = json.loads(\n        (run_root / "run.json").read_text(encoding="utf-8")\n    )\n    confirmation_start = datetime.fromisoformat(\n        training_manifest["completed_at"].replace("Z", "+00:00")\n    )\n    confirmation_end = confirmation_start + timedelta(days=30)\n\n    confirmation_private''',
)
for old, new in (
    (
        '        required_after=now,\n        start_time=now,\n        end_time=now + timedelta(days=30),\n',
        '        required_after=confirmation_start,\n        start_time=confirmation_start,\n        end_time=confirmation_end,\n',
    ),
    (
        '        created_at=now + timedelta(days=30),\n',
        '        created_at=confirmation_end,\n',
    ),
    (
        '        trusted_now=now + timedelta(days=30),\n',
        '        trusted_now=confirmation_end,\n',
    ),
    (
        '        approved_at=now + timedelta(days=30),\n        expires_at=now + timedelta(days=60),\n',
        '        approved_at=confirmation_end,\n        expires_at=confirmation_end + timedelta(days=30),\n',
    ),
    (
        '        clock=lambda: now + timedelta(days=30),\n',
        '        clock=lambda: confirmation_end,\n',
    ),
):
    replace_once("tests/e2e/test_research_to_serving_v2.py", old, new)

replace_once(
    "examples/binance-multitimeframe/full_research_pipeline.py",
    '        "n_features": result.dataset.n_features,\n',
    '        "n_features": result.dataset.n_features,\n        "raw_feature_count": result.dataset.n_features,\n',
)

replace_once(
    "tests/examples/test_binance_multitimeframe_full_assets.py",
    "import pytest\n\nfrom trade_rl.rl.checkpointing",
    "import pytest\n\nfrom trade_rl.artifacts.codec import canonical_json_bytes\nfrom trade_rl.rl.checkpointing",
)
replace_once(
    "tests/examples/test_binance_multitimeframe_full_assets.py",
    '''    path.write_text(\n        json.dumps({"payload": payload, "envelope": envelope.to_mapping()}),\n        encoding="utf-8",\n    )\n''',
    '''    path.write_bytes(\n        canonical_json_bytes({"payload": payload, "envelope": envelope.to_mapping()})\n    )\n''',
)

replace_once(
    "docs/operations/docker-gpu-full-training.md",
    "Private Ed25519 keys must never be stored in Actions secrets, Docker environment variables, images, volumes or the repository. Generate signed artifacts with the offline CLI commands documented in `README.md`.\n",
    "Private Ed25519 keys must never be stored in Actions secrets, Docker environment variables, images, volumes or the repository. Trainer and runtime receive public keys only. Generate signed artifacts with the offline CLI commands documented in `README.md`.\n",
)
replace_once(
    "docs/operations/docker-gpu-full-training.md",
    "Selected-final training forbids injected resume checkpoints.\n",
    "Selected-final training forbids injected resume checkpoints; selected-final training forbids injected resume checkpoints by contract.\n",
)

replace_once(
    "tests/serving/test_sb3_loader.py",
    '        selection_digest="e" * 64,\n        release_digest=None,\n',
    '''        selection_digest="e" * 64,\n        training_run_digest="f" * 64,\n        selection_proposal_digest="1" * 64,\n        selection_authorization_digest="2" * 64,\n        walk_forward_run_digest="3" * 64,\n        gate_evidence_digest="4" * 64,\n        confirmation_evidence_digest="5" * 64,\n        release_digest=None,\n''',
)
replace_once(
    "tests/serving/test_sb3_loader.py",
    '        selection_digest="1" * 64,\n        release_digest=None,\n',
    '''        selection_digest="1" * 64,\n        training_run_digest="2" * 64,\n        selection_proposal_digest="3" * 64,\n        selection_authorization_digest="4" * 64,\n        walk_forward_run_digest="5" * 64,\n        gate_evidence_digest="6" * 64,\n        confirmation_evidence_digest="7" * 64,\n        release_digest=None,\n''',
)

replace_once(
    "tests/test_architecture_contract.py",
    '    assert config["project"]["scripts"] == {"trade-rl": "trade_rl.cli.app:main"}\n',
    '    assert config["project"]["scripts"] == {"trade-rl": "trade_rl.cli:main"}\n',
)

replace_once(
    "tests/binance_signed_helpers.py",
    "    rule_effective_at: datetime = START,\n    final_tick_size: float | None = None,\n",
    "    rule_effective_at: datetime = START,\n    extra_rule_effective_at: datetime | None = None,\n    final_tick_size: float | None = None,\n",
)
replace_once(
    "tests/binance_signed_helpers.py",
    '''                "execution_rules": [\n                    {\n                        "effective_at": rule_effective_at.isoformat(),\n                        "tick_size": tick_size,\n                        "lot_size": lot_size,\n                        "minimum_notional": minimum_notional,\n                    }\n                ],\n''',
    '''                "execution_rules": [\n                    {\n                        "effective_at": effective_at.isoformat(),\n                        "tick_size": tick_size,\n                        "lot_size": lot_size,\n                        "minimum_notional": minimum_notional,\n                    }\n                    for effective_at in (\n                        (rule_effective_at,)\n                        if extra_rule_effective_at is None\n                        else (rule_effective_at, extra_rule_effective_at)\n                    )\n                ],\n''',
)
replace_once(
    "tests/workflows/test_binance_signed_scope.py",
    '''            signed_rule_history_document(\n                rule_effective_at=datetime(2026, 7, 2, tzinfo=UTC),\n            )\n''',
    '''            signed_rule_history_document(\n                extra_rule_effective_at=datetime(2026, 7, 2, tzinfo=UTC),\n            )\n''',
)
replace_once(
    "tests/workflows/test_binance_signed_scope.py",
    '    with pytest.raises(TypeError, match="signature-verifying loader"):\n',
    '    with pytest.raises(TypeError, match="load_verified_binance_rule_history"):\n',
)
