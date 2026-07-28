from __future__ import annotations

from pathlib import Path


TEST_PATH = Path("tests/serving/test_package.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} anchor count was {count}, expected 1")
    print(f"applied: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TEST_PATH.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from trade_rl.release.offline_signing import public_key_bytes\n"
        "from trade_rl.serving.bundle import load_serving_bundle\n",
        "from trade_rl.release.offline_signing import public_key_bytes\n"
        "from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA\n"
        "from trade_rl.serving.bundle import load_serving_bundle\n",
        label="sequence schema import",
    )
    text = replace_once(
        text,
        "from trade_rl.serving.package import package_selected_training_run\n",
        "from trade_rl.serving.package import package_selected_training_run\n"
        "from trade_rl.serving.policy_loader import STRUCTURED_POLICY_LOADER_NAME\n",
        label="structured loader name import",
    )
    text = replace_once(
        text,
        '''def _training_run(
    root: Path,
    *,
    run_kind: str,
    execution_path_mode: str = "conservative",
    execution_policy_digest: str | None = None,
) -> TrainingRunManifest:
''',
        '''def _training_run(
    root: Path,
    *,
    run_kind: str,
    execution_path_mode: str = "conservative",
    execution_policy_digest: str | None = None,
    observation_schema: str = "portfolio_observation_v3",
    architecture_digest: str | None = None,
    include_structured_loader: bool = False,
) -> TrainingRunManifest:
''',
        label="structured helper parameters",
    )
    text = replace_once(
        text,
        '        "normalizer_digest": None,\n'
        '        "observation_schema": "portfolio_observation_v3",\n',
        '        "normalizer_digest": None,\n'
        '        "architecture_digest": architecture_digest,\n'
        '        "observation_schema": observation_schema,\n',
        label="ensemble architecture fields",
    )
    text = replace_once(
        text,
        '    (root / "policy-loader.json").write_text("{}", encoding="utf-8")\n'
        '    (root / "policy.zip").write_bytes(b"policy")\n',
        '    (root / "policy-loader.json").write_text("{}", encoding="utf-8")\n'
        '    (root / "policy.zip").write_bytes(b"policy")\n'
        '    artifact_paths = [\n'
        '        "ensemble.json",\n'
        '        "environment.json",\n'
        '        "execution-evidence.json",\n'
        '        "metadata-promotion.json",\n'
        '        "policy-loader.json",\n'
        '        "policy.zip",\n'
        '    ]\n'
        '    if include_structured_loader:\n'
        '        (root / STRUCTURED_POLICY_LOADER_NAME).write_text(\n'
        '            "{}", encoding="utf-8"\n'
        '        )\n'
        '        artifact_paths.append(STRUCTURED_POLICY_LOADER_NAME)\n',
        label="structured loader fixture",
    )
    text = replace_once(
        text,
        '''        artifact_paths=(
            "ensemble.json",
            "environment.json",
            "execution-evidence.json",
            "metadata-promotion.json",
            "policy-loader.json",
            "policy.zip",
        ),
''',
        '''        artifact_paths=tuple(artifact_paths),
''',
        label="dynamic artifact paths",
    )
    tests = '''

def test_package_binds_structured_loader_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architecture_digest = "9" * 64
    training_root = tmp_path / "training"
    training = _training_run(
        training_root,
        run_kind="research_selected_final",
        observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
        architecture_digest=architecture_digest,
        include_structured_loader=True,
    )
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)
    monkeypatch.setattr(
        "trade_rl.serving.policy_loader.load_structured_policy_loader_manifest",
        lambda _: {"architecture_digest": architecture_digest},
    )

    manifest = package_selected_training_run(
        training_root=training_root,
        confirmation_path=confirmation_path,
        output_root=tmp_path / "bundle",
        signal_digest="a" * 64,
        selection_digest="b" * 64,
        trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
        trusted_now=training.completed_at + timedelta(days=30),
    )

    assert manifest.architecture_digest == architecture_digest


def test_package_rejects_structured_loader_architecture_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_root = tmp_path / "training"
    training = _training_run(
        training_root,
        run_kind="research_selected_final",
        observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
        architecture_digest="9" * 64,
        include_structured_loader=True,
    )
    confirmation_path = tmp_path / "confirmation.json"
    _confirmation(confirmation_path, training)
    monkeypatch.setattr(
        "trade_rl.serving.policy_loader.load_structured_policy_loader_manifest",
        lambda _: {"architecture_digest": "8" * 64},
    )

    with pytest.raises(ValueError, match="architecture differs"):
        package_selected_training_run(
            training_root=training_root,
            confirmation_path=confirmation_path,
            output_root=tmp_path / "bundle",
            signal_digest="a" * 64,
            selection_digest="b" * 64,
            trusted_confirmation_keys={PUBLIC_KEY.key_id: PUBLIC_KEY},
            trusted_now=training.completed_at + timedelta(days=30),
        )
'''
    text = replace_once(
        text,
        "\n\ndef test_package_rejects_missing_paper_reconciliation(tmp_path: Path) -> None:\n",
        tests
        + "\n\ndef test_package_rejects_missing_paper_reconciliation(tmp_path: Path) -> None:\n",
        label="structured packaging coverage tests",
    )
    TEST_PATH.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
