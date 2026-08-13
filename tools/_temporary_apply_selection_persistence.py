from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"patch target drifted in {path}: {old[:120]!r}")
    target.write_text(text.replace(old, new), encoding="utf-8")


facade = "trade_rl/workflows/universal_causal_alpha_teacher.py"
replace_once(
    facade,
    "from trade_rl.artifacts.hashing import content_digest\n",
    "from trade_rl.artifacts.atomic_write import atomic_write_bytes\nfrom trade_rl.artifacts.codec import canonical_json_bytes\nfrom trade_rl.artifacts.hashing import content_digest\n",
)
replace_once(
    facade,
    '''    feature_schema_digest: str,\n    episode_hours: float | None = None,\n''',
    '''    feature_schema_digest: str,\n    selection_evidence_path: Path,\n    episode_hours: float | None = None,\n''',
)
replace_once(
    facade,
    '''    selection = evaluate_causal_alpha_selection(\n        train_symbols=symbols,\n        samples=samples,\n        partitions=partitions,\n        candidates=candidate_values,\n        environment_factories=environment_factories,\n        episode_hours=resolved_episode_hours,\n    )\n    selected_evidence = tuple(\n''',
    '''    selection = evaluate_causal_alpha_selection(\n        train_symbols=symbols,\n        samples=samples,\n        partitions=partitions,\n        candidates=candidate_values,\n        environment_factories=environment_factories,\n        episode_hours=resolved_episode_hours,\n    )\n    selection_path = Path(selection_evidence_path)\n    atomic_write_bytes(\n        selection_path,\n        canonical_json_bytes(selection.to_payload()) + b"\\n",\n    )\n    selected_evidence = tuple(\n''',
)

u5 = "trade_rl/workflows/universal_stage_a_training.py"
replace_once(
    u5,
    '''            fold_train_range=fold_train_range,\n            feature_schema_digest=feature_schema_digest,\n        )\n''',
    '''            fold_train_range=fold_train_range,\n            feature_schema_digest=feature_schema_digest,\n            selection_evidence_path=(\n                Path(output_root)\n                / "_shared-causal-teacher"\n                / "causal-teacher-selection.json"\n            ),\n        )\n''',
)

u6 = "trade_rl/workflows/universal_full_research_training.py"
replace_once(
    u6,
    '''            fold_train_range=fold_train_range,\n            feature_schema_digest=feature_schema_digest,\n        )\n''',
    '''            fold_train_range=fold_train_range,\n            feature_schema_digest=feature_schema_digest,\n            selection_evidence_path=(\n                Path(output_root)\n                / "_shared-causal-teacher"\n                / "causal-teacher-selection.json"\n            ),\n        )\n''',
)

runner = "trade_rl/workflows/universal_training_runner.py"
replace_once(
    runner,
    '''    causal_teacher_package: UniversalCausalAlphaTeacherPackage | None = None,\n    verbose: int = 0,\n''',
    '''    causal_teacher_package: UniversalCausalAlphaTeacherPackage | None = None,\n    causal_teacher_evidence_root: Path | None = None,\n    verbose: int = 0,\n''',
)
replace_once(
    runner,
    '''        if resolved_causal_package is None:\n            resolved_causal_package = build_universal_causal_alpha_teacher_package(\n''',
    '''        if resolved_causal_package is None:\n            if causal_teacher_evidence_root is None:\n                raise ValueError(\n                    "Universal causal teacher auto-build requires causal_teacher_evidence_root"\n                )\n            resolved_causal_package = build_universal_causal_alpha_teacher_package(\n''',
)
replace_once(
    runner,
    '''                fold_train_range=fold_train_range,\n                feature_schema_digest=feature_schema_digest,\n            )\n        batches = dict(resolved_causal_package.batches)\n''',
    '''                fold_train_range=fold_train_range,\n                feature_schema_digest=feature_schema_digest,\n                selection_evidence_path=(\n                    Path(causal_teacher_evidence_root)\n                    / "causal-teacher-selection.json"\n                ),\n            )\n        batches = dict(resolved_causal_package.batches)\n''',
)

shared_test = "tests/workflows/test_universal_causal_alpha_shared_training.py"
replace_once(
    shared_test,
    '''    assert len(package_calls) == 1\n    assert len(assembly_packages) == len(tuple(UniversalArchitectureName))\n''',
    '''    assert len(package_calls) == 1\n    assert package_calls[0]["selection_evidence_path"] == (\n        tmp_path / "_shared-causal-teacher" / "causal-teacher-selection.json"\n    )\n    assert len(assembly_packages) == len(tuple(UniversalArchitectureName))\n''',
)
replace_once(
    shared_test,
    '''    assert len(package_calls) == 1\n    assert len(assembly_packages) == len(tuple(FullResearchAlgorithm))\n''',
    '''    assert len(package_calls) == 1\n    assert package_calls[0]["selection_evidence_path"] == (\n        tmp_path / "_shared-causal-teacher" / "causal-teacher-selection.json"\n    )\n    assert len(assembly_packages) == len(tuple(FullResearchAlgorithm))\n''',
)

assembly_test = "tests/workflows/test_universal_sb3_training_assembly.py"
path = Path(assembly_test)
text = path.read_text(encoding="utf-8")
append = '''\n\ndef test_causal_teacher_auto_build_requires_evidence_root() -> None:\n    from trade_rl.workflows.universal_training_runner import (\n        assemble_universal_sb3_training_backend,\n    )\n\n    with pytest.raises(ValueError, match="causal_teacher_evidence_root"):\n        assemble_universal_sb3_training_backend(\n            routed_environment_factory=_routed_factory(),\n            training=_training(behavior_cloning_teacher="causal_alpha_ridge"),\n            fold_train_range=(5, 30),\n            normalizer_digest=_digest("normalizer"),\n            feature_schema_digest=_digest("features"),\n        )\n'''
if "test_causal_teacher_auto_build_requires_evidence_root" in text:
    raise SystemExit("assembly evidence-root test already exists")
path.write_text(text + append, encoding="utf-8")
