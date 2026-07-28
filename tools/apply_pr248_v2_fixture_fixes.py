from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} anchor count was {count}, expected 1")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"applied: {label}")


def main() -> None:
    replace_once(
        Path("trade_rl/workflows/training_run.py"),
        '            optional={"onnx", "torchscript", "tolerance"},\n',
        '            optional={\n'
        '                "onnx",\n'
        '                "structured_torchscript",\n'
        '                "torchscript",\n'
        '                "tolerance",\n'
        '            },\n',
        label="structured export allowlist",
    )

    replace_once(
        Path("tests/e2e/test_research_to_serving_v2.py"),
        '            {\n                "training": {\n',
        '            {\n'
        '                "schema_version": "training_run_config_v2",\n'
        '                "training": {\n',
        label="research-to-serving v2 schema",
    )
    replace_once(
        Path("tests/workflows/test_explicit_sealed_ledger_mode.py"),
        '    return {\n        "training": {\n',
        '    return {\n'
        '        "schema_version": "training_run_config_v2",\n'
        '        "training": {\n',
        label="sealed-ledger candidate v2 schema",
    )
    replace_once(
        Path("tests/workflows/test_walk_forward_manifest_provenance.py"),
        '    return {\n        "training": {\n',
        '    return {\n'
        '        "schema_version": "training_run_config_v2",\n'
        '        "training": {\n',
        label="walk-forward provenance v2 schema",
    )

    for path in (
        Path("tests/examples/test_docker_training_assets.py"),
        Path("tests/examples/test_gpu_performance_comparison_assets.py"),
    ):
        replace_once(
            path,
            "gpu_sequence_target_oracle_bc_training_smoke_v7",
            "gpu_sequence_target_oracle_bc_training_smoke_v8",
            label=f"{path.name} GPU evidence schema",
        )

    market_test = Path("tests/workflows/test_market_walk_forward.py")
    replace_once(
        market_test,
        "from dataclasses import replace\n",
        "from dataclasses import asdict, replace\n",
        label="walk-forward mapping import",
    )
    mapping_helper = '''

def _training_run_mapping(run) -> dict[str, object]:
    environment = asdict(run.environment)
    execution = environment.pop("execution_cost")
    environment.pop("reward_config", None)
    environment.pop("reward", None)
    return {
        "schema_version": run.schema_version,
        "training": run.training.digest_payload(),
        "environment": environment,
        "execution": execution,
        "risk": asdict(run.risk),
        "portfolio_risk": asdict(run.portfolio_risk),
        "reward": asdict(run.reward),
        "trend": asdict(run.trend),
        "action": asdict(run.action),
        "alpha_contract": asdict(run.alpha_contract),
        "exports": {
            "onnx": run.export_onnx,
            "structured_torchscript": run.export_structured_torchscript,
            "torchscript": run.export_torchscript,
            "tolerance": run.export_tolerance,
        },
        "git_commit": run.git_commit,
        "git_dirty": run.git_dirty,
    }
'''
    replace_once(
        market_test,
        "\n\ndef test_structured_training_view_preserves_exact_sequence_and_reward_preroll() -> None:\n",
        mapping_helper
        + "\n\ndef test_structured_training_view_preserves_exact_sequence_and_reward_preroll() -> None:\n",
        label="walk-forward public config serializer",
    )
    replace_once(
        market_test,
        '                "candidates": [{"name": "sequence-ppo", "run": run.digest_payload()}],\n',
        '                "candidates": [\n'
        '                    {"name": "sequence-ppo", "run": _training_run_mapping(run)}\n'
        '                ],\n',
        label="walk-forward public config use",
    )

    replace_once(
        Path("trade_rl/rl/training.py"),
        '''        structured_values = (
            self.structured_export_manifest_path,
            self.structured_export_manifest_digest,
            self.structured_export_model_path,
            self.structured_export_model_digest,
            self.architecture_digest,
        )
        if any(value is not None for value in structured_values) and any(
            value is None for value in structured_values
        ):
            raise ValueError("structured export identity must be complete")
''',
        '''        structured_export_values = (
            self.structured_export_manifest_path,
            self.structured_export_manifest_digest,
            self.structured_export_model_path,
            self.structured_export_model_digest,
        )
        if any(value is not None for value in structured_export_values) and (
            any(value is None for value in structured_export_values)
            or self.architecture_digest is None
        ):
            raise ValueError("structured export identity must be complete")
''',
        label="architecture identity independent from export files",
    )
    replace_once(
        Path("tests/rl/test_policy_training_result_structured_export.py"),
        '''def test_policy_training_result_accepts_complete_structured_export_identity() -> None:
''',
        '''def test_policy_training_result_accepts_architecture_identity_without_export() -> None:
    result = _result(architecture_digest="e" * 64)

    assert result.architecture_digest == "e" * 64
    assert result.structured_export_manifest_path is None


def test_policy_training_result_accepts_complete_structured_export_identity() -> None:
''',
        label="architecture-only result regression test",
    )


if __name__ == "__main__":
    main()
