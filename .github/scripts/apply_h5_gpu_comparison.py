from __future__ import annotations

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one patch seam, found {count}")
    path.write_text(source.replace(old, new), encoding="utf-8")


smoke = Path("examples/binance-multitimeframe/run_gpu_training_smoke.py")
replace_once(
    smoke,
    "def _smoke_config_payload(timesteps: int) -> dict[str, Any]:\n"
    "    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:\n"
    "        raise ValueError(\"timesteps must be a positive integer\")\n"
    "    payload = _load_template()\n",
    "def _smoke_config_payload(\n"
    "    timesteps: int, runtime_profile: str = \"compatibility\"\n"
    ") -> dict[str, Any]:\n"
    "    if isinstance(timesteps, bool) or not isinstance(timesteps, int) or timesteps <= 0:\n"
    "        raise ValueError(\"timesteps must be a positive integer\")\n"
    "    if runtime_profile not in {\"compatibility\", \"accelerated\"}:\n"
    "        raise ValueError(\"runtime_profile must be compatibility or accelerated\")\n"
    "    accelerated = runtime_profile == \"accelerated\"\n"
    "    payload = _load_template()\n",
)
replace_once(
    smoke,
    '            "sequence_dropout": 0.05,\n'
    '            "max_policy_parameters": 2_500_000,\n',
    '            "sequence_dropout": 0.05,\n'
    '            "sequence_compile": accelerated,\n'
    '            "sequence_compile_mode": "reduce-overhead",\n'
    '            "sequence_transfer_mode": (\n'
    '                "pinned_non_blocking" if accelerated else "synchronous"\n'
    '            ),\n'
    '            "vector_environment_mode": (\n'
    '                "subprocess" if accelerated else "in_process"\n'
    '            ),\n'
    '            "max_policy_parameters": 2_500_000,\n',
)
replace_once(
    smoke,
    "def build_smoke_config(timesteps: int) -> TrainingRunConfig:\n"
    "    \"\"\"Build the tiny run while retaining maintained CUDA/model dimensions.\"\"\"\n\n"
    "    return TrainingRunConfig.from_mapping(_smoke_config_payload(timesteps))\n",
    "def build_smoke_config(\n"
    "    timesteps: int, runtime_profile: str = \"compatibility\"\n"
    ") -> TrainingRunConfig:\n"
    "    \"\"\"Build the tiny run while retaining maintained CUDA/model dimensions.\"\"\"\n\n"
    "    return TrainingRunConfig.from_mapping(\n"
    "        _smoke_config_payload(timesteps, runtime_profile=runtime_profile)\n"
    "    )\n",
)
replace_once(
    smoke,
    "def run_gpu_training_smoke(*, work_root: Path, timesteps: int) -> dict[str, object]:\n",
    "def run_gpu_training_smoke(\n"
    "    *, work_root: Path, timesteps: int, runtime_profile: str = \"compatibility\"\n"
    ") -> dict[str, object]:\n",
)
replace_once(
    smoke,
    "    config_payload = _smoke_config_payload(timesteps)\n",
    "    config_payload = _smoke_config_payload(\n"
    "        timesteps, runtime_profile=runtime_profile\n"
    "    )\n",
)
replace_once(
    smoke,
    '    evidence: dict[str, object] = {\n'
    '        "actual_timesteps": int(ensemble["actual_timesteps"]),\n',
    '    evidence: dict[str, object] = {\n'
    '        "actual_timesteps": int(ensemble["actual_timesteps"]),\n'
    '        "git_commit": config.git_commit,\n'
    '        "runtime_profile": runtime_profile,\n',
)
replace_once(
    smoke,
    '        "schema": "gpu_sequence_target_oracle_bc_training_smoke_v6",\n',
    '        "schema": "gpu_sequence_target_oracle_bc_training_smoke_v7",\n',
)
replace_once(
    smoke,
    '    parser.add_argument("--timesteps", type=int, default=128)\n'
    '    return parser\n',
    '    parser.add_argument("--timesteps", type=int, default=128)\n'
    '    parser.add_argument(\n'
    '        "--runtime-profile",\n'
    '        choices=("compatibility", "accelerated"),\n'
    '        default="compatibility",\n'
    '    )\n'
    '    return parser\n',
)
replace_once(
    smoke,
    "        work_root=work_root,\n"
    "        timesteps=args.timesteps,\n"
    "    )\n",
    "        work_root=work_root,\n"
    "        timesteps=args.timesteps,\n"
    "        runtime_profile=args.runtime_profile,\n"
    "    )\n",
)

nightly = Path(".github/workflows/gpu-nightly.yml")
replace_once(
    nightly,
    '          --timesteps "$REQUESTED_TIMESTEPS"\n',
    '          --timesteps "$REQUESTED_TIMESTEPS"\n'
    '          --runtime-profile accelerated\n',
)
replace_once(
    nightly,
    '          assert evidence["schema"] == "gpu_sequence_target_oracle_bc_training_smoke_v6"\n'
    '          assert evidence["resolved_device"] == "cuda"\n',
    '          assert evidence["schema"] == "gpu_sequence_target_oracle_bc_training_smoke_v7"\n'
    '          assert evidence["git_commit"] == "${{ github.sha }}"\n'
    '          assert evidence["runtime_profile"] == "accelerated"\n'
    '          assert evidence["resolved_device"] == "cuda"\n',
)

asset_test = Path("tests/examples/test_gpu_training_performance_assets.py")
replace_once(
    asset_test,
    '    assert "gpu_sequence_target_oracle_bc_training_smoke_v6" in workflow\n',
    '    assert "gpu_sequence_target_oracle_bc_training_smoke_v7" in workflow\n'
    '    assert "--runtime-profile accelerated" in workflow\n'
    '    assert \'evidence["runtime_profile"] == "accelerated"\' in workflow\n',
)

comparison_workflow = Path(".github/workflows/gpu-performance-comparison.yml")
replace_once(
    comparison_workflow,
    '          (\n'
    '            cd candidate\n'
    '            uv run python examples/binance-multitimeframe/compare_gpu_training_smoke.py \\\n'
    '              "${baseline_args[@]/#/../}" \\\n'
    '              "${candidate_args[@]/#/../}" \\\n'
    '              --baseline-ref "$BASELINE_SHA" \\\n'
    '              --candidate-ref "$CANDIDATE_SHA" \\\n'
    '              --output ../gpu-comparison-evidence/gpu-performance-comparison.json\n'
    '          )\n',
    '          candidate/.venv/bin/python \\\n'
    '            candidate/examples/binance-multitimeframe/compare_gpu_training_smoke.py \\\n'
    '            "${baseline_args[@]}" \\\n'
    '            "${candidate_args[@]}" \\\n'
    '            --baseline-ref "$BASELINE_SHA" \\\n'
    '            --candidate-ref "$CANDIDATE_SHA" \\\n'
    '            --output gpu-comparison-evidence/gpu-performance-comparison.json\n',
)
