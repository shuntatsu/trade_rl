from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_REF = "1f597caf85fe5200fe7abc34461236b65ebb8b1d"


def test_gpu_comparison_workflow_uses_exact_refs_and_repeated_samples() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "gpu-performance-comparison.yml"
    ).read_text(encoding="utf-8")

    assert BASELINE_REF in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu, nvidia]" in workflow
    assert "persist-credentials: false" in workflow
    assert f"REQUESTED_BASELINE_REF: {BASELINE_REF}" in workflow
    assert "${{ inputs.baseline_ref }}" not in workflow
    assert "--baseline-ref" in workflow
    assert "repeats" in workflow
    assert "--runtime-profile accelerated" in workflow
    assert "compare_gpu_training_smoke.py" in workflow
    assert "gpu-performance-comparison.json" in workflow
    assert "minimum" not in workflow.lower()


def test_gpu_smoke_schema_records_runtime_identity() -> None:
    facade = (ROOT / "trade_rl" / "operations" / "gpu_training_smoke.py").read_text(
        encoding="utf-8"
    )
    implementation = (
        ROOT / "trade_rl" / "operations" / "_gpu_training_smoke_impl.py"
    ).read_text(encoding="utf-8")

    assert (
        'GPU_TRAINING_SMOKE_SCHEMA = "gpu_sequence_target_oracle_bc_training_smoke_v8"'
        in facade
    )
    assert '"git_commit": config.git_commit' in implementation
    assert '"runtime_profile": runtime_profile' in implementation
