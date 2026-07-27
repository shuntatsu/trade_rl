from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_comparison_workflow_uses_exact_refs_and_repeated_samples() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "gpu-performance-comparison.yml"
    ).read_text(encoding="utf-8")

    assert "1f597caf85fe5200fe7abc34461236b65ebb8b1d" in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu, nvidia]" in workflow
    assert "persist-credentials: false" in workflow
    assert "baseline_ref" in workflow
    assert "repeats" in workflow
    assert "--runtime-profile accelerated" in workflow
    assert "compare_gpu_training_smoke.py" in workflow
    assert "gpu-performance-comparison.json" in workflow
    assert "minimum" not in workflow.lower()


def test_gpu_smoke_schema_records_runtime_identity() -> None:
    smoke = (
        ROOT / "examples" / "binance-multitimeframe" / "run_gpu_training_smoke.py"
    ).read_text(encoding="utf-8")

    assert "gpu_sequence_target_oracle_bc_training_smoke_v7" in smoke
    assert '"git_commit": config.git_commit' in smoke
    assert '"runtime_profile": runtime_profile' in smoke
