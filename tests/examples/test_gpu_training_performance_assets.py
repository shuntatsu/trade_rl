from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_nightly_requires_internal_training_performance_evidence() -> None:
    workflow = (ROOT / ".github" / "workflows" / "gpu-nightly.yml").read_text(
        encoding="utf-8"
    )
    smoke = (
        ROOT / "examples" / "binance-multitimeframe" / "run_gpu_training_smoke.py"
    ).read_text(encoding="utf-8")

    assert "gpu_sequence_target_oracle_bc_training_smoke_v6" in workflow
    assert 'evidence["performance"]["training_artifact"]' in workflow
    assert 'evidence["resume"]["performance"]["training_artifact"]' in workflow
    assert 'performance["device_type"] == "cuda"' in workflow
    assert 'performance["peak_cuda_allocated_bytes"] > 0' in workflow
    assert 'performance["peak_cuda_reserved_bytes"] > 0' in workflow
    assert 'resumed["peak_cuda_allocated_bytes"] > 0' in workflow
    assert 'resumed["peak_cuda_reserved_bytes"] > 0' in workflow
    assert '"training_artifact": training_performance' in smoke
    assert '"training_artifact": resumed_training_performance' in smoke
