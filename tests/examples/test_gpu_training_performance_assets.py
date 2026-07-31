from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gpu_workflows_use_the_maintained_evidence_validator() -> None:
    workflow_root = ROOT / ".github" / "workflows"
    workflows = (
        workflow_root / "gpu-nightly.yml",
        workflow_root / "finalize-pr227-gpu-verification.yml",
    )

    for path in workflows:
        workflow = path.read_text(encoding="utf-8")
        assert "trade_rl.operations.gpu_training_smoke" in workflow
        assert "gpu_sequence_target_oracle_bc_training_smoke_v7" not in workflow
        assert "gpu_sequence_target_oracle_bc_training_smoke_v8" not in workflow


def test_gpu_training_example_is_a_thin_operations_wrapper() -> None:
    smoke = (
        ROOT / "examples" / "binance-multitimeframe" / "run_gpu_training_smoke.py"
    ).read_text(encoding="utf-8")

    assert "from trade_rl.operations.gpu_training_smoke import" in smoke
    assert "def run_gpu_training_smoke(" not in smoke
    assert "gpu_sequence_target_oracle_bc_training_smoke_v7" not in smoke
    assert "gpu_sequence_target_oracle_bc_training_smoke_v8" not in smoke
