from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
REUSABLE = WORKFLOWS / "reusable-gpu-training-verification.yml"
MANUAL = WORKFLOWS / "gpu-nightly.yml"
MAIN = WORKFLOWS / "main-gpu-verification.yml"
LEGACY = WORKFLOWS / "finalize-pr227-gpu-verification.yml"


def test_gpu_verification_has_one_reusable_implementation() -> None:
    assert REUSABLE.is_file()
    assert MANUAL.is_file()
    assert MAIN.is_file()
    assert not LEGACY.exists()

    reusable = REUSABLE.read_text(encoding="utf-8")
    assert "workflow_call:" in reusable
    for input_name in (
        "timesteps:",
        "runtime_profile:",
        "use_docker:",
        "artifact_name:",
    ):
        assert input_name in reusable
    assert "runs-on: [self-hosted, linux, x64, gpu, nvidia]" in reusable
    assert "environment: gpu-full-training" in reusable
    assert "github.actor == github.repository_owner" in reusable
    assert "github.ref == 'refs/heads/main'" in reusable
    assert "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5" in reusable
    assert "persist-credentials: false" in reusable
    assert "trade_rl.operations.gpu_training_smoke" in reusable
    assert (
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in reusable
    )


def test_gpu_workflow_callers_are_thin_and_schema_free() -> None:
    reusable_path = "uses: ./.github/workflows/reusable-gpu-training-verification.yml"
    for path in (MANUAL, MAIN):
        content = path.read_text(encoding="utf-8")
        assert reusable_path in content
        assert "runs-on:" not in content
        assert "actions/checkout@" not in content
        assert "trade_rl.operations.gpu_training_smoke" not in content
        assert "gpu_sequence_target_oracle_bc_training_smoke_v7" not in content
        assert "gpu_sequence_target_oracle_bc_training_smoke_v8" not in content

    main = MAIN.read_text(encoding="utf-8")
    assert "push:" in main
    assert "- main" in main
    assert "use_docker: true" in main

    manual = MANUAL.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in manual
    assert "use_docker: false" in manual
