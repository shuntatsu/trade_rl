from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "causal-scenario-c3-gpu.yml"
RUNBOOK = ROOT / "docs" / "operations" / "causal-scenario-c3-execution.md"


def test_c3_gpu_workflow_is_manual_evaluation_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "request_path:" in text
    assert "output_path:" in text
    assert "runs-on: [self-hosted, linux, x64, gpu, nvidia]" in text
    assert "nvidia-smi" in text
    assert "uv run trade-rl causal-scenario evaluate" in text
    assert "actions/upload-artifact@" in text
    assert "c3-execution-evidence" in text
    assert "schedule:" not in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "release approve" not in text
    assert "serving package" not in text
    assert "selection authorize" not in text


def test_c3_gpu_workflow_pins_third_party_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- uses:"):
            continue
        revision = stripped.rsplit("@", 1)[1]
        assert len(revision) == 40
        assert all(character in "0123456789abcdef" for character in revision)


def test_c3_execution_runbook_documents_authoritative_artifacts() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "trade-rl causal-scenario evaluate" in text
    assert "trade-rl causal-scenario publish" in text
    assert "trade-rl causal-scenario verify" in text
    assert "report.json" in text
    assert "gate.json" in text
    assert "report.md" in text
    assert "walk-forward-config.json" in text
    assert "execution-sensitivity.json" in text
    assert "NO-GO" in text
    assert "does not authorize production" in text
