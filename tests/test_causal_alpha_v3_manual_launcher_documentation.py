from __future__ import annotations

from tests.architecture.repository_paths import REPOSITORY_ROOT


def test_universal_training_documents_causal_alpha_v3_manual_control_contract() -> None:
    document = (REPOSITORY_ROOT / "docs/UNIVERSAL_TRAINING.md").read_text(
        encoding="utf-8"
    )

    for phrase in (
        ".github/workflows/causal-alpha-v3-research.yml",
        "start / status / collect / stop",
        "TRADE_RL_UNIVERSAL_ARTIFACT_ROOT",
        "TRADE_RL_CAUSAL_ALPHA_V3_STATE_ROOT",
        "0 = admitted",
        "2 = signal_rejected",
        "3 = selection_rejected",
        "4 = admission_rejected",
        "operator_stopped",
        "source run",
        "self-hosted GPU",
    ):
        assert phrase in document

    assert "launcher success does not prove profitability or Production GO" in document
    assert "collect does not delete the durable source run" in document
