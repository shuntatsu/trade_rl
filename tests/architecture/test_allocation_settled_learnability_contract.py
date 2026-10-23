"""The new lane remains bounded software evidence with a real-backend CI route."""

from pathlib import Path

from tests.integrations.test_allocation_settled_learnability import (
    CASES,
    PROTOCOL,
    SLOTS,
)
from trade_rl.strategies.rl.allocation_training_protocol import (
    AllocationPPOTrainingProtocol,
)


def test_frozen_budget_roster_and_cpu_protocol_remain_explicit():
    assert SLOTS == [
        (0, "signal"),
        (0, "fee_only"),
        (7, "signal"),
        (7, "fee_only"),
        (17, "signal"),
        (17, "fee_only"),
    ]
    declaration = AllocationPPOTrainingProtocol(**PROTOCOL)
    assert (declaration.n_steps, declaration.batch_size, declaration.n_epochs) == (
        32,
        32,
        10,
    )
    assert (declaration.gamma, declaration.gae_lambda) == (1.0, 0.95)
    assert declaration.learning_rate == 0.002
    assert declaration.pi_layers == declaration.vf_layers == (32, 32)
    assert CASES == {
        "signal": {"price_response": 0.02, "fee": 0.0005},
        "fee_only": {"price_response": 0.0, "fee": 0.005},
    }
    assert declaration.payload()["backend"]["device"] == "cpu"


def test_actual_backend_job_selects_and_retains_complete_learning_and_failure_evidence():
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/ci.yml").read_text("utf-8")
    runtime = workflow.split("  ppo-runtime:", 1)[1].split("  guide:", 1)[0]
    assert "uv sync --extra dev --extra train-sb3" in runtime
    assert "tests/integrations/test_allocation_settled_learnability.py" in runtime
    assert "**/fixed-settled-learnability/**" in runtime
    evidence = runtime.split("Preserve settled learnability evidence", 1)[1]
    assert "if: always()" in evidence and "overwrite: false" in evidence
