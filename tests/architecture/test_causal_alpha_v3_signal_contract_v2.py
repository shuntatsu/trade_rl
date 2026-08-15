from __future__ import annotations

import importlib

from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3SignalGate
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
)


def test_v3_runner_exposes_only_clustered_signal_gate_v2() -> None:
    runner = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_runner"
    )
    signal_contracts = importlib.import_module(
        "trade_rl.workflows.universal_causal_alpha_v3_signal"
    )

    assert (
        runner.evaluate_causal_alpha_v3_signal_gate
        is evaluate_causal_alpha_v3_signal_gate_clustered
    )
    assert not hasattr(signal_contracts, "evaluate_causal_alpha_v3_signal_gate")


def test_v3_signal_gate_public_fields_have_explicit_units() -> None:
    fields = set(CausalAlphaV3SignalGate.__dataclass_fields__)

    assert "minimum_independent_episode_count" in fields
    assert "minimum_raw_scope_coverage" in fields
    assert "minimum_scope_count" not in fields
    assert "minimum_scope_coverage" not in fields
