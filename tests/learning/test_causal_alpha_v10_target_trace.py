from __future__ import annotations

from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10TargetPath


def test_v10_target_artifact_declares_closed_loop_trace_fields() -> None:
    fields = set(CausalAlphaV10TargetPath.__dataclass_fields__)
    assert {
        "hierarchy_input_digest",
        "hierarchy_reasons",
        "hierarchy_reason_counts",
    } <= fields
