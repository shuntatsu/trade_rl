from __future__ import annotations

from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate
from trade_rl.workflows.universal_causal_alpha_v10_gates import (
    V8_CANDIDATE_BY_V10,
    V10_CANDIDATE_BY_V8,
)


def test_v10_gate_mapping_is_complete_unique_and_hierarchy_last() -> None:
    assert tuple(V8_CANDIDATE_BY_V10) == tuple(CausalAlphaV10Candidate)
    assert set(V8_CANDIDATE_BY_V10.values()) == set(CausalAlphaV8Candidate)
    assert V8_CANDIDATE_BY_V10[CausalAlphaV10Candidate.HIERARCHICAL_WAVE] is (
        CausalAlphaV8Candidate.ROBUST_CALIBRATED
    )
    assert V10_CANDIDATE_BY_V8 == {
        value: key for key, value in V8_CANDIDATE_BY_V10.items()
    }

