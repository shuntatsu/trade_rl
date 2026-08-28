from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v6 import (
    CAUSAL_ALPHA_V6_TARGET_REASONS,
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v10 import (
    CAUSAL_ALPHA_V10_TARGET_SCHEMA,
    CausalAlphaV10Candidate,
    CausalAlphaV10TargetPath,
)


def _v6_path() -> CausalAlphaV6TargetPath:
    return CausalAlphaV6TargetPath(
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        initial_weight=0.0,
        decision_indices=np.asarray([0], dtype=np.int64),
        targets=np.asarray([0.0]),
        fast_proposals=np.asarray([0.0]),
        expected_returns_4h=np.asarray([0.0]),
        expected_returns_24h=np.asarray([0.0]),
        expected_returns_72h=np.asarray([0.0]),
        direction_scores_4h=np.asarray([0.0]),
        uncertainties_4h=np.asarray([0.0]),
        one_way_cost_rates=np.asarray([0.0]),
        liquidity_weight_caps=np.asarray([0.1]),
        risk_weight_caps=np.asarray([0.1]),
        objectives=np.asarray([0.0]),
        confirmation_counts=np.asarray([0], dtype=np.int64),
        actionable_mask=np.asarray([True]),
        slow_states=(CausalAlphaV6SlowState.FLAT,),
        reasons=("hold_flat",),
        reason_counts=(("hold_flat", 1),),
        submitted_change_count=0,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        risk_projection_count=0,
        forecast_digest="a" * 64,
        config_digest="b" * 64,
    )


def _base_kwargs(candidate: CausalAlphaV10Candidate) -> dict[str, object]:
    return {
        "candidate": candidate,
        "v6_target_path": _v6_path(),
        "source_forecast_digest": "c" * 64,
        "fast_fit_digest": "d" * 64,
        "slow_fit_digest": "e" * 64,
        "v10_config_digest": "f" * 64,
    }


def test_v10_target_artifact_declares_closed_loop_trace_fields() -> None:
    fields = set(CausalAlphaV10TargetPath.__dataclass_fields__)
    assert {
        "hierarchy_input_digest",
        "hierarchy_reasons",
        "hierarchy_reason_counts",
    } <= fields


def test_v10_target_schema_is_bumped_for_closed_loop_artifacts() -> None:
    assert CAUSAL_ALPHA_V10_TARGET_SCHEMA == "causal_alpha_v10_target_v2"


def test_v10_hierarchical_target_requires_closed_loop_trace() -> None:
    with pytest.raises(ValueError, match="hierarchy"):
        CausalAlphaV10TargetPath(
            **_base_kwargs(CausalAlphaV10Candidate.HIERARCHICAL_WAVE)
        )


def test_v10_hierarchical_target_accepts_bound_trace() -> None:
    target = CausalAlphaV10TargetPath(
        **_base_kwargs(CausalAlphaV10Candidate.HIERARCHICAL_WAVE),
        hierarchy_input_digest="1" * 64,
        hierarchy_reasons=("entry_floor_hold",),
        hierarchy_reason_counts=(("entry_floor_hold", 1),),
    )

    assert target.hierarchy_input_digest == "1" * 64
    assert target.hierarchy_reasons == ("entry_floor_hold",)
    assert target.to_payload()["hierarchy_input_digest"] == "1" * 64


def test_v10_control_target_rejects_hierarchy_trace() -> None:
    with pytest.raises(ValueError, match="control"):
        CausalAlphaV10TargetPath(
            **_base_kwargs(CausalAlphaV10Candidate.V8_ROBUST_CONTROL),
            hierarchy_input_digest="1" * 64,
            hierarchy_reasons=("entry_floor_hold",),
            hierarchy_reason_counts=(("entry_floor_hold", 1),),
        )


def test_v10_specific_execution_reason_is_not_generic_v6_vocabulary() -> None:
    assert "execution_contract_hold" not in CAUSAL_ALPHA_V6_TARGET_REASONS
