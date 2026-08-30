from __future__ import annotations

import numpy as np

from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config
from trade_rl.learning.causal_alpha_v9_wave import causal_alpha_v9_wave_target_path
from trade_rl.learning.causal_alpha_v11 import (
    CausalAlphaV11Candidate,
    CausalAlphaV11Config,
    CausalAlphaV11StudyArm,
)
from trade_rl.learning.causal_alpha_v11_policy import (
    CausalAlphaV11TracePolicy,
    compile_causal_alpha_v11_target,
)


def _inputs(heads: np.ndarray) -> dict[str, object]:
    rows = heads.shape[1]
    return {
        "decision_indices": np.arange(rows, dtype=np.int64),
        "head_predictions": heads,
        "one_way_cost_rates": np.full(rows, 0.0007),
        "liquidity_weight_caps": np.full(rows, 0.1),
        "risk_weight_caps": np.full(rows, 0.1),
        "actionable_mask": np.ones(rows, dtype=np.bool_),
        "source_forecast_digest": "a" * 64,
        "wave_fit_digest": "b" * 64,
        "v9_config": CausalAlphaV9Config(),
        "v11_config": CausalAlphaV11Config(),
        "initial_weight": 0.0,
    }


def _heads(*, rows: int = 81, signals: dict[int, float]) -> np.ndarray:
    result = np.zeros((3, rows), dtype=np.float64)
    for index, value in signals.items():
        result[:, index] = value
    return result


def test_v11_control_actions_and_reasons_equal_v9_exactly() -> None:
    inputs = _inputs(_heads(signals={0: 0.01, 16: 0.01, 48: -0.01, 64: -0.01}))
    v9 = causal_alpha_v9_wave_target_path(
        **{
            name: value
            for name, value in inputs.items()
            if name not in {"wave_fit_digest", "v9_config", "v11_config"}
        },
        config=inputs["v9_config"],
    )
    v11 = compile_causal_alpha_v11_target(study_arm=None, **inputs)

    assert v11.target.candidate is CausalAlphaV11Candidate.V9_CONTROL
    np.testing.assert_array_equal(v11.target.v6_target_path.targets, v9.targets)
    assert v11.target.v6_target_path.reasons == v9.reasons


def test_v11_l1_exits_after_two_actionable_neutral_cadences() -> None:
    inputs = _inputs(_heads(signals={0: 0.01, 16: 0.01}))

    compiled = compile_causal_alpha_v11_target(
        study_arm=CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2,
        **inputs,
    )

    assert compiled.target.v6_target_path.targets[16] == 0.1
    assert compiled.target.v6_target_path.targets[32] == 0.1
    assert compiled.target.v6_target_path.targets[48] == 0.0
    assert compiled.target.v6_target_path.reasons[48] == "exit"
    assert compiled.policy_reasons[48] == "neutral_expiry_2"


def test_v11_l1_unactionable_neutral_does_not_advance_expiry() -> None:
    inputs = _inputs(_heads(signals={0: 0.01, 16: 0.01}))
    actionable = np.asarray(inputs["actionable_mask"]).copy()
    actionable[32] = False
    inputs["actionable_mask"] = actionable

    compiled = compile_causal_alpha_v11_target(
        study_arm=CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2,
        **inputs,
    )

    assert compiled.target.v6_target_path.targets[48] == 0.1
    assert compiled.target.v6_target_path.targets[64] == 0.0


def test_v11_e1_filters_entry_below_round_trip_cost() -> None:
    inputs = _inputs(_heads(signals={0: 0.0012, 16: 0.0012}))

    compiled = compile_causal_alpha_v11_target(
        study_arm=CausalAlphaV11StudyArm.AFTER_COST_ENTRY,
        **inputs,
    )

    assert np.all(compiled.target.v6_target_path.targets == 0.0)
    assert compiled.after_cost_entry_objectives[0] < 0.0


def test_v11_trace_policy_emits_aligned_metadata_without_changing_action() -> None:
    compiled = compile_causal_alpha_v11_target(
        study_arm=None,
        **_inputs(_heads(rows=17, signals={0: 0.01, 16: 0.01})),
    )
    policy = CausalAlphaV11TracePolicy(compiled)

    action, state = policy.predict({"current_weights": np.asarray([0.0])})

    assert state is None
    assert action.shape == (1,)
    assert float(action[0]) == 0.0
    assert policy.last_step_trace_metadata["fast_qualified_direction"] == 1
    assert "after_cost_entry_objective" in policy.last_step_trace_metadata
