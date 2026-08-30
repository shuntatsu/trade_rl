from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v11 import (
    CausalAlphaV11Candidate,
    CausalAlphaV11Config,
    CausalAlphaV11StudyArm,
    evaluate_v11_sizing_feasibility,
)


def test_v11_candidates_and_study_arms_are_fixed() -> None:
    assert tuple(item.value for item in CausalAlphaV11Candidate) == (
        "v8_cash_sanity",
        "v9_control",
        "treatment",
    )
    assert tuple(item.value for item in CausalAlphaV11StudyArm) == (
        "neutral_expiry_2",
        "after_cost_entry",
        "sign_calibrated_entry",
        "calibrated_edge_sizing",
    )


def test_v11_config_is_fixed_and_digest_bound() -> None:
    config = CausalAlphaV11Config()

    assert len(config.digest) == 64
    assert config.to_payload()["neutral_expiry_count"] == 2
    with pytest.raises(ValueError, match="constants must remain fixed"):
        replace(config, calibration_hours=24)


def test_v11_sizing_preflight_rejects_subthreshold_targets() -> None:
    result = evaluate_v11_sizing_feasibility(
        targets=np.asarray([0.0, 0.04, 0.099]),
        entry_threshold=0.1,
        no_trade_band=0.05,
    )

    assert not result.executable
    assert result.generated_nonzero_count == 2
    assert result.executable_nonzero_count == 0
    assert result.rejection_reasons == ("entry_threshold",)
    assert len(result.digest) == 64


def test_v11_sizing_preflight_accepts_an_accessible_target() -> None:
    result = evaluate_v11_sizing_feasibility(
        targets=np.asarray([0.0, -0.1, 0.2]),
        entry_threshold=0.1,
        no_trade_band=0.05,
    )

    assert result.executable
    assert result.executable_nonzero_count == 2
    assert result.rejection_reasons == ()


@pytest.mark.parametrize(
    ("entry_threshold", "no_trade_band"),
    [(-0.1, 0.05), (0.1, -0.05), (0.0, 0.0), (0.05, 0.1)],
)
def test_v11_sizing_preflight_rejects_invalid_execution_contract(
    entry_threshold: float,
    no_trade_band: float,
) -> None:
    with pytest.raises(ValueError, match="execution thresholds"):
        evaluate_v11_sizing_feasibility(
            targets=np.asarray([0.1]),
            entry_threshold=entry_threshold,
            no_trade_band=no_trade_band,
        )
