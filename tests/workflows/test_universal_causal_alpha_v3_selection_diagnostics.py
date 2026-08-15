from __future__ import annotations

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v3_diagnostics import (
    CausalAlphaV3ReplayDiagnostics,
    summarize_causal_alpha_v3_targets,
)

_RUN = "1" * 64
_FREEZE = "2" * 64
_CANDIDATE = "3" * 64
_CONTRACT = "4" * 64
_REPLAY = "5" * 64
_FIT = "6" * 64
_FORECAST = "7" * 64
_TARGET = "8" * 64


def _diagnostics() -> CausalAlphaV3ReplayDiagnostics:
    return summarize_causal_alpha_v3_targets(
        run_manifest_digest=_RUN,
        freeze_digest=_FREEZE,
        candidate_digest=_CANDIDATE,
        symbol="BTCUSDT",
        episode_index=3,
        contract_digest=_CONTRACT,
        replay_metric_digest=_REPLAY,
        fit_digest=_FIT,
        forecast_digest=_FORECAST,
        target_path_digest=_TARGET,
        targets=np.asarray((-0.2, 0.0, 0.1, 0.0), dtype=np.float64),
        expected_returns=np.asarray((-0.03, -0.01, 0.02, 0.0), dtype=np.float64),
        uncertainties=np.asarray((0.02, 0.01, 0.04, 0.0), dtype=np.float64),
        liquidity_weight_caps=np.asarray((0.25, 0.25, 0.2, 0.2), dtype=np.float64),
        chosen_objectives=np.asarray((0.01, 0.0, 0.005, 0.0), dtype=np.float64),
        stay_objectives=np.asarray((0.0, 0.0, 0.001, 0.0), dtype=np.float64),
        reasons=("rebalance", "hold", "rebalance", "hold"),
    )


def test_target_diagnostics_summarize_direction_uncertainty_and_objective_margin() -> None:
    diagnostics = _diagnostics()

    assert diagnostics.decision_count == 4
    assert diagnostics.long_target_count == 1
    assert diagnostics.short_target_count == 1
    assert diagnostics.flat_target_count == 2
    assert diagnostics.positive_forecast_count == 1
    assert diagnostics.negative_forecast_count == 2
    assert diagnostics.near_zero_forecast_count == 1
    assert diagnostics.mean_target == pytest.approx(-0.025)
    assert diagnostics.mean_absolute_target == pytest.approx(0.075)
    assert diagnostics.maximum_absolute_target == pytest.approx(0.2)
    assert diagnostics.mean_expected_return == pytest.approx(-0.005)
    assert diagnostics.mean_uncertainty == pytest.approx(0.0175)
    assert diagnostics.p90_uncertainty == pytest.approx(0.034)
    assert diagnostics.mean_objective_improvement == pytest.approx(0.0035)
    assert diagnostics.target_reason_counts == (("hold", 2), ("rebalance", 2))
    assert diagnostics.research_only is True
    assert diagnostics.promotion_eligible is False


def test_replay_diagnostics_payload_is_strict_and_digest_bound() -> None:
    diagnostics = _diagnostics()
    payload = diagnostics.to_payload()

    assert CausalAlphaV3ReplayDiagnostics.from_payload(payload) == diagnostics

    extra = dict(payload)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        CausalAlphaV3ReplayDiagnostics.from_payload(extra)

    tampered = dict(payload)
    tampered["mean_target"] = 0.5
    with pytest.raises(ValueError, match="digest"):
        CausalAlphaV3ReplayDiagnostics.from_payload(tampered)


def test_target_diagnostics_fail_closed_on_shape_or_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="align"):
        summarize_causal_alpha_v3_targets(
            run_manifest_digest=_RUN,
            freeze_digest=_FREEZE,
            candidate_digest=_CANDIDATE,
            symbol="BTCUSDT",
            episode_index=3,
            contract_digest=_CONTRACT,
            replay_metric_digest=_REPLAY,
            fit_digest=_FIT,
            forecast_digest=_FORECAST,
            target_path_digest=_TARGET,
            targets=np.asarray((0.0, 0.1)),
            expected_returns=np.asarray((0.1,)),
            uncertainties=np.asarray((0.1, 0.1)),
            liquidity_weight_caps=np.asarray((0.2, 0.2)),
            chosen_objectives=np.asarray((0.0, 0.0)),
            stay_objectives=np.asarray((0.0, 0.0)),
            reasons=("hold", "rebalance"),
        )

    with pytest.raises(ValueError, match="finite"):
        summarize_causal_alpha_v3_targets(
            run_manifest_digest=_RUN,
            freeze_digest=_FREEZE,
            candidate_digest=_CANDIDATE,
            symbol="BTCUSDT",
            episode_index=3,
            contract_digest=_CONTRACT,
            replay_metric_digest=_REPLAY,
            fit_digest=_FIT,
            forecast_digest=_FORECAST,
            target_path_digest=_TARGET,
            targets=np.asarray((0.0, np.nan)),
            expected_returns=np.asarray((0.1, 0.1)),
            uncertainties=np.asarray((0.1, 0.1)),
            liquidity_weight_caps=np.asarray((0.2, 0.2)),
            chosen_objectives=np.asarray((0.0, 0.0)),
            stay_objectives=np.asarray((0.0, 0.0)),
            reasons=("hold", "rebalance"),
        )
