from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
)
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


def _store(root: Path) -> CausalAlphaV3ArtifactStore:
    return CausalAlphaV3ArtifactStore(
        root,
        run_manifest_digest=_RUN,
        freeze_digest=_FREEZE,
    )


def test_target_diagnostics_summarize_direction_uncertainty_and_objective_margin() -> (
    None
):
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


def test_replay_diagnostics_loader_rejects_empty_artifact_digest() -> None:
    payload = _diagnostics().to_payload()
    payload["artifact_digest"] = ""

    with pytest.raises(ValueError, match="digest"):
        CausalAlphaV3ReplayDiagnostics.from_payload(payload)


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


def test_diagnostics_store_round_trips_and_missing_leaves_are_optional(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostics()
    store = _store(tmp_path)

    assert (
        store.load_replay_diagnostics(
            expected_replay_metric_digests={diagnostics.identity: _REPLAY}
        )
        == {}
    )

    path = store.write_replay_diagnostics(diagnostics)
    assert path == (
        tmp_path / "selection" / "diagnostics" / _CANDIDATE / "BTCUSDT" / "3.json"
    )
    assert store.load_replay_diagnostics(
        expected_replay_metric_digests={diagnostics.identity: _REPLAY}
    ) == {diagnostics.identity: diagnostics}


def test_diagnostics_store_rejects_conflicts_and_replay_digest_drift(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostics()
    store = _store(tmp_path)
    store.write_replay_diagnostics(diagnostics)

    conflicting = replace(diagnostics, mean_target=0.0, digest="")
    with pytest.raises(ValueError, match="identity drifted"):
        store.write_replay_diagnostics(conflicting)

    with pytest.raises(ValueError, match="replay metric identity drifted"):
        store.load_replay_diagnostics(
            expected_replay_metric_digests={diagnostics.identity: "9" * 64}
        )


def test_diagnostics_store_rejects_run_freeze_and_path_identity_drift(
    tmp_path: Path,
) -> None:
    diagnostics = _diagnostics()
    wrong_run = CausalAlphaV3ArtifactStore(
        tmp_path / "wrong-run",
        run_manifest_digest="9" * 64,
        freeze_digest=_FREEZE,
    )
    with pytest.raises(ValueError, match="run manifest identity mismatch"):
        wrong_run.write_replay_diagnostics(diagnostics)

    wrong_freeze = CausalAlphaV3ArtifactStore(
        tmp_path / "wrong-freeze",
        run_manifest_digest=_RUN,
        freeze_digest="9" * 64,
    )
    with pytest.raises(ValueError, match="freeze identity mismatch"):
        wrong_freeze.write_replay_diagnostics(diagnostics)

    store = _store(tmp_path / "path")
    correct = store.write_replay_diagnostics(diagnostics)
    wrong = correct.with_name("4.json")
    correct.replace(wrong)
    with pytest.raises(ValueError, match="path identity drifted"):
        store.load_replay_diagnostics(
            expected_replay_metric_digests={diagnostics.identity: _REPLAY}
        )
