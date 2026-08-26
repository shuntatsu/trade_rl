from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from trade_rl.learning.causal_alpha_v7 import (
    CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES,
    CausalAlphaV7CalibrationConfig,
    CausalAlphaV7Candidate,
)
from trade_rl.workflows import universal_causal_alpha_v7_stage_entry as stage_entry
from trade_rl.workflows.universal_causal_alpha_v5_calibration import (
    CausalAlphaV5CalibrationSplit,
)


def _digest(char: str) -> str:
    return char * 64


def test_v7_range_binds_exact_v5_split() -> None:
    split = CausalAlphaV5CalibrationSplit(
        train_symbols=("A", "B"),
        calibration_start=500,
        train_stop=1_000,
        block_boundaries=(500, 625, 750, 875, 1_000),
    )

    resolved = stage_entry._calibration_range(split)

    assert resolved.base_fit_cutoff == split.calibration_start
    assert resolved.calibration_start == split.calibration_start
    assert resolved.train_stop == split.train_stop
    assert resolved.block_boundaries == split.block_boundaries
    assert resolved.split_digest == split.digest
    assert resolved.feature_names == CAUSAL_ALPHA_V7_CALIBRATION_FEATURE_NAMES


def test_v7_signal_metrics_bind_all_three_candidates_to_one_calibration() -> None:
    calibration = SimpleNamespace(
        digest=_digest("a"),
        return_model=SimpleNamespace(digest=_digest("b")),
        direction_model=SimpleNamespace(digest=_digest("c")),
        positive_direction_support=111,
        negative_direction_support=99,
    )
    forecast = SimpleNamespace(digest=_digest("d"))
    targets = {
        candidate: SimpleNamespace(
            digest=_digest(str(index + 1)),
            v6_target_path=SimpleNamespace(
                targets=np.asarray((0.0, 0.2, 0.2, -0.1)),
                initial_weight=0.0,
                actionable_mask=np.asarray((True, True, True, True)),
                submitted_change_count=2,
                sign_flip_count=1,
            ),
        )
        for index, candidate in enumerate(CausalAlphaV7Candidate)
    }
    prepared = SimpleNamespace(run_manifest_digest=_digest("e"))
    contract = SimpleNamespace(
        episode_index=3,
        start=1_000,
        stop=1_005,
        digest=_digest("f"),
    )

    metrics = stage_entry._v7_signal_scope_metrics(
        prepared=prepared,
        symbol="BTCUSDT",
        contract=contract,
        forecast=forecast,
        calibration_fit=calibration,
        targets=targets,
        v7_config_digest=_digest("0"),
    )

    assert tuple(metric.candidate for metric in metrics) == tuple(
        CausalAlphaV7Candidate
    )
    assert all(
        metric.calibration_fit_digest == calibration.digest for metric in metrics
    )
    assert all(metric.source_forecast_digest == forecast.digest for metric in metrics)
    assert all(metric.non_flat_target_count == 3 for metric in metrics)
    assert all(metric.positive_direction_support == 111 for metric in metrics)
    assert all(metric.negative_direction_support == 99 for metric in metrics)


def test_v7_config_digest_binds_calibration_and_v6_target_contracts() -> None:
    first = stage_entry.causal_alpha_v7_stage_config_digest(
        calibration=CausalAlphaV7CalibrationConfig(),
        target=stage_entry.CausalAlphaV6TargetConfig(),
    )
    second = stage_entry.causal_alpha_v7_stage_config_digest(
        calibration=CausalAlphaV7CalibrationConfig(),
        target=stage_entry.CausalAlphaV6TargetConfig(),
    )

    assert first == second
    assert len(first) == 64
