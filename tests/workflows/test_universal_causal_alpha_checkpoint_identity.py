from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateEpisodeMetricsV2,
)
from trade_rl.workflows.universal_causal_alpha_teacher import (
    load_causal_alpha_selection_checkpoint_v2,
    write_causal_alpha_selection_checkpoint_metric_v2,
)


def _metric() -> CausalAlphaCandidateEpisodeMetricsV2:
    signal = evaluate_causal_alpha_signal_diagnostics(
        np.asarray([-0.02, -0.01, 0.01, 0.02]),
        np.asarray([-0.01, -0.02, 0.02, 0.01]),
    )
    return CausalAlphaCandidateEpisodeMetricsV2(
        candidate_digest=content_digest("candidate"),
        symbol="BTCUSDT",
        episode_index=0,
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.3,
        total_execution_cost=4.0,
        trade_count=2,
        signal_24h=signal,
        signal_72h=signal,
        cost_suppressed_change_count=3,
        submitted_change_count=2,
        strong_reversal_count=1,
        command_sign_flip_count=1,
        execution_rejection_count=0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        hard_risk_violation=False,
    )


def test_v2_checkpoint_rejects_generator_code_drift(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    grid_digest = content_digest("grid")
    original_code_digest = content_digest("generator-v1")
    write_causal_alpha_selection_checkpoint_metric_v2(
        path,
        _metric(),
        grid_digest=grid_digest,
        generator_code_digest=original_code_digest,
    )

    with pytest.raises(ValueError, match="generator code digest"):
        load_causal_alpha_selection_checkpoint_v2(
            path,
            expected_grid_digest=grid_digest,
            expected_generator_code_digest=content_digest("generator-v2"),
        )
