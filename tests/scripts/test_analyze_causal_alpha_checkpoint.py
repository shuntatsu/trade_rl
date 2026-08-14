from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import analyze_causal_alpha_checkpoint as module
from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateEpisodeMetricsV2,
)


def _checkpoint(path: Path) -> None:
    signal = evaluate_causal_alpha_signal_diagnostics(
        np.asarray((-0.02, -0.01, 0.01, 0.02), dtype=np.float64),
        np.asarray((-0.01, -0.02, 0.01, 0.03), dtype=np.float64),
    )
    metric = CausalAlphaCandidateEpisodeMetricsV2(
        candidate_digest="a" * 64,
        symbol="BTCUSDT",
        episode_index=0,
        gross_return=-0.01,
        net_return=-0.02,
        turnover_per_day=0.1,
        total_execution_cost=10.0,
        trade_count=2,
        signal_24h=signal,
        signal_72h=signal,
        cost_suppressed_change_count=0,
        submitted_change_count=1,
        strong_reversal_count=0,
        command_sign_flip_count=0,
        execution_rejection_count=0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        hard_risk_violation=False,
        liquidity_deleveraging_count=0,
        liquidity_weight_cap_min=0.1,
        liquidity_weight_cap_median=0.2,
        liquidity_weight_cap_max=0.3,
    )
    payload = {
        **metric.to_payload(),
        "generator_code_digest": "1" * 64,
        "grid_digest": "2" * 64,
        "schema_version": "causal_alpha_selection_checkpoint_metric_v2",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_cli_emits_non_promotable_diagnostic_json(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    _checkpoint(checkpoint)

    exit_code = module.main([str(checkpoint)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["promotion_eligible"] is False
    assert payload["row_count"] == 1
    assert payload["unique_prediction_episode_count"] == 1
    assert payload["generator_code_digest"] == "1" * 64
