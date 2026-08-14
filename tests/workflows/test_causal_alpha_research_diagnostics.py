from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_diagnostics import (
    evaluate_causal_alpha_signal_diagnostics,
)
from trade_rl.workflows import causal_alpha_research_diagnostics as module
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaCandidateEpisodeMetricsV2,
)


_GRID = "2" * 64
_GENERATOR = "1" * 64
_CANDIDATE_A = "a" * 64
_CANDIDATE_B = "b" * 64


def _signal(offset: float = 0.0):
    predicted = np.asarray((-0.03, -0.01, 0.01, 0.03), dtype=np.float64) + offset
    realized = np.asarray((-0.02, -0.015, 0.012, 0.025), dtype=np.float64)
    return evaluate_causal_alpha_signal_diagnostics(predicted, realized)


def _metric(
    candidate: str,
    *,
    symbol: str,
    episode: int,
    gross: float,
    net: float,
    signal_offset: float = 0.0,
) -> CausalAlphaCandidateEpisodeMetricsV2:
    signal_24h = _signal(signal_offset)
    signal_72h = _signal(signal_offset + 0.001)
    return CausalAlphaCandidateEpisodeMetricsV2(
        candidate_digest=candidate,
        symbol=symbol,
        episode_index=episode,
        gross_return=gross,
        net_return=net,
        turnover_per_day=0.1,
        total_execution_cost=10.0,
        trade_count=3,
        signal_24h=signal_24h,
        signal_72h=signal_72h,
        cost_suppressed_change_count=1,
        submitted_change_count=2,
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


def _write_checkpoint(
    path: Path,
    rows: list[tuple[CausalAlphaCandidateEpisodeMetricsV2, str, str]],
) -> None:
    payloads: list[str] = []
    for metric, grid_digest, generator_digest in rows:
        payload = {
            **metric.to_payload(),
            "generator_code_digest": generator_digest,
            "grid_digest": grid_digest,
            "schema_version": "causal_alpha_selection_checkpoint_metric_v2",
        }
        payloads.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(payloads) + "\n", encoding="utf-8")


def test_diagnostic_checkpoint_accepts_historical_generator_and_deduplicates_signals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.jsonl"
    shared_a = _metric(
        _CANDIDATE_A,
        symbol="BTCUSDT",
        episode=0,
        gross=-0.02,
        net=-0.03,
    )
    shared_b = replace(shared_a, candidate_digest=_CANDIDATE_B, digest="")
    unique = _metric(
        _CANDIDATE_A,
        symbol="BNBUSDT",
        episode=1,
        gross=0.01,
        net=-0.01,
        signal_offset=0.004,
    )
    _write_checkpoint(
        path,
        [
            (shared_a, _GRID, _GENERATOR),
            (shared_b, _GRID, _GENERATOR),
            (unique, _GRID, _GENERATOR),
        ],
    )

    snapshot = module.load_causal_alpha_diagnostic_checkpoint_v2(path)
    report = module.build_causal_alpha_research_report(snapshot)

    assert snapshot.grid_digest == _GRID
    assert snapshot.generator_code_digest == _GENERATOR
    assert snapshot.row_count == 3
    assert report.promotion_eligible is False
    assert report.unique_prediction_episode_count == 2
    assert report.duplicate_signal_row_count == 1
    assert {item.candidate_digest for item in report.candidates} == {
        _CANDIDATE_A,
        _CANDIDATE_B,
    }


def test_diagnostic_checkpoint_rejects_mixed_generator_identity(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    first = _metric(
        _CANDIDATE_A,
        symbol="BTCUSDT",
        episode=0,
        gross=0.0,
        net=0.0,
    )
    second = _metric(
        _CANDIDATE_B,
        symbol="BTCUSDT",
        episode=0,
        gross=0.0,
        net=0.0,
    )
    _write_checkpoint(
        path,
        [
            (first, _GRID, _GENERATOR),
            (second, _GRID, "3" * 64),
        ],
    )

    with pytest.raises(ValueError, match="generator identity"):
        module.load_causal_alpha_diagnostic_checkpoint_v2(path)


def test_paired_candidate_delta_uses_only_exact_common_episode_scopes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.jsonl"
    rows = [
        _metric(
            _CANDIDATE_A,
            symbol="BTCUSDT",
            episode=0,
            gross=0.10,
            net=0.08,
        ),
        _metric(
            _CANDIDATE_A,
            symbol="BTCUSDT",
            episode=1,
            gross=0.20,
            net=0.15,
            signal_offset=0.002,
        ),
        _metric(
            _CANDIDATE_B,
            symbol="BTCUSDT",
            episode=1,
            gross=0.50,
            net=0.45,
            signal_offset=0.002,
        ),
        _metric(
            _CANDIDATE_B,
            symbol="BTCUSDT",
            episode=2,
            gross=0.70,
            net=0.60,
            signal_offset=0.005,
        ),
    ]
    _write_checkpoint(path, [(row, _GRID, _GENERATOR) for row in rows])
    snapshot = module.load_causal_alpha_diagnostic_checkpoint_v2(path)

    delta = module.paired_candidate_delta(
        snapshot,
        _CANDIDATE_A,
        _CANDIDATE_B,
    )

    assert delta.common_scope_count == 1
    assert delta.left_only_scope_count == 1
    assert delta.right_only_scope_count == 1
    assert delta.mean_gross_return_delta == pytest.approx(-0.30)
    assert delta.mean_net_return_delta == pytest.approx(-0.30)
    assert delta.common_scopes == (("BTCUSDT", 1),)
