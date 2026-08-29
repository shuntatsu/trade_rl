from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from trade_rl.learning.causal_alpha_v6 import CausalAlphaV6Candidate
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8Candidate
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Candidate
from trade_rl.workflows.universal_causal_alpha_v6_replay import (
    CausalAlphaV6ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionCell,
)
from trade_rl.workflows.universal_causal_alpha_v8_attribution import (
    CausalAlphaV8AttributionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v8_replay import (
    CausalAlphaV8ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v10_gates import (
    V8_CANDIDATE_BY_V10,
    V10_CANDIDATE_BY_V8,
    build_causal_alpha_v10_dual_run_binding,
    evaluate_causal_alpha_v10_selection,
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


def _digest(char: str) -> str:
    return char * 64


def _replay_metric(
    candidate: CausalAlphaV8Candidate, symbol: str
) -> CausalAlphaV8ReplayMetric:
    base = CausalAlphaV6ReplayMetric(
        run_manifest_digest=_digest("a"),
        v4_context_manifest_digest=_digest("b"),
        config_digest=_digest("c"),
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        symbol=symbol,
        episode_index=3,
        contract_digest=_digest("d"),
        fit_digest=_digest("e"),
        forecast_digest=_digest("f"),
        target_path_digest=_digest("1"),
        decision_count=1,
        gross_return=0.02,
        gross_wealth=math.exp(0.02),
        net_return=0.01,
        net_wealth=math.exp(0.01),
        reward_total=1.0,
        reward_scale=100.0,
        turnover_per_day=0.1,
        total_execution_cost=0.001,
        target_change_count=1,
        submitted_change_count=1,
        downstream_no_trade_suppression_count=0,
        executed_change_count=1,
        closed_trade_count=1,
        sign_flip_count=0,
        maximum_drawdown=0.0,
        actionable_coverage=1.0,
        flat_time_fraction=0.0,
        time_weighted_absolute_exposure=0.1,
        completed_holding_durations_hours=(),
        open_holding_duration_hours=1.0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        target_reason_counts=(("entry", 1),),
        hard_risk_violation=False,
        has_meaningful_execution=True,
    )
    target_digest = _digest(str(tuple(CausalAlphaV8Candidate).index(candidate) + 2))
    attribution = CausalAlphaV8AttributionEvidence(
        candidate=candidate,
        target_path_digest=target_digest,
        boundaries_digest=_digest("3"),
        step_economics_digest=_digest("4"),
        decision_count=1,
        gross_log_return=0.02,
        net_log_return=0.01,
        total_execution_cost=0.001,
        total_exposure_hours=1.0,
        cells=(
            CausalAlphaV7AttributionCell(
                dimension="exposure",
                key="long",
                support=1,
                gross_log_return=0.02,
                net_log_return=0.01,
                execution_cost=0.001,
                exposure_hours=1.0,
            ),
        ),
    )
    return CausalAlphaV8ReplayMetric(
        candidate=candidate,
        v6_metric=base,
        attribution=attribution,
        v8_target_path_digest=target_digest,
        source_forecast_digest=_digest("5"),
        calibration_fit_digest=_digest(
            str(tuple(CausalAlphaV8Candidate).index(candidate) + 6)
        ),
        v8_config_digest=_digest("7"),
    )


def test_v10_pairs_scopes_when_candidate_model_fit_identities_differ() -> None:
    symbols = tuple(f"S{index}USDT" for index in range(9))
    metrics = tuple(
        _replay_metric(candidate, symbol)
        for candidate in CausalAlphaV8Candidate
        for symbol in symbols
    )

    evidence = evaluate_causal_alpha_v10_selection(
        metrics,
        expected_symbols=symbols,
    )

    assert evidence.passed
    assert evidence.to_payload()["paired_scope_count"] == 9
    assert evidence.rejection_reasons == ()


def _dual_config(*, modes: tuple[str, ...], marker: str = "shared") -> object:
    return SimpleNamespace(
        environment=SimpleNamespace(initial_state_modes=modes),
        candidate_digest_payload=lambda: {
            "environment": {"initial_state_modes": list(modes), "marker": marker},
            "recipe": marker,
        },
    )


def _dual_prepared(*, run: str = "a") -> object:
    contract = SimpleNamespace(
        dataset_id=_digest("8"),
        episode_index=0,
        start=10,
        stop=20,
    )
    nested = SimpleNamespace(
        signal_contracts=(contract,),
        economic_contracts=(
            SimpleNamespace(
                dataset_id=_digest("8"),
                episode_index=1,
                start=21,
                stop=31,
            ),
        ),
        holdout_contract=SimpleNamespace(
            dataset_id=_digest("8"),
            episode_index=2,
            start=32,
            stop=42,
        ),
    )
    return SimpleNamespace(
        train_symbols=("BTCUSDT",),
        nested_partitions={"BTCUSDT": nested},
        nested_partition_digest=_digest("b"),
        base_runtime_manifest_digest=_digest("c"),
        v4_context_manifest_digest=_digest("d"),
        config_digest=_digest("e"),
        execution_identity_digest=_digest("f"),
        generator_code_digest=_digest("1"),
        run_manifest_digest=_digest(run),
    )


def test_v10_dual_run_binding_allows_only_initial_state_mode_delta() -> None:
    binding = build_causal_alpha_v10_dual_run_binding(
        signal_config=_dual_config(modes=("cash", "baseline")),
        selection_config=_dual_config(modes=("cash",)),
        signal_prepared=_dual_prepared(run="2"),
        selection_prepared=_dual_prepared(run="3"),
        allow_initial_state_split=True,
    )

    assert binding.signal_initial_state_modes == ("cash", "baseline")
    assert binding.selection_initial_state_modes == ("cash",)
    assert binding.signal_run_manifest_digest == _digest("2")
    assert binding.selection_run_manifest_digest == _digest("3")
    assert len(binding.digest) == 64


def test_v10_dual_run_binding_rejects_non_initial_config_drift() -> None:
    with pytest.raises(ValueError, match="outside initial_state_modes"):
        build_causal_alpha_v10_dual_run_binding(
            signal_config=_dual_config(modes=("cash", "baseline")),
            selection_config=_dual_config(
                modes=("cash",),
                marker="different",
            ),
            signal_prepared=_dual_prepared(run="2"),
            selection_prepared=_dual_prepared(run="3"),
        )

