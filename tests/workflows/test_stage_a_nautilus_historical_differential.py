from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tests.workflows.test_stage_a_execution_store_funding import (
    _digest,
    _plan,
    _request,
    _source_paths,
)
from trade_rl.integrations.nautilus.historical_execution import (
    NautilusHistoricalExecutionResult,
)
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore
from trade_rl.workflows.stage_a_nautilus_economic_comparison import (
    StageANautilusHistoricalEconomicClosure,
    compare_stage_a_nautilus_historical_economics,
)
from trade_rl.workflows.stage_a_nautilus_historical_differential import (
    build_stage_a_nautilus_historical_differential_evidence,
)
from trade_rl.workflows.stage_a_nautilus_historical_replay import (
    StageANautilusHistoricalExecutionResult,
)


def _stored_replay(tmp_path: Path):
    plan = _plan()
    request = _request(plan)
    candidate_config_digest = plan.candidate("candidate-a").candidate_config_digest
    event_path, evidence_path, funding_path = _source_paths(
        tmp_path / "source",
        request,
        candidate_config_digest=candidate_config_digest,
    )
    store = StageAExecutionPromotionStore(tmp_path / "store")
    store.publish(
        request=request,
        candidate_config_digest=candidate_config_digest,
        actions=((0.4,),),
        observation_digests=(_digest("obs-0"), _digest("obs-1")),
        equity_curve=(1_000.0, 1_100.0),
        transition_end_indices=(request.evaluation_range.stop,),
        event_artifact_path=event_path,
        execution_evidence_path=evidence_path,
        funding_evidence_path=funding_path,
    )
    return store.load(request.digest)


def _candidate(
    *, terminal_position_lots: int
) -> StageANautilusHistoricalExecutionResult:
    return StageANautilusHistoricalExecutionResult(
        execution=NautilusHistoricalExecutionResult(
            runtime_version="1.230.0",
            fills=(),
            fee_minor=0,
            final_balance_minor=100_000_000_000,
            terminal_position_lots=terminal_position_lots,
            terminal_open_orders=0,
            position_snapshots=(),
        ),
        funding_records=(),
    )


def test_persisted_historical_differential_binds_replay_and_structural_parity(
    tmp_path: Path,
) -> None:
    stored = _stored_replay(tmp_path)

    evidence = build_stage_a_nautilus_historical_differential_evidence(
        stored,
        _candidate(terminal_position_lots=1_000),
    )

    assert evidence.replay_digest == stored.artifact.digest
    assert evidence.request_digest == stored.artifact.cell_identity.request_digest
    assert evidence.dataset_id == stored.artifact.cell_identity.dataset_id
    assert evidence.candidate_runtime_version == "1.230.0"
    assert evidence.legacy_terminal_position_lots == 1_000
    assert evidence.candidate_terminal_position_lots == 1_000
    assert evidence.terminal_position_matches is True
    assert evidence.terminal_open_orders_passed is True
    assert evidence.funding_matches is True
    assert evidence.structural_passed is True


def test_persisted_historical_differential_reports_terminal_position_mismatch(
    tmp_path: Path,
) -> None:
    stored = _stored_replay(tmp_path)

    evidence = build_stage_a_nautilus_historical_differential_evidence(
        stored,
        _candidate(terminal_position_lots=0),
    )

    assert evidence.legacy_terminal_position_lots == 1_000
    assert evidence.candidate_terminal_position_lots == 0
    assert evidence.terminal_position_matches is False
    assert evidence.structural_passed is False


def test_historical_economic_comparison_normalizes_only_execution_cost_representation(
    tmp_path: Path,
) -> None:
    stored = _stored_replay(tmp_path)
    structural = build_stage_a_nautilus_historical_differential_evidence(
        stored,
        _candidate(terminal_position_lots=1_000),
    )

    evidence = compare_stage_a_nautilus_historical_economics(
        structural=structural,
        legacy=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=99_900,
            execution_cost_minor=100,
        ),
        candidate=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=99_800,
            execution_cost_minor=200,
        ),
    )

    assert evidence.legacy_cost_neutral_equity_minor == 100_000
    assert evidence.candidate_cost_neutral_equity_minor == 100_000
    assert evidence.execution_cost_representation_delta_minor == 100
    assert evidence.normalized_equity_delta_minor == 0
    assert evidence.economic_passed is True


def test_historical_economic_comparison_cannot_hide_structural_mismatch(
    tmp_path: Path,
) -> None:
    stored = _stored_replay(tmp_path)
    structural = build_stage_a_nautilus_historical_differential_evidence(
        stored,
        _candidate(terminal_position_lots=0),
    )

    evidence = compare_stage_a_nautilus_historical_economics(
        structural=structural,
        legacy=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=99_900,
            execution_cost_minor=100,
        ),
        candidate=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=99_800,
            execution_cost_minor=200,
        ),
    )

    assert evidence.normalized_equity_delta_minor == 0
    assert evidence.economic_passed is False


def test_historical_economic_comparison_cannot_hide_funding_mismatch(
    tmp_path: Path,
) -> None:
    stored = _stored_replay(tmp_path)
    structural = build_stage_a_nautilus_historical_differential_evidence(
        stored,
        _candidate(terminal_position_lots=1_000),
    )
    structural = replace(structural, funding_matches=False, structural_passed=False)

    evidence = compare_stage_a_nautilus_historical_economics(
        structural=structural,
        legacy=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=99_900,
            execution_cost_minor=100,
        ),
        candidate=StageANautilusHistoricalEconomicClosure(
            final_equity_minor=99_800,
            execution_cost_minor=200,
        ),
    )

    assert evidence.normalized_equity_delta_minor == 0
    assert evidence.funding_matches is False
    assert evidence.economic_passed is False
