from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3NestedSelectionConfig,
    CausalAlphaV3ResearchConfig,
    CausalAlphaV3SelectionGate,
    CausalAlphaV3SignalGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateEvidence,
    CausalAlphaV3ReplayMetric,
    CausalAlphaV3SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
)
from trade_rl.workflows.universal_causal_alpha_v3_runner import (
    CausalAlphaV3AdmissionRejected,
    CausalAlphaV3PreparedResearchData,
    CausalAlphaV3SignalRejected,
    run_universal_causal_alpha_v3_research,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3BootstrapEvidence,
    CausalAlphaV3SignalGateEvidence,
    CausalAlphaV3SignalScopeMetric,
)


def _candidate() -> CausalAlphaV3Candidate:
    return CausalAlphaV3Candidate(
        name="baseline",
        fit=CausalAlphaV3FitConfig(ridge_strength=0.1),
        target=CausalAlphaV3TargetConfig(
            target_magnitudes=(0.0, 0.05),
            uncertainty_multiplier=1.0,
            execution_cost_multiplier=1.5,
            edge_margin=0.001,
            alpha_rebalance_decisions=2,
            strong_reversal_threshold=0.02,
            max_target_delta=0.05,
        ),
    )


def _config() -> CausalAlphaV3ResearchConfig:
    return CausalAlphaV3ResearchConfig(
        nested_selection=CausalAlphaV3NestedSelectionConfig(
            signal_contract_count=1,
            minimum_economic_contract_count=1,
        ),
        signal_gate=CausalAlphaV3SignalGate(
            minimum_independent_episode_count=1,
            minimum_raw_scope_coverage=1.0,
            minimum_rank_ic_lower_ci=0.0,
            minimum_top_bottom_spread_lower_ci=0.0,
            minimum_direction_accuracy_excess_lower_ci=0.0,
            bootstrap_resamples=10,
            bootstrap_seed=0,
            bootstrap_block_size=1,
        ),
        selection_gate=CausalAlphaV3SelectionGate(
            minimum_mean_gross_return=0.0,
            minimum_mean_net_return=0.0,
            minimum_symbol_episode_net_return=-0.05,
            maximum_mean_turnover_per_day=1.0,
            maximum_unexplained_execution_rejections=0,
            minimum_positive_gross_episode_fraction=0.5,
        ),
        candidates=(_candidate(),),
    )


def _contract(episode: int, start: int) -> OracleEpisodeContract:
    return OracleEpisodeContract(
        dataset_id="d" * 64,
        episode_index=episode,
        start=start,
        stop=start + 6,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def _samples() -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 30, dtype=np.int64)
    features = np.column_stack(
        (decisions.astype(np.float64), np.ones(decisions.size, dtype=np.float64))
    )
    return CausalAlphaSymbolSamples(
        symbol="BTCUSDT",
        dataset_id="d" * 64,
        feature_names=("signal", "descriptor"),
        feature_schema_digest="4" * 64,
        context_digest="c" * 64,
        reference_equity_mode="initial_capital",
        reference_equity=1_000.0,
        decision_indices=decisions,
        features=features,
        feature_available=np.ones_like(features, dtype=np.bool_),
        labels_24h=0.001 * decisions,
        label_end_indices_24h=decisions + 1,
        labels_72h=0.002 * decisions,
        label_end_indices_72h=decisions + 2,
    )


def _environment() -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(dataset_id="d" * 64, n_symbols=1),
        decision_bars=1,
        config=SimpleNamespace(
            execution_cost=ExecutionCostConfig(),
            signal_delay_decisions=1,
        ),
        initial_weights_for_reset=lambda mode, start: np.zeros(1, dtype=np.float64),
        close=lambda: None,
    )


def _prepared() -> CausalAlphaV3PreparedResearchData:
    contracts = (_contract(0, 5), _contract(1, 12), _contract(2, 19))
    partition = CausalAlphaEpisodePartition(
        contracts=contracts,
        selection_contracts=contracts[:-1],
        holdout_contract=contracts[-1],
        train_start=0,
        train_stop=contracts[-1].stop,
    )
    identity = CausalAlphaV3ExecutionIdentity(
        train_symbols=("BTCUSDT",),
        training_contract_digest="6" * 64,
        instrument_context_schema_digest="7" * 64,
        source_tree_digest="8" * 64,
        shared_clock_digest="a" * 64,
        dependency_lock_digest="b" * 64,
        python_runtime_digest="c" * 64,
        symbol_runtime_digests=(("BTCUSDT", "9" * 64),),
    )
    return CausalAlphaV3PreparedResearchData(
        train_symbols=("BTCUSDT",),
        partitions={"BTCUSDT": partition},
        samples={"BTCUSDT": _samples()},
        environment_factories={"BTCUSDT": _environment},
        episode_hours=24.0,
        execution_costs={"BTCUSDT": ExecutionCostConfig()},
        signal_delays={"BTCUSDT": 1},
        decision_bars={"BTCUSDT": 1},
        max_position_to_market_notional=0.02,
        catalog_digest="1" * 64,
        partition_digest="2" * 64,
        split_manifest_digest="3" * 64,
        feature_schema_digest="4" * 64,
        statistics_digest="5" * 64,
        execution_identity=identity,
    )


def _signal_evidence(*, passed: bool) -> CausalAlphaV3SignalGateEvidence:
    candidate = _candidate()
    metric = CausalAlphaV3SignalScopeMetric(
        fit_config_digest=candidate.fit.digest,
        symbol="BTCUSDT",
        episode_index=0,
        contract_start=5,
        contract_stop=11,
        contract_digest="6" * 64,
        fit_digest="7" * 64,
        forecast_digest="8" * 64,
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.6,
        top_bottom_realized_spread=0.01,
        cohort_indices=(5, 8),
    )
    boot = CausalAlphaV3BootstrapEvidence(
        mean=0.1,
        p_value=0.1,
        lower_ci=0.01,
        upper_ci=0.2,
        block_size=1,
    )
    return CausalAlphaV3SignalGateEvidence(
        metrics=(metric,),
        raw_scope_count=1,
        expected_raw_scope_count=1,
        raw_scope_coverage=1.0,
        independent_episode_count=1,
        expected_independent_episode_count=1,
        rank_ic=boot,
        top_bottom_spread=boot,
        direction_accuracy_excess=boot,
        gate_digest=_config().signal_gate.digest,
        passed=passed,
        rejection_reasons=() if passed else ("rank_ic_lower_ci",),
    )


def _scope_metric(*, passed: bool, **kwargs) -> CausalAlphaV3SignalScopeMetric:
    base = _signal_evidence(passed=passed).metrics[0]
    contract = kwargs["contract"]
    return replace(
        base,
        symbol=kwargs["symbol"],
        episode_index=contract.episode_index,
        contract_start=contract.start,
        contract_stop=contract.stop,
        contract_digest=contract.digest,
        digest="",
    )


def _selection(freeze_digest: str) -> CausalAlphaV3SelectionEvidence:
    candidate = _candidate()
    metric = CausalAlphaV3ReplayMetric(
        run_manifest_digest="9" * 64,
        freeze_digest=freeze_digest,
        candidate_digest=candidate.digest,
        symbol="BTCUSDT",
        episode_index=1,
        contract_digest="a" * 64,
        fit_digest="b" * 64,
        forecast_digest="c" * 64,
        target_path_digest="e" * 64,
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.2,
        total_execution_cost=1.0,
        trade_count=2,
        submitted_change_count=1,
        sign_flip_count=0,
        liquidity_deleveraging_count=0,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        target_reason_counts=(("hold", 1),),
        hard_risk_violation=False,
    )
    candidate_evidence = CausalAlphaV3CandidateEvidence(
        candidate=candidate,
        episode_metrics=(metric,),
        lower_tail_net_return=0.01,
        mean_gross_return=0.02,
        mean_net_return=0.01,
        turnover_per_day=0.2,
        total_execution_cost=1.0,
        positive_gross_episode_fraction=1.0,
        total_trade_count=2,
        unexplained_execution_rejection_count=0,
        hard_risk_violation=False,
        admissible=True,
        rejection_reasons=(),
    )
    return CausalAlphaV3SelectionEvidence(
        candidates=(candidate_evidence,),
        selected_candidate_digest=candidate.digest,
        freeze_digest=freeze_digest,
    )


def _batch() -> EpisodeOracleBatch:
    contract = _contract(2, 19)
    return EpisodeOracleBatch(
        dataset_id=contract.dataset_id,
        teacher_config_digest="f" * 64,
        sampling_config_digest="0" * 64,
        contracts=(contract,),
        targets=(np.zeros((contract.stop - contract.start - 1, 1), dtype=np.float32),),
    )


def test_signal_rejection_stops_before_selection_and_holdout(
    monkeypatch, tmp_path
) -> None:
    import trade_rl.workflows.universal_causal_alpha_v3_runner as module

    monkeypatch.setattr(
        module,
        "build_causal_alpha_v3_signal_scope_metric",
        lambda **kwargs: _scope_metric(passed=False, **kwargs),
    )
    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_v3_signal_gate",
        lambda *args, **kwargs: _signal_evidence(passed=False),
    )
    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_v3_selection",
        lambda **kwargs: pytest.fail("selection must not run after signal rejection"),
    )
    with pytest.raises(CausalAlphaV3SignalRejected):
        run_universal_causal_alpha_v3_research(
            config=_config(), prepared=_prepared(), output_root=tmp_path
        )
    assert (tmp_path / "signal" / "rejection.json").is_file()
    assert not (tmp_path / "selection").exists()
    assert not (tmp_path / "admission").exists()


def test_admission_rejection_never_creates_teacher_package(
    monkeypatch, tmp_path
) -> None:
    import trade_rl.workflows.universal_causal_alpha_v3_runner as module

    monkeypatch.setattr(
        module,
        "build_causal_alpha_v3_signal_scope_metric",
        lambda **kwargs: _scope_metric(passed=True, **kwargs),
    )
    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_v3_signal_gate",
        lambda *args, **kwargs: _signal_evidence(passed=True),
    )
    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_v3_selection",
        lambda **kwargs: _selection(kwargs["freeze_digest"]),
    )
    monkeypatch.setattr(
        module,
        "build_causal_alpha_v3_episode_batch",
        lambda **kwargs: _batch(),
    )
    monkeypatch.setattr(
        module,
        "evaluate_causal_alpha_v3_admission",
        lambda **kwargs: SimpleNamespace(
            passed=False,
            digest=content_digest("admission-rejected"),
            to_payload=lambda: {"passed": False},
        ),
    )
    with pytest.raises(CausalAlphaV3AdmissionRejected):
        run_universal_causal_alpha_v3_research(
            config=_config(), prepared=_prepared(), output_root=tmp_path
        )
    assert (tmp_path / "freeze" / "candidates.json").is_file()
    assert (tmp_path / "selection" / "evidence.json").is_file()
    assert (tmp_path / "admission" / "evidence.json").is_file()
    assert not (tmp_path / "teacher" / "package.json").exists()
