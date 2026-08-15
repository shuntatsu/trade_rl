from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3FitConfig,
    CausalAlphaV3TargetConfig,
)
from trade_rl.learning.episode_oracle_teacher import OracleEpisodeContract
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.workflows.universal_causal_alpha_contracts import (
    CausalAlphaEpisodePartition,
    CausalAlphaSymbolSamples,
)
from trade_rl.workflows.universal_causal_alpha_v3_artifact_store import (
    CausalAlphaV3ArtifactStore,
)
from trade_rl.workflows.universal_causal_alpha_v3_config import (
    CausalAlphaV3Candidate,
    CausalAlphaV3NestedSelectionConfig,
    CausalAlphaV3ResearchConfig,
    CausalAlphaV3SelectionGate,
    CausalAlphaV3SignalGate,
)
from trade_rl.workflows.universal_causal_alpha_v3_identity import (
    CausalAlphaV3ExecutionIdentity,
)
from trade_rl.workflows.universal_causal_alpha_v3_pipeline import (
    CausalAlphaV3SignalRejected,
    run_universal_causal_alpha_v3_research_pipeline,
)
from trade_rl.workflows.universal_causal_alpha_v3_runtime import (
    CausalAlphaV3PreparedResearchData,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    evaluate_causal_alpha_v3_signal_gate_clustered,
)


def _sha(token: str) -> str:
    return token * 64


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


def _config(*, signal_contract_count: int) -> CausalAlphaV3ResearchConfig:
    return CausalAlphaV3ResearchConfig(
        nested_selection=CausalAlphaV3NestedSelectionConfig(
            signal_contract_count=signal_contract_count,
            minimum_economic_contract_count=1,
        ),
        signal_gate=CausalAlphaV3SignalGate(
            minimum_independent_episode_count=signal_contract_count,
            minimum_raw_scope_coverage=1.0,
            minimum_rank_ic_lower_ci=1.0,
            minimum_top_bottom_spread_lower_ci=-1.0,
            minimum_direction_accuracy_excess_lower_ci=-1.0,
            bootstrap_resamples=20,
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


def _contract(episode_index: int) -> OracleEpisodeContract:
    start = 5 + episode_index * 7
    return OracleEpisodeContract(
        dataset_id=_sha("d"),
        episode_index=episode_index,
        start=start,
        stop=start + 6,
        initial_state_mode="cash",
        initial_weights=np.zeros(1, dtype=np.float64),
    )


def _samples() -> CausalAlphaSymbolSamples:
    decisions = np.arange(2, 50, dtype=np.int64)
    features = np.column_stack(
        (decisions.astype(np.float64), np.ones(decisions.size, dtype=np.float64))
    )
    return CausalAlphaSymbolSamples(
        symbol="BTCUSDT",
        dataset_id=_sha("d"),
        feature_names=("signal", "descriptor"),
        feature_schema_digest=_sha("4"),
        context_digest=_sha("c"),
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


def _prepared(*, signal_contract_count: int) -> CausalAlphaV3PreparedResearchData:
    total_contract_count = signal_contract_count + 2
    contracts = tuple(_contract(index) for index in range(total_contract_count))
    partition = CausalAlphaEpisodePartition(
        contracts=contracts,
        selection_contracts=contracts[:-1],
        holdout_contract=contracts[-1],
        train_start=0,
        train_stop=contracts[-1].stop,
    )
    identity = CausalAlphaV3ExecutionIdentity(
        train_symbols=("BTCUSDT",),
        training_contract_digest=_sha("6"),
        instrument_context_schema_digest=_sha("7"),
        source_tree_digest=_sha("8"),
        shared_clock_digest=_sha("9"),
        dependency_lock_digest=_sha("a"),
        python_runtime_digest=_sha("b"),
        symbol_runtime_digests=(("BTCUSDT", _sha("c")),),
    )
    return CausalAlphaV3PreparedResearchData(
        train_symbols=("BTCUSDT",),
        partitions={"BTCUSDT": partition},
        samples={"BTCUSDT": _samples()},
        environment_factories={"BTCUSDT": lambda: SimpleNamespace(close=lambda: None)},
        episode_hours=24.0,
        execution_costs={"BTCUSDT": ExecutionCostConfig()},
        signal_delays={"BTCUSDT": 1},
        decision_bars={"BTCUSDT": 1},
        max_position_to_market_notional=0.02,
        catalog_digest=_sha("1"),
        partition_digest=_sha("2"),
        split_manifest_digest=_sha("3"),
        feature_schema_digest=_sha("4"),
        statistics_digest=_sha("5"),
        execution_identity=identity,
    )


def _signal_metric(
    *,
    run_manifest_digest: str,
    fit_config_digest: str,
    symbol: str,
    contract: OracleEpisodeContract,
) -> CausalAlphaV3SignalScopeMetric:
    return CausalAlphaV3SignalScopeMetric(
        run_manifest_digest=run_manifest_digest,
        fit_config_digest=fit_config_digest,
        symbol=symbol,
        episode_index=contract.episode_index,
        contract_start=contract.start,
        contract_stop=contract.stop,
        contract_digest=contract.digest,
        fit_digest=_sha("e"),
        forecast_digest=_sha("f"),
        sample_count=2,
        rank_correlation=0.2,
        direction_accuracy=0.6,
        top_bottom_realized_spread=0.01,
        cohort_indices=(contract.start, contract.start + 1),
    )


def _run_rejected(
    *,
    tmp_path,
    signal_contract_count: int,
    signal_scope_builder,
) -> None:
    with pytest.raises(CausalAlphaV3SignalRejected):
        run_universal_causal_alpha_v3_research_pipeline(
            config=_config(signal_contract_count=signal_contract_count),
            prepared=_prepared(signal_contract_count=signal_contract_count),
            output_root=tmp_path,
            signal_scope_builder=signal_scope_builder,
            signal_gate_evaluator=evaluate_causal_alpha_v3_signal_gate_clustered,
            selection_evaluator=lambda **kwargs: pytest.fail(
                "selection must not run after signal rejection"
            ),
            admission_evaluator=lambda **kwargs: pytest.fail(
                "admission must not run after signal rejection"
            ),
        )


def test_signal_leaf_is_reused_without_rebuilding_same_scope(tmp_path) -> None:
    build_calls: list[int] = []

    def build_once(**kwargs) -> CausalAlphaV3SignalScopeMetric:
        contract = kwargs["contract"]
        build_calls.append(contract.episode_index)
        return _signal_metric(
            run_manifest_digest=kwargs["run_manifest_digest"],
            fit_config_digest=kwargs["candidate"].fit.digest,
            symbol=kwargs["symbol"],
            contract=contract,
        )

    _run_rejected(
        tmp_path=tmp_path,
        signal_contract_count=1,
        signal_scope_builder=build_once,
    )
    assert build_calls == [0]

    _run_rejected(
        tmp_path=tmp_path,
        signal_contract_count=1,
        signal_scope_builder=lambda **kwargs: pytest.fail(
            "persisted signal leaf must be reused"
        ),
    )


def test_signal_resume_rebuilds_only_missing_leaf_after_partial_crash(
    tmp_path,
) -> None:
    first_calls: list[int] = []

    def crash_after_first(**kwargs) -> CausalAlphaV3SignalScopeMetric:
        contract = kwargs["contract"]
        first_calls.append(contract.episode_index)
        if contract.episode_index == 1:
            raise RuntimeError("simulated signal-stage crash")
        return _signal_metric(
            run_manifest_digest=kwargs["run_manifest_digest"],
            fit_config_digest=kwargs["candidate"].fit.digest,
            symbol=kwargs["symbol"],
            contract=contract,
        )

    with pytest.raises(RuntimeError, match="simulated signal-stage crash"):
        run_universal_causal_alpha_v3_research_pipeline(
            config=_config(signal_contract_count=2),
            prepared=_prepared(signal_contract_count=2),
            output_root=tmp_path,
            signal_scope_builder=crash_after_first,
            signal_gate_evaluator=evaluate_causal_alpha_v3_signal_gate_clustered,
            selection_evaluator=lambda **kwargs: pytest.fail("selection must not run"),
            admission_evaluator=lambda **kwargs: pytest.fail("admission must not run"),
        )
    assert first_calls == [0, 1]

    resumed_calls: list[int] = []

    def build_missing(**kwargs) -> CausalAlphaV3SignalScopeMetric:
        contract = kwargs["contract"]
        resumed_calls.append(contract.episode_index)
        return _signal_metric(
            run_manifest_digest=kwargs["run_manifest_digest"],
            fit_config_digest=kwargs["candidate"].fit.digest,
            symbol=kwargs["symbol"],
            contract=contract,
        )

    _run_rejected(
        tmp_path=tmp_path,
        signal_contract_count=2,
        signal_scope_builder=build_missing,
    )
    assert resumed_calls == [1]


def test_signal_resume_rejects_corrupt_leaf_before_builder_runs(tmp_path) -> None:
    def build_once(**kwargs) -> CausalAlphaV3SignalScopeMetric:
        contract = kwargs["contract"]
        return _signal_metric(
            run_manifest_digest=kwargs["run_manifest_digest"],
            fit_config_digest=kwargs["candidate"].fit.digest,
            symbol=kwargs["symbol"],
            contract=contract,
        )

    _run_rejected(
        tmp_path=tmp_path,
        signal_contract_count=1,
        signal_scope_builder=build_once,
    )
    records = tuple((tmp_path / "signal" / "records").rglob("*.json"))
    assert len(records) == 1
    raw = json.loads(records[0].read_text(encoding="utf-8"))
    raw["schema_version"] = "tampered"
    records[0].write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        _run_rejected(
            tmp_path=tmp_path,
            signal_contract_count=1,
            signal_scope_builder=lambda **kwargs: pytest.fail(
                "corrupt persisted leaf must fail before builder invocation"
            ),
        )


def test_signal_store_rejects_cross_run_leaf_write(tmp_path) -> None:
    store = CausalAlphaV3ArtifactStore(tmp_path, run_manifest_digest=_sha("1"))
    metric = _signal_metric(
        run_manifest_digest=_sha("2"),
        fit_config_digest=_candidate().fit.digest,
        symbol="BTCUSDT",
        contract=_contract(0),
    )

    with pytest.raises(ValueError, match="run manifest"):
        store.write_signal_scope_metric(metric)


def test_signal_gate_rejects_cross_run_metric_mix() -> None:
    candidate = _candidate()
    metrics = (
        _signal_metric(
            run_manifest_digest=_sha("1"),
            fit_config_digest=candidate.fit.digest,
            symbol="BTCUSDT",
            contract=_contract(0),
        ),
        _signal_metric(
            run_manifest_digest=_sha("2"),
            fit_config_digest=candidate.fit.digest,
            symbol="ETHUSDT",
            contract=_contract(0),
        ),
    )

    with pytest.raises(ValueError, match="run manifest"):
        evaluate_causal_alpha_v3_signal_gate_clustered(
            metrics,
            expected_raw_scope_count=2,
            expected_independent_episode_count=1,
            gate=_config(signal_contract_count=1).signal_gate,
        )
