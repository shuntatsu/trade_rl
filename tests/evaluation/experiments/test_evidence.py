from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import publish_market_dataset_artifact
from trade_rl.data.build import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.comparison.strategies import compare_strategies_by_symbol
from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    ResolvedRunConfig,
    StudyPlan,
)
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import (
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs.config import (
    CandidateRunConfig,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent


def _dataset():
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        24
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(24, dtype=np.float64)
    open_price = np.concatenate([close[:1], close[:-1]])
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full(24, 1_000_000.0),
        funding_rate=np.zeros(24),
        tradable=np.ones(24, dtype=np.bool_),
    )
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="signal", kind=FeatureKind.LOG_RETURN),),
    )
    contract = InstrumentContract(
        symbol="BTCUSDT",
        listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    return MarketDatasetBuilder(config).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (contract,),
    )


def _config() -> CandidateRunConfig:
    return CandidateRunConfig(
        signal_name="signal",
        feature_names=("signal",),
        fit_symbol_names=("BTCUSDT",),
        fit_cutoff=np.datetime64("2026-01-01T12:00:00", "ns"),
        evaluation_start=np.datetime64("2026-01-01T12:00:00", "ns"),
        evaluation_stop_exclusive=np.datetime64("2026-01-01T20:00:00", "ns"),
        rule_entry_threshold=0.10,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=32,
        ppo_seed=2,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )


def _resolved_contract(spec) -> ResolvedRunConfig:
    config = spec.config
    lean = spec.lean_config
    return ResolvedRunConfig(
        signal_name=config.signal_name,
        signal_index=lean.signal_index,
        feature_names=config.feature_names,
        feature_indices=lean.feature_indices,
        fit_symbol_names=config.fit_symbol_names,
        fit_symbol_indices=lean.fit_symbol_indices,
        fit_cutoff=str(lean.fit_cutoff),
        rule_entry_threshold=lean.rule_entry_threshold,
        rule_exit_threshold=lean.rule_exit_threshold,
        forecast_entry_threshold=lean.forecast_entry_threshold,
        forecast_exit_threshold=lean.forecast_exit_threshold,
        ppo_total_timesteps=lean.ppo_total_timesteps,
        ppo_seed=lean.ppo_seed,
        evaluation_start=str(config.evaluation_start),
        evaluation_stop_exclusive=str(config.evaluation_stop_exclusive),
        gross_budget=config.gross_budget,
        initial_capital=config.initial_capital,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )


def _plan(dataset, artifact) -> StudyPlan:
    config = _config()
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=config,
    )
    provenance = build_candidate_run_provenance()
    return StudyPlan(
        research_question="Does the candidate improve under fixed development evidence?",
        dataset_id=dataset.dataset_id,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        symbols=tuple(dataset.symbols),
        baseline_config=_resolved_contract(spec),
        ppo_seeds=(2, 5, 9),
        allowed_factors=(ControlledFactor.PPO_TRAINING_BUDGET,),
        max_experiments=3,
        n_bootstrap=100,
        bootstrap_seed=17,
        implementation_digest=str(provenance["implementation_digest"]),
        runtime_environment_digest=str(provenance["runtime_environment_digest"]),
    )


def _fake_execute(*, drift_trend: bool = False, fail_seed: int | None = None):
    def execute(dataset, spec):
        seed = spec.config.ppo_seed
        if seed == fail_seed:
            raise RuntimeError(f"seed failed: {seed}")
        trend_intent = (
            PositionIntent.SHORT if drift_trend and seed == 5 else PositionIntent.LONG
        )
        ppo_intent = PositionIntent.LONG if seed % 2 == 0 else PositionIntent.SHORT
        strategies = {
            "cash": ConstantIntentStrategy(PositionIntent.FLAT),
            "constant_long": ConstantIntentStrategy(PositionIntent.LONG),
            "constant_short": ConstantIntentStrategy(PositionIntent.SHORT),
            "trend": ConstantIntentStrategy(trend_intent),
            "mean_reversion": ConstantIntentStrategy(PositionIntent.SHORT),
            "ridge24": ConstantIntentStrategy(PositionIntent.LONG),
            "lightgbm24": ConstantIntentStrategy(PositionIntent.SHORT),
            "ppo": ConstantIntentStrategy(ppo_intent),
        }
        comparison = compare_strategies_by_symbol(
            dataset,
            strategies,
            start_index=spec.evaluation_start_index,
            stop_index=spec.evaluation_stop_index,
            gross_budget=spec.config.gross_budget,
            initial_capital=spec.config.initial_capital,
        )
        return CandidateRunResult(
            spec=spec,
            symbols=tuple(dataset.symbols),
            comparison=comparison,
        )

    return execute


def test_evidence_set_executes_every_seed_through_shared_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    context = content_digest({"kind": "baseline", "study": plan.digest})
    seen: list[int] = []
    original_resolve = evidence_module.resolve_candidate_run_spec

    def recording_resolve(*args, **kwargs):
        spec = original_resolve(*args, **kwargs)
        seen.append(spec.config.ppo_seed)
        return spec

    monkeypatch.setattr(
        evidence_module, "resolve_candidate_run_spec", recording_resolve
    )
    monkeypatch.setattr(
        evidence_module,
        "execute_candidate_run",
        _fake_execute(),
    )

    store = StudyStore(tmp_path / "study")
    result = execute_evidence_set(
        store=store,
        target=Path("evidence/baseline"),
        dataset_root=dataset_root,
        plan=plan,
        config=_config(),
        research_context_digest=context,
    )
    loaded = load_evidence_set(store.root / "evidence" / "baseline")

    assert seen == [2, 5, 9]
    assert result.ppo_seeds == (2, 5, 9)
    assert result.research_context_digest == context
    assert loaded.evidence == result
    assert tuple(loaded.runs) == (2, 5, 9)
    assert [
        loaded.runs[seed].summary["candidate_config"]["ppo_seed"]
        for seed in plan.ppo_seeds
    ] == [2, 5, 9]


def test_evidence_set_rejects_deterministic_seed_drift_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    monkeypatch.setattr(
        evidence_module,
        "execute_candidate_run",
        _fake_execute(drift_trend=True),
    )
    store = StudyStore(tmp_path / "study")

    with pytest.raises(ArtifactIntegrityError, match="deterministic strategy"):
        execute_evidence_set(
            store=store,
            target=Path("evidence/baseline"),
            dataset_root=dataset_root,
            plan=plan,
            config=_config(),
            research_context_digest=content_digest({"study": plan.digest}),
        )

    assert not (store.root / "evidence" / "baseline").exists()


def test_evidence_set_checks_plan_provenance_before_any_seed_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = replace(_plan(dataset, artifact), implementation_digest="0" * 64)
    calls = 0

    def should_not_execute(dataset, spec):
        del dataset, spec
        nonlocal calls
        calls += 1
        raise AssertionError("execution must not begin")

    monkeypatch.setattr(evidence_module, "execute_candidate_run", should_not_execute)

    with pytest.raises(ArtifactIntegrityError, match="implementation provenance"):
        execute_evidence_set(
            store=StudyStore(tmp_path / "study"),
            target=Path("evidence/baseline"),
            dataset_root=dataset_root,
            plan=plan,
            config=_config(),
            research_context_digest=content_digest({"study": plan.digest}),
        )

    assert calls == 0


def test_partial_seed_failure_publishes_nothing_and_retry_is_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    context = content_digest({"study": plan.digest})
    store = StudyStore(tmp_path / "study")
    monkeypatch.setattr(
        evidence_module,
        "execute_candidate_run",
        _fake_execute(fail_seed=5),
    )

    with pytest.raises(RuntimeError, match="seed failed: 5"):
        execute_evidence_set(
            store=store,
            target=Path("evidence/baseline"),
            dataset_root=dataset_root,
            plan=plan,
            config=_config(),
            research_context_digest=context,
        )

    assert not (store.root / "evidence" / "baseline").exists()
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())

    result = execute_evidence_set(
        store=store,
        target=Path("evidence/baseline"),
        dataset_root=dataset_root,
        plan=plan,
        config=_config(),
        research_context_digest=context,
    )
    assert result.ppo_seeds == plan.ppo_seeds
