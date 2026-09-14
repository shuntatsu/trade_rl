from __future__ import annotations

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
from trade_rl.evaluation.experiments.contracts import (
    ControlledFactor,
    ResolvedRunConfig,
    StudyPlan,
)
from trade_rl.evaluation.experiments.evidence import execute_evidence_set
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs.config import (
    CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    CandidateRunConfig,
    resolve_candidate_run_spec,
)
from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance


def _dataset():
    timestamps = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(
        24
    ) * np.timedelta64(1, "h")
    close = 100.0 + np.arange(24, dtype=np.float64)
    raw = RawMarketSeries(
        timestamps=timestamps,
        open=np.concatenate([close[:1], close[:-1]]),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full(24, 1_000_000.0),
        funding_rate=np.zeros(24),
        tradable=np.ones(24, dtype=np.bool_),
    )
    return MarketDatasetBuilder(
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="signal", kind=FeatureKind.LOG_RETURN),),
        )
    ).build(
        InMemoryMarketDataSource({"BTCUSDT": raw}),
        (
            InstrumentContract(
                symbol="BTCUSDT",
                listed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        ),
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


def test_evidence_execution_reinjects_study_frozen_causal_overlay_before_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    config = _config()
    resolved = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        config=config,
        execution_overlay=CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY,
    )
    provenance = build_candidate_run_provenance()
    plan = StudyPlan(
        research_question="Does causal calibrated execution survive?",
        dataset_id=dataset.dataset_id,
        dataset_artifact_schema=artifact.schema_version,
        dataset_artifact_digest=artifact.artifact_digest,
        symbols=tuple(dataset.symbols),
        baseline_config=ResolvedRunConfig.from_candidate_spec(resolved),
        ppo_seeds=(2, 5, 9),
        allowed_factors=(ControlledFactor.PPO_TRAINING_BUDGET,),
        max_experiments=1,
        n_bootstrap=100,
        bootstrap_seed=17,
        implementation_digest=str(provenance["implementation_digest"]),
        runtime_environment_digest=str(provenance["runtime_environment_digest"]),
    )

    seen: list[str] = []
    original_resolve = evidence_module.resolve_candidate_run_spec

    class StopBeforeReplay(RuntimeError):
        pass

    def capture_overlay(*args, **kwargs):
        seen.append(kwargs["execution_overlay"])
        original_resolve(*args, **kwargs)
        raise StopBeforeReplay

    monkeypatch.setattr(evidence_module, "resolve_candidate_run_spec", capture_overlay)

    with pytest.raises(StopBeforeReplay):
        execute_evidence_set(
            store=StudyStore(tmp_path / "study"),
            target=Path("evidence/baseline"),
            dataset_root=dataset_root,
            plan=plan,
            config=config,
            research_context_digest=content_digest({"study": plan.digest}),
        )

    assert seen == [CAUSAL_PREVIOUS_BAR_CAPACITY_EXECUTION_OVERLAY]
    assert not (tmp_path / "study" / "evidence" / "baseline").exists()
