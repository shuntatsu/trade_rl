"""Canonical market walk-forward orchestration and explicit evidence trust boundary."""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import trade_rl.workflows._market_walk_forward_core as _core
from trade_rl._source_checkout import source_checkout_root
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.provenance import capture_runtime_provenance
from trade_rl.artifacts.run_manifest import (
    WalkForwardRunManifest,
    validate_walk_forward_run_directory,
    write_walk_forward_run_manifest,
)
from trade_rl.artifacts.store import ArtifactStore
from trade_rl.catalog.postgres_sealed_test import (
    PostgresSealedTestReservationStore,
)
from trade_rl.catalog.sealed_test import PostgresSealedTestLedger
from trade_rl.data import load_market_dataset_artifact
from trade_rl.evaluation.walk_forward.sealed_test import (
    SealedTestLedger,
    SealedTestLedgerProtocol,
)
from trade_rl.workflows._market_walk_forward_core import (
    CandidateConfiguration,
    ConcreteFoldRunner,
    FoldExecutionConfig,
    MarketCandidateEvaluator,
    MarketCandidateTrainer,
    WalkForwardExecutionResult,
    WalkForwardRunResult,
    execute_walk_forward,
    resolve_signal_digest,
)
from trade_rl.workflows.market_walk_forward_config import (
    MarketWalkForwardConfig,
    NamedCandidateRun,
    SealedTestLedgerMode,
)

__all__ = [
    "MarketCandidateEvaluator",
    "MarketCandidateTrainer",
    "MarketWalkForwardConfig",
    "NamedCandidateRun",
    "SealedTestLedgerMode",
    "WalkForwardRunResult",
    "execute_market_walk_forward",
]


def _validate_for_store(path: Path) -> bool:
    validate_walk_forward_run_directory(path)
    return True


def _sealed_test_ledger(mode: SealedTestLedgerMode) -> SealedTestLedgerProtocol:
    if mode is SealedTestLedgerMode.LOCAL_EXPLORATORY:
        return SealedTestLedger()
    database_url = os.environ.get("TRADE_RL_DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError(
            "durable PostgreSQL sealed-test ledger requires TRADE_RL_DATABASE_URL"
        )
    store = PostgresSealedTestReservationStore(database_url)
    return PostgresSealedTestLedger(store)


def execute_market_walk_forward(
    *,
    config_path: Path,
    dataset_path: Path,
    store_root: Path,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> WalkForwardRunResult:
    """Run concrete nested walk-forward research and publish immutable evidence."""

    resolved_created_at = created_at or datetime.now(UTC)
    resolved_run_id = run_id or resolved_created_at.strftime("wf-%Y%m%dT%H%M%SZ")
    dataset = load_market_dataset_artifact(dataset_path)
    config = MarketWalkForwardConfig.from_json(config_path, n_bars=dataset.n_bars)
    resolved_signal = resolve_signal_digest(config, dataset_id=dataset.dataset_id)
    config = replace(config, signal_digest=resolved_signal)
    config_digest = content_digest(config.digest_payload())
    provenance = capture_runtime_provenance(
        source_checkout_root(),
        git_commit=config.candidates[0].run.git_commit,
        git_dirty=config.candidates[0].run.git_dirty,
        deterministic_seed_config={
            "candidate_seeds": tuple(
                {
                    "name": item.name,
                    "seeds": item.run.training.seeds,
                }
                for item in config.candidates
            ),
            "workflow_config_digest": config_digest,
        },
    )
    experiment_plan_digest = _core._experiment_plan_digest(
        config,
        dataset_id=dataset.dataset_id,
    )
    store = ArtifactStore(store_root)
    stage = store.stage_run(resolved_run_id)
    _core._write_json(stage / "provenance.json", asdict(provenance))
    registry: dict[str, _core._PolicyRecord] = {}
    candidate_map = {item.name: item.run for item in config.candidates}
    trainer = MarketCandidateTrainer(
        dataset=dataset,
        candidates=candidate_map,
        root=stage,
        created_at=resolved_created_at,
        registry=registry,
        checkpoint_finalists_per_seed=config.checkpoint_finalists_per_seed,
    )
    evaluator = MarketCandidateEvaluator(
        dataset=dataset,
        baseline_run=config.candidates[0].run,
        registry=registry,
    )
    try:
        fold_runner = ConcreteFoldRunner(
            config=FoldExecutionConfig(
                dataset_id=dataset.dataset_id,
                signal_digest=config.signal_digest,
                candidates=tuple(
                    CandidateConfiguration(item.name) for item in config.candidates
                ),
                minimum_selection_uplift=config.minimum_selection_uplift,
                minimum_selection_score=config.minimum_selection_score,
                minimum_seed_success_fraction=(config.minimum_seed_success_fraction),
                minimum_worst_seed_uplift=config.minimum_worst_seed_uplift,
                maximum_seed_score_std=config.maximum_seed_score_std,
                maximum_selection_turnover_per_day=(
                    config.maximum_selection_turnover_per_day
                ),
                maximum_selection_cost_fraction=(
                    config.maximum_selection_cost_fraction
                ),
                maximum_selection_drawdown=config.maximum_selection_drawdown,
                selected_at=resolved_created_at,
                experiment_plan_digest=experiment_plan_digest,
            ),
            trainer=trainer,
            evaluator=evaluator,
            sealed_test_ledger=_sealed_test_ledger(config.sealed_test_ledger_mode),
        )
        result: WalkForwardExecutionResult = execute_walk_forward(
            config.workflow,
            dataset_id=dataset.dataset_id,
            runner=fold_runner,
        )
        sensitivity_payload = _core._evaluate_execution_sensitivity(
            config=config.execution_sensitivity,
            dataset=dataset,
            result=result,
            evaluator=evaluator,
            experiment_plan_digest=experiment_plan_digest,
        )
        sensitivity_by_fold: dict[int, dict[str, Any]] = {}
        if sensitivity_payload is not None:
            _core._write_json(stage / "execution-sensitivity.json", sensitivity_payload)
            sensitivity_by_fold = {
                int(item["fold_index"]): item
                for item in sensitivity_payload["folds"]
                if isinstance(item, dict)
            }
        folds_payload: list[dict[str, object]] = []
        for fold, fold_result in zip(
            result.folds,
            result.fold_results,
            strict=True,
        ):
            sealed_count = evaluator.outer_test_counts.get(fold.fold_index, 0)
            expected_count = (
                1 if fold_result.selection.selected_policy_digest is None else 2
            )
            if sealed_count != expected_count:
                raise RuntimeError(
                    "sealed outer test evaluation count violates the fold contract"
                )
            payload = _core._fold_payload(
                fold,
                fold_result,
                sealed_test_evaluations=sealed_count,
                initial_capital=config.candidates[0].run.environment.initial_capital,
                bar_hours=dataset.bar_hours,
            )
            sensitivity_fold = sensitivity_by_fold.get(fold.fold_index)
            if sensitivity_fold is not None:
                access_payload = sensitivity_fold.get("access")
                payload["execution_sensitivity_access"] = access_payload
                payload["execution_sensitivity_scenario_digests"] = tuple(
                    item.get("scenario_result_digest")
                    for item in sensitivity_fold.get("scenarios", ())
                    if isinstance(item, dict)
                )
            folds_payload.append(payload)
            _core._write_json(
                stage / f"fold-{fold.fold_index:03d}" / "result.json",
                payload,
            )
        walk_forward_payload = {
            "baseline_metrics": (
                None
                if result.baseline_metrics is None
                else asdict(result.baseline_metrics)
            ),
            "baseline_independent_summary": (
                None
                if result.baseline_independent_summary is None
                else asdict(result.baseline_independent_summary)
            ),
            "dataset_id": dataset.dataset_id,
            "evidence_tier": (
                "durable_sealed"
                if config.sealed_test_ledger_mode
                is SealedTestLedgerMode.DURABLE_POSTGRES
                else "exploratory_process_local"
            ),
            "evaluation_digest": result.evaluation_digest,
            "execution_sensitivity_digest": (
                None
                if sensitivity_payload is None
                else sensitivity_payload["artifact_digest"]
            ),
            "execution_sensitivity_gate": (
                None if sensitivity_payload is None else sensitivity_payload["gate"]
            ),
            "experiment_plan_digest": experiment_plan_digest,
            "folds": tuple(folds_payload),
            "production_status": "NO-GO",
            "sealed_test_ledger_durable": (
                config.sealed_test_ledger_mode is SealedTestLedgerMode.DURABLE_POSTGRES
            ),
            "sealed_test_ledger_mode": config.sealed_test_ledger_mode.value,
            "schema_version": "market_walk_forward_run_v5_deployable_ensemble",
            "selected_metrics": (
                None
                if result.selected_metrics is None
                else asdict(result.selected_metrics)
            ),
            "selected_independent_summary": (
                None
                if result.selected_independent_summary is None
                else asdict(result.selected_independent_summary)
            ),
            "stitch_mode": config.workflow.stitch_mode.value,
        }
        _core._write_json(stage / "walk-forward.json", walk_forward_payload)
        _core._write_json(stage / "walk-forward-config.json", config.digest_payload())
        _core._write_json(
            stage / "dataset-reference.json",
            {
                "artifact_path": str(dataset_path),
                "dataset_id": dataset.dataset_id,
                "feature_config_digest": dataset.feature_config_digest,
                "schema_version": "dataset_reference_v2",
            },
        )
        policy_digest = content_digest(
            {
                "policies": tuple(
                    {
                        "algorithm": record.algorithm,
                        "normalizer_digest": record.normalizer.digest,
                        "policy_digest": digest,
                        "run_config_digest": content_digest(
                            record.run.digest_payload()
                        ),
                    }
                    for digest, record in sorted(registry.items())
                ),
                "schema_version": "walk_forward_policy_set_v2",
            }
        )
        environment_digest = content_digest(
            {
                "candidates": tuple(
                    {
                        "name": item.name,
                        "run_environment": {
                            "action": asdict(item.run.action),
                            "environment": asdict(item.run.environment),
                            "risk": asdict(item.run.risk),
                            "reward": asdict(item.run.reward),
                            "trend": asdict(item.run.trend),
                        },
                    }
                    for item in config.candidates
                ),
                "schema_version": "walk_forward_environment_set_v1",
            }
        )
        run_manifest = WalkForwardRunManifest.build(
            root=stage,
            run_id=resolved_run_id,
            dataset_id=dataset.dataset_id,
            environment_digest=environment_digest,
            evaluation_digest=result.evaluation_digest,
            workflow_config_digest=config_digest,
            policy_set_digest=policy_digest,
            provenance_digest=provenance.digest,
            fold_count=len(result.folds),
            artifact_paths=_core._artifact_paths(stage),
            created_at=resolved_created_at,
        )
        write_walk_forward_run_manifest(stage, run_manifest)
        validate_walk_forward_run_directory(stage)
        published = store.publish_run(resolved_run_id, validate=_validate_for_store)
        return WalkForwardRunResult(
            run_id=resolved_run_id,
            status="published",
            path=published,
            run_digest=run_manifest.digest,
            evaluation_digest=result.evaluation_digest,
            dataset_id=dataset.dataset_id,
        )
    except Exception as error:
        if stage.is_dir():
            try:
                store.mark_failed(resolved_run_id)
            except Exception as isolation_error:
                error.add_note(
                    "failed to isolate partial walk-forward artifacts: "
                    f"{type(isolation_error).__name__}: {isolation_error}"
                )
        raise
