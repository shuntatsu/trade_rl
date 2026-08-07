"""Fail-closed Stage A evaluation command surface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from trade_rl.catalog.postgres_stage_a_sealed_test import (
    PostgresStageASealedTestLedger,
)
from trade_rl.evaluation.stage_a_sealed_test import StageASealedTestLedgerProtocol
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageAZeroShotEvaluationPlan,
    load_stage_a_evaluation_evidence,
    load_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.evaluation.stage_a_zero_shot_gate import (
    load_stage_a_validation_selection,
)
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetManifest,
    load_stage_a_evaluation_dataset_manifest,
)
from trade_rl.workflows.stage_a_execution_store import StageAExecutionPromotionStore
from trade_rl.workflows.stage_a_production_evaluator import (
    ArtifactBackedStageAEvaluationCellEvaluator,
)
from trade_rl.workflows.stage_a_zero_shot_artifacts import (
    StageAZeroShotArtifactPublisher,
)
from trade_rl.workflows.stage_a_zero_shot_runner import (
    StageAZeroShotEvaluationOrchestrator,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellEvaluator,
    StageASealedTestRun,
    StageAValidationRun,
)


def _write_json(stdout: TextIO, payload: object) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    stdout.write("\n")


def _load_context(
    args: argparse.Namespace,
) -> tuple[StageAZeroShotEvaluationPlan, StageAEvaluationDatasetManifest]:
    plan = load_stage_a_zero_shot_evaluation_plan(args.plan)
    manifest = load_stage_a_evaluation_dataset_manifest(args.manifest)
    plan.validate_manifest(manifest)
    return plan, manifest


def _build_evaluator(
    *,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
    execution_store: str | Path,
    baseline_config_digest: str,
) -> StageAEvaluationCellEvaluator:
    return ArtifactBackedStageAEvaluationCellEvaluator(
        plan=plan,
        manifest=manifest,
        store=StageAExecutionPromotionStore(execution_store),
        baseline_candidate_config_digest=baseline_config_digest,
    )


def _build_ledger(database_url: str) -> StageASealedTestLedgerProtocol:
    return PostgresStageASealedTestLedger(database_url)


def _database_url(args: argparse.Namespace) -> str:
    value = args.database_url or os.environ.get("TRADE_RL_DATABASE_URL")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "provide --database-url or set TRADE_RL_DATABASE_URL for Stage A sealed test"
        )
    return value


def _evaluator(
    *,
    args: argparse.Namespace,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
) -> StageAEvaluationCellEvaluator:
    return _build_evaluator(
        plan=plan,
        manifest=manifest,
        execution_store=args.execution_store,
        baseline_config_digest=args.baseline_config_digest,
    )


def _load_validation_run(
    package: str | Path,
    *,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
) -> StageAValidationRun:
    root = Path(package)
    evidence = load_stage_a_evaluation_evidence(
        root / "evidence.json",
        plan=plan,
        manifest=manifest,
    )
    selection = load_stage_a_validation_selection(
        root / "selection.json",
        plan=plan,
        evidence=evidence,
    )
    return StageAValidationRun(evidence=evidence, selection=selection)


def _validation_payload(
    *,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
    run: StageAValidationRun,
    package: Path,
) -> dict[str, object]:
    return {
        "evaluation_dataset_manifest_digest": manifest.digest,
        "package_path": str(package),
        "passed": run.selection.passed,
        "plan_digest": plan.digest,
        "reason": run.selection.reason,
        "selected_candidate_id": run.selection.selected_candidate_id,
        "validation_evidence_digest": run.evidence.digest,
        "validation_run_digest": run.digest,
        "validation_selection_digest": run.selection.digest,
    }


def _sealed_test_payload(
    *,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
    run: StageASealedTestRun,
    package: Path,
) -> dict[str, object]:
    return {
        "authorization_batch_digest": run.authorization_batch_digest,
        "decision_digest": run.decision.digest,
        "evaluation_dataset_manifest_digest": manifest.digest,
        "package_path": str(package),
        "passed": run.decision.passed,
        "plan_digest": plan.digest,
        "reason": run.decision.reason,
        "sealed_test_run_digest": run.digest,
        "selected_candidate_id": run.decision.selected_candidate_id,
        "test_evidence_digest": run.evidence.digest,
        "validation_run_digest": run.validation_run.digest,
    }


def _evaluate_validation(
    *,
    args: argparse.Namespace,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
    evaluator: StageAEvaluationCellEvaluator,
) -> tuple[StageAValidationRun, Path]:
    run = StageAZeroShotEvaluationOrchestrator(
        plan=plan,
        manifest=manifest,
        evaluator=evaluator,
    ).evaluate_validation()
    package = StageAZeroShotArtifactPublisher(args.output_root).publish_validation(run)
    return run, package


def _validation(args: argparse.Namespace, stdout: TextIO) -> int:
    plan, manifest = _load_context(args)
    run, package = _evaluate_validation(
        args=args,
        plan=plan,
        manifest=manifest,
        evaluator=_evaluator(args=args, plan=plan, manifest=manifest),
    )
    _write_json(
        stdout,
        {
            "schema": "stage_a_validation_cli_result_v1",
            **_validation_payload(
                plan=plan,
                manifest=manifest,
                run=run,
                package=package,
            ),
        },
    )
    return 0


def _sealed_test(args: argparse.Namespace, stdout: TextIO) -> int:
    plan, manifest = _load_context(args)
    validation_run = _load_validation_run(
        args.validation_package,
        plan=plan,
        manifest=manifest,
    )
    evaluator = _evaluator(args=args, plan=plan, manifest=manifest)
    ledger = _build_ledger(_database_url(args))
    run = StageAZeroShotEvaluationOrchestrator(
        plan=plan,
        manifest=manifest,
        evaluator=evaluator,
        sealed_test_ledger=ledger,
    ).evaluate_sealed_test(validation_run)
    package = StageAZeroShotArtifactPublisher(args.output_root).publish_sealed_test(run)
    _write_json(
        stdout,
        {
            "schema": "stage_a_sealed_test_cli_result_v1",
            **_sealed_test_payload(
                plan=plan,
                manifest=manifest,
                run=run,
                package=package,
            ),
        },
    )
    return 0


def _complete_run(args: argparse.Namespace, stdout: TextIO) -> int:
    plan, manifest = _load_context(args)
    evaluator = _evaluator(args=args, plan=plan, manifest=manifest)
    validation_run, validation_package = _evaluate_validation(
        args=args,
        plan=plan,
        manifest=manifest,
        evaluator=evaluator,
    )
    validation_payload = _validation_payload(
        plan=plan,
        manifest=manifest,
        run=validation_run,
        package=validation_package,
    )
    if not validation_run.selection.passed:
        _write_json(
            stdout,
            {
                "evaluation_dataset_manifest_digest": manifest.digest,
                "plan_digest": plan.digest,
                "schema": "stage_a_complete_run_cli_result_v1",
                "sealed_test": None,
                "validation": validation_payload,
            },
        )
        return 0

    ledger = _build_ledger(_database_url(args))
    sealed_run = StageAZeroShotEvaluationOrchestrator(
        plan=plan,
        manifest=manifest,
        evaluator=evaluator,
        sealed_test_ledger=ledger,
    ).evaluate_sealed_test(validation_run)
    sealed_package = StageAZeroShotArtifactPublisher(
        args.output_root
    ).publish_sealed_test(sealed_run)
    _write_json(
        stdout,
        {
            "evaluation_dataset_manifest_digest": manifest.digest,
            "plan_digest": plan.digest,
            "schema": "stage_a_complete_run_cli_result_v1",
            "sealed_test": _sealed_test_payload(
                plan=plan,
                manifest=manifest,
                run=sealed_run,
                package=sealed_package,
            ),
            "validation": validation_payload,
        },
    )
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--execution-store", required=True)
    parser.add_argument("--baseline-config-digest", required=True)
    parser.add_argument("--output-root", required=True)


def _add_commands(subparsers: argparse._SubParsersAction) -> None:
    validation = subparsers.add_parser(
        "validation",
        help="evaluate and atomically publish Stage A validation",
    )
    _add_common_arguments(validation)
    validation.set_defaults(handler=_validation)

    sealed_test = subparsers.add_parser(
        "sealed-test",
        help="open, evaluate, and atomically publish the Stage A sealed test",
    )
    _add_common_arguments(sealed_test)
    sealed_test.add_argument("--validation-package", required=True)
    sealed_test.add_argument("--database-url")
    sealed_test.set_defaults(handler=_sealed_test)

    complete = subparsers.add_parser(
        "run",
        help="run validation and conditionally open the Stage A sealed test",
    )
    _add_common_arguments(complete)
    complete.add_argument("--database-url")
    complete.set_defaults(handler=_complete_run)


def add_stage_a_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register Stage A commands on the authoritative application parser."""

    stage_a = subparsers.add_parser(
        "stage-a",
        help="unseen-symbol validation and one-shot sealed-test evaluation",
    )
    commands = stage_a.add_subparsers(dest="stage_a_command", required=True)
    _add_commands(commands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade-rl stage-a",
        description="Stage A unseen-symbol evaluation orchestration.",
    )
    commands = parser.add_subparsers(dest="stage_a_command", required=True)
    _add_commands(commands)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    del stderr
    arguments = list(sys.argv[1:] if argv is None else argv)
    output = stdout or sys.stdout
    args = build_parser().parse_args(arguments)
    handler = args.handler
    return int(handler(args, output))


__all__ = ["add_stage_a_parser", "build_parser", "main"]
