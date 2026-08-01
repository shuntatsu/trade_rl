from __future__ import annotations

import hashlib
import json
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest

from tests.stage_a_helpers import stage_a_test_manifest
from trade_rl.cli import build_parser, main
from trade_rl.cli import stage_a as stage_a_cli
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    StageACandidate,
    build_stage_a_zero_shot_evaluation_plan,
    write_stage_a_zero_shot_evaluation_plan,
)
from trade_rl.workflows.stage_a_evaluation_dataset_manifest import (
    StageAEvaluationDatasetManifest,
    write_stage_a_evaluation_dataset_manifest,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageAEvaluationCellRequest,
    StageAEvaluationCellResult,
)


_COMMON_ARGS = [
    "--plan",
    "plan.json",
    "--manifest",
    "manifest.json",
    "--execution-store",
    "execution-store",
    "--baseline-config-digest",
    "a" * 64,
    "--output-root",
    "output",
]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest() -> StageAEvaluationDatasetManifest:
    return stage_a_test_manifest(
        symbol_disjoint_manifest_digest=_digest("symbol-manifest"),
        symbol_disjoint_triplet_manifest_digest=_digest("triplet-manifest"),
        feature_identity=_digest("features"),
        validation_triplet_ids=(
            _digest("validation-triplet-a"),
            _digest("validation-triplet-b"),
        ),
        test_triplet_ids=(_digest("test-triplet-a"), _digest("test-triplet-b")),
        folds=(0, 1),
    )


def _plan(
    manifest: StageAEvaluationDatasetManifest,
    *,
    passing_threshold: float = 0.05,
):
    seeds = (0, 1)
    candidates = tuple(
        StageACandidate.create(
            candidate_id=candidate_id,
            candidate_config_digest=_digest(f"{candidate_id}:config"),
            final_training_completion_digest=_digest(f"{candidate_id}:complete"),
            policy_identity=_digest(f"{candidate_id}:policy"),
            checkpoint_digests=tuple(
                (seed, _digest(f"{candidate_id}:checkpoint:{seed}")) for seed in seeds
            ),
        )
        for candidate_id in ("candidate-a", "candidate-b")
    )
    return build_stage_a_zero_shot_evaluation_plan(
        symbol_disjoint_manifest_digest=manifest.symbol_disjoint_manifest_digest,
        symbol_disjoint_triplet_manifest_digest=(
            manifest.symbol_disjoint_triplet_manifest_digest
        ),
        evaluation_dataset_manifest_digest=manifest.digest,
        feature_identity=manifest.feature_identity,
        execution_identity=_digest("execution"),
        evaluation_identity=_digest("evaluation"),
        candidates=candidates,
        seeds=seeds,
        folds=manifest.folds_declared,
        validation_triplet_ids=manifest.triplet_ids_for("validation"),
        test_triplet_ids=manifest.triplet_ids_for("test"),
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=1_000,
        bootstrap_seed=17,
        minimum_validation_lower_bound=passing_threshold,
        minimum_test_lower_bound=passing_threshold,
        minimum_validation_worst_triplet_excess=passing_threshold,
        minimum_test_worst_triplet_excess=passing_threshold,
        minimum_validation_worst_seed_excess=passing_threshold,
        minimum_test_worst_seed_excess=passing_threshold,
        minimum_validation_triplet_pass_fraction=1.0,
        minimum_test_triplet_pass_fraction=1.0,
    )


class RecordingEvaluator:
    def __init__(
        self,
        *,
        growth_by_candidate: dict[str | None, float] | None = None,
    ) -> None:
        self.growth_by_candidate = growth_by_candidate or {
            None: 0.0,
            "candidate-a": 0.20,
            "candidate-b": 0.10,
        }
        self.requests: list[StageAEvaluationCellRequest] = []

    def evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult:
        self.requests.append(request)
        return StageAEvaluationCellResult(
            request_digest=request.digest,
            execution_evidence_digest=_digest(f"execution:{request.digest}"),
            log_growth=self.growth_by_candidate[request.candidate_id],
        )


def _write_inputs(
    root: Path,
    *,
    passing_threshold: float = 0.05,
):
    manifest = _manifest()
    plan = _plan(manifest, passing_threshold=passing_threshold)
    plan_path = write_stage_a_zero_shot_evaluation_plan(root / "plan.json", plan)
    manifest_path = write_stage_a_evaluation_dataset_manifest(
        root / "manifest.json", manifest
    )
    return plan, manifest, plan_path, manifest_path


def test_parser_exposes_stage_a_commands() -> None:
    parser = build_parser()

    validation = parser.parse_args(["stage-a", "validation", *_COMMON_ARGS])
    assert validation.stage_a_command == "validation"

    sealed = parser.parse_args(
        [
            "stage-a",
            "sealed-test",
            *_COMMON_ARGS,
            "--validation-package",
            "output/validation",
            "--database-url",
            "postgresql://example",
        ]
    )
    assert sealed.stage_a_command == "sealed-test"

    complete = parser.parse_args(
        [
            "stage-a",
            "run",
            *_COMMON_ARGS,
            "--database-url",
            "postgresql://example",
        ]
    )
    assert complete.stage_a_command == "run"


def test_top_level_cli_routes_stage_a_without_importing_application(monkeypatch) -> None:
    calls: list[tuple[list[str], object, object]] = []
    fake = ModuleType("trade_rl.cli.stage_a")

    def fake_main(argv, *, stdout, stderr):
        calls.append((list(argv), stdout, stderr))
        return 17

    fake.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trade_rl.cli.stage_a", fake)
    stdout = StringIO()
    stderr = StringIO()

    assert main(["stage-a", "validation"], stdout=stdout, stderr=stderr) == 17
    assert calls == [(["validation"], stdout, stderr)]


def test_stage_a_subcommands_require_common_identity_inputs() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["stage-a", "validation"])


def test_validation_command_publishes_complete_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, manifest, plan_path, manifest_path = _write_inputs(tmp_path)
    evaluator = RecordingEvaluator()
    monkeypatch.setattr(
        stage_a_cli,
        "_build_evaluator",
        lambda **_: evaluator,
        raising=False,
    )
    output_root = tmp_path / "result"
    stdout = StringIO()

    assert (
        main(
            [
                "stage-a",
                "validation",
                "--plan",
                str(plan_path),
                "--manifest",
                str(manifest_path),
                "--execution-store",
                str(tmp_path / "execution-store"),
                "--baseline-config-digest",
                _digest("baseline-config"),
                "--output-root",
                str(output_root),
            ],
            stdout=stdout,
        )
        == 0
    )

    package = output_root / "validation"
    assert (package / "evidence.json").is_file()
    assert (package / "selection.json").is_file()
    payload = json.loads(stdout.getvalue())
    assert payload == {
        "evaluation_dataset_manifest_digest": manifest.digest,
        "package_path": str(package),
        "passed": True,
        "plan_digest": plan.digest,
        "reason": "candidate_selected_by_validation_gate",
        "schema": "stage_a_validation_cli_result_v1",
        "selected_candidate_id": "candidate-a",
        "validation_evidence_digest": payload["validation_evidence_digest"],
        "validation_run_digest": payload["validation_run_digest"],
        "validation_selection_digest": payload["validation_selection_digest"],
    }
    assert all(len(payload[key]) == 64 for key in (
        "validation_evidence_digest",
        "validation_run_digest",
        "validation_selection_digest",
    ))
    expected_cells = (
        len(plan.validation_triplet_ids) * len(plan.folds) * len(plan.seeds)
    )
    assert len([request for request in evaluator.requests if request.is_baseline]) == (
        expected_cells
    )
    assert len([request for request in evaluator.requests if not request.is_baseline]) == (
        expected_cells * len(plan.candidate_ids)
    )
