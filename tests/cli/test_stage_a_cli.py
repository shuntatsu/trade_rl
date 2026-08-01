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
from trade_rl.evaluation.stage_a_sealed_test import (
    StageASealedTestAuthorizationBatch,
    StageASealedTestLedger,
)
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


class RecordingLedger:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.delegate = StageASealedTestLedger()

    @property
    def records(self) -> tuple[StageASealedTestAuthorizationBatch, ...]:
        return self.delegate.records

    def authorize_once(
        self,
        batch: StageASealedTestAuthorizationBatch,
    ) -> StageASealedTestAuthorizationBatch:
        return self.delegate.authorize_once(batch)


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


def _command_args(
    *,
    plan_path: Path,
    manifest_path: Path,
    root: Path,
) -> list[str]:
    return [
        "--plan",
        str(plan_path),
        "--manifest",
        str(manifest_path),
        "--execution-store",
        str(root / "execution-store"),
        "--baseline-config-digest",
        _digest("baseline-config"),
        "--output-root",
        str(root / "result"),
    ]


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
    monkeypatch.setattr(stage_a_cli, "_build_evaluator", lambda **_: evaluator)
    command_args = _command_args(
        plan_path=plan_path,
        manifest_path=manifest_path,
        root=tmp_path,
    )
    stdout = StringIO()

    assert main(["stage-a", "validation", *command_args], stdout=stdout) == 0

    package = tmp_path / "result" / "validation"
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
    assert all(
        len(payload[key]) == 64
        for key in (
            "validation_evidence_digest",
            "validation_run_digest",
            "validation_selection_digest",
        )
    )
    expected_cells = (
        len(plan.validation_triplet_ids) * len(plan.folds) * len(plan.seeds)
    )
    assert len([request for request in evaluator.requests if request.is_baseline]) == (
        expected_cells
    )
    assert len([request for request in evaluator.requests if not request.is_baseline]) == (
        expected_cells * len(plan.candidate_ids)
    )


def test_sealed_test_command_loads_validation_and_uses_explicit_database_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, manifest, plan_path, manifest_path = _write_inputs(tmp_path)
    evaluator = RecordingEvaluator()
    monkeypatch.setattr(stage_a_cli, "_build_evaluator", lambda **_: evaluator)
    command_args = _command_args(
        plan_path=plan_path,
        manifest_path=manifest_path,
        root=tmp_path,
    )
    assert main(["stage-a", "validation", *command_args], stdout=StringIO()) == 0
    evaluator.requests.clear()

    database_url = "postgresql://stage-a-explicit"
    ledger = RecordingLedger(database_url)
    database_urls: list[str] = []

    def build_ledger(value: str) -> RecordingLedger:
        database_urls.append(value)
        return ledger

    monkeypatch.setattr(stage_a_cli, "_build_ledger", build_ledger, raising=False)
    stdout = StringIO()
    assert (
        main(
            [
                "stage-a",
                "sealed-test",
                *command_args,
                "--validation-package",
                str(tmp_path / "result" / "validation"),
                "--database-url",
                database_url,
            ],
            stdout=stdout,
        )
        == 0
    )

    package = tmp_path / "result" / "sealed-test"
    assert (package / "evidence.json").is_file()
    assert (package / "decision.json").is_file()
    assert (package / "access-records.json").is_file()
    assert database_urls == [database_url]
    assert len(ledger.records) == 1
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "stage_a_sealed_test_cli_result_v1"
    assert payload["plan_digest"] == plan.digest
    assert payload["evaluation_dataset_manifest_digest"] == manifest.digest
    assert payload["package_path"] == str(package)
    assert payload["passed"] is True
    assert payload["selected_candidate_id"] == "candidate-a"
    assert payload["authorization_batch_digest"] == ledger.records[0].batch_digest
    assert database_url not in stdout.getvalue()
    expected_cells = len(plan.test_triplet_ids) * len(plan.folds) * len(plan.seeds)
    assert len([request for request in evaluator.requests if request.is_baseline]) == (
        expected_cells
    )
    assert len([request for request in evaluator.requests if not request.is_baseline]) == (
        expected_cells
    )


def test_sealed_test_rejects_tampered_validation_before_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, plan_path, manifest_path = _write_inputs(tmp_path)
    evaluator = RecordingEvaluator()
    monkeypatch.setattr(stage_a_cli, "_build_evaluator", lambda **_: evaluator)
    command_args = _command_args(
        plan_path=plan_path,
        manifest_path=manifest_path,
        root=tmp_path,
    )
    assert main(["stage-a", "validation", *command_args], stdout=StringIO()) == 0
    selection_path = tmp_path / "result" / "validation" / "selection.json"
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    payload["selected_candidate_id"] = "candidate-b"
    selection_path.write_text(json.dumps(payload), encoding="utf-8")

    ledger_calls: list[str] = []
    monkeypatch.setattr(
        stage_a_cli,
        "_build_ledger",
        lambda value: ledger_calls.append(value),
        raising=False,
    )
    with pytest.raises(ValueError):
        main(
            [
                "stage-a",
                "sealed-test",
                *command_args,
                "--validation-package",
                str(tmp_path / "result" / "validation"),
                "--database-url",
                "postgresql://must-not-open",
            ],
            stdout=StringIO(),
        )
    assert ledger_calls == []


def test_complete_run_does_not_resolve_database_after_failed_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, plan_path, manifest_path = _write_inputs(tmp_path)
    evaluator = RecordingEvaluator(
        growth_by_candidate={None: 0.0, "candidate-a": 0.0, "candidate-b": 0.0}
    )
    monkeypatch.setattr(stage_a_cli, "_build_evaluator", lambda **_: evaluator)
    monkeypatch.delenv("TRADE_RL_DATABASE_URL", raising=False)
    ledger_calls: list[str] = []
    monkeypatch.setattr(
        stage_a_cli,
        "_build_ledger",
        lambda value: ledger_calls.append(value),
        raising=False,
    )
    stdout = StringIO()

    assert (
        main(
            [
                "stage-a",
                "run",
                *_command_args(
                    plan_path=plan_path,
                    manifest_path=manifest_path,
                    root=tmp_path,
                ),
            ],
            stdout=stdout,
        )
        == 0
    )

    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "stage_a_complete_run_cli_result_v1"
    assert payload["validation"]["passed"] is False
    assert payload["sealed_test"] is None
    assert (tmp_path / "result" / "validation").is_dir()
    assert not (tmp_path / "result" / "sealed-test").exists()
    assert ledger_calls == []
    assert all(request.split == "validation" for request in evaluator.requests)


def test_complete_run_uses_environment_database_url_only_after_validation_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, manifest, plan_path, manifest_path = _write_inputs(tmp_path)
    evaluator = RecordingEvaluator()
    monkeypatch.setattr(stage_a_cli, "_build_evaluator", lambda **_: evaluator)
    database_url = "postgresql://stage-a-from-environment"
    monkeypatch.setenv("TRADE_RL_DATABASE_URL", database_url)
    ledgers: list[RecordingLedger] = []

    def build_ledger(value: str) -> RecordingLedger:
        ledger = RecordingLedger(value)
        ledgers.append(ledger)
        return ledger

    monkeypatch.setattr(stage_a_cli, "_build_ledger", build_ledger, raising=False)
    stdout = StringIO()
    assert (
        main(
            [
                "stage-a",
                "run",
                *_command_args(
                    plan_path=plan_path,
                    manifest_path=manifest_path,
                    root=tmp_path,
                ),
            ],
            stdout=stdout,
        )
        == 0
    )

    assert len(ledgers) == 1
    assert ledgers[0].database_url == database_url
    assert len(ledgers[0].records) == 1
    assert (tmp_path / "result" / "validation").is_dir()
    assert (tmp_path / "result" / "sealed-test").is_dir()
    payload = json.loads(stdout.getvalue())
    assert payload["schema"] == "stage_a_complete_run_cli_result_v1"
    assert payload["plan_digest"] == plan.digest
    assert payload["evaluation_dataset_manifest_digest"] == manifest.digest
    assert payload["validation"]["passed"] is True
    assert payload["sealed_test"]["passed"] is True
    assert payload["sealed_test"]["authorization_batch_digest"] == (
        ledgers[0].records[0].batch_digest
    )
    assert database_url not in stdout.getvalue()
