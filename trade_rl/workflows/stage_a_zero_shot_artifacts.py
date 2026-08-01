"""Atomic publication of completed Stage A orchestration phases."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.stage_a_zero_shot_contracts import (
    write_stage_a_evaluation_evidence,
)
from trade_rl.evaluation.stage_a_zero_shot_gate import (
    write_stage_a_sealed_test_decision,
    write_stage_a_validation_selection,
)
from trade_rl.workflows.stage_a_zero_shot_runner_contracts import (
    StageASealedTestRun,
    StageAValidationRun,
)

_STAGE_A_ACCESS_RECORDS_SCHEMA = "stage_a_sealed_test_access_records_v3"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class StageAZeroShotArtifactPublisher:
    """Publish immutable validation and sealed-test directories fail-closed."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _publish(
        self,
        name: str,
        writer: Callable[[Path], None],
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / name
        lock = self.root / f".{name}.lock"
        staging: Path | None = None
        lock_acquired = False
        try:
            with lock.open("xb"):
                pass
            lock_acquired = True
            if final.exists():
                raise FileExistsError(f"Stage A {name} package already exists")
            staging = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=self.root))
            writer(staging)
            if final.exists():
                raise FileExistsError(f"Stage A {name} package already exists")
            os.replace(staging, final)
            staging = None
            _fsync_directory(self.root)
            return final
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
            if lock_acquired:
                lock.unlink(missing_ok=True)

    def publish_validation(self, run: StageAValidationRun) -> Path:
        def write(staging: Path) -> None:
            write_stage_a_evaluation_evidence(staging / "evidence.json", run.evidence)
            write_stage_a_validation_selection(
                staging / "selection.json", run.selection
            )

        return self._publish("validation", write)

    def publish_sealed_test(self, run: StageASealedTestRun) -> Path:
        def write(staging: Path) -> None:
            write_stage_a_evaluation_evidence(staging / "evidence.json", run.evidence)
            write_stage_a_sealed_test_decision(staging / "decision.json", run.decision)
            batch_digests = {
                record.authorization_batch_digest for record in run.access_records
            }
            if len(batch_digests) != 1:
                raise ValueError(
                    "Stage A sealed-test access records must share one batch"
                )
            authorization_batch_digest = next(iter(batch_digests))
            records = tuple(
                {
                    "access_digest": record.access_digest,
                    "authorization_batch_digest": (
                        record.authorization_batch_digest
                    ),
                    "dataset_id": record.dataset_id,
                    "evaluation_dataset_manifest_digest": (
                        record.evaluation_dataset_manifest_digest
                    ),
                    "experiment_plan_digest": (
                        record.ledger_record.experiment_plan_digest
                    ),
                    "fold_index": record.fold,
                    "ledger_access_digest": record.ledger_record.access_digest,
                    "selected_configuration": (
                        record.ledger_record.selected_configuration
                    ),
                    "selected_policy_digest": (
                        record.ledger_record.selected_policy_digest
                    ),
                    "test_range": (record.test_range.start, record.test_range.stop),
                    "triplet_id": record.triplet_id,
                }
                for record in run.access_records
            )
            body: dict[str, object] = {
                "authorization_batch_digest": authorization_batch_digest,
                "records": records,
                "schema_version": _STAGE_A_ACCESS_RECORDS_SCHEMA,
            }
            payload = {"digest": content_digest(body), **body}
            atomic_write_bytes(
                staging / "access-records.json", canonical_json_bytes(payload)
            )

        return self._publish("sealed-test", write)


__all__ = ["StageAZeroShotArtifactPublisher"]
