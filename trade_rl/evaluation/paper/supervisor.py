"""Single-collector public evidence acquisition bound to source and runtime."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

from trade_rl._validation import require_aware_datetime
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper.account import timestamp
from trade_rl.evaluation.paper.control import CollectionControl
from trade_rl.evaluation.paper.engine import EvidenceRef, PaperEngine, PaperSettings
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.integrations.binance.forward import capture_forward_snapshot
from trade_rl.integrations.binance.forward_rules import capture_forward_rules

_SCHEMA = "paper_public_collection_v1"
_FAILURE = "collection-failure.json"


def _now() -> datetime:
    return datetime.now(UTC)


def seal_collection(
    root: str | Path,
    *,
    settings: PaperSettings,
    research_plan: dict[str, Any],
    clock: Callable[[], datetime] = _now,
) -> str:
    """Freeze collection identity; the caller owns the separate economic plan."""
    created = require_aware_datetime(clock(), field="collection seal")
    if created >= settings.start_at:
        raise ValueError("collection must be sealed before its future start")
    if not isinstance(research_plan, dict) or not research_plan:
        raise ValueError("collection requires an explicit research plan")
    return PaperEngine.initialize(
        root,
        settings=settings,
        study=dict(
            schema=_SCHEMA,
            created_at=created.isoformat(),
            interval_seconds=60,
            terminal_grace_seconds=180,
            provenance=build_candidate_run_provenance(),
            research_plan=research_plan,
        ),
    )


class PaperCollector:
    """Use as a context manager. Any collection failure permanently halts I/O."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_protocol_sha256: str,
        clock: Callable[[], datetime] = _now,
        transport: Any = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root).absolute()
        self.digest = expected_protocol_sha256
        self.clock, self.transport, self.monotonic = clock, transport, monotonic
        self.engine: PaperEngine | None = None
        self._lock: sqlite3.Connection | None = None
        self._study: dict[str, Any] = {}
        self._rule: EvidenceRef | None = None
        self._rule_at: datetime | None = None
        self._control: CollectionControl | None = None
        self._cycle_id: int | None = None
        self._halted = False

    def __enter__(self) -> PaperCollector:
        if self._lock is not None:
            raise RuntimeError("collector is already open")
        path = self.root / "protocol.json"
        if self.root.is_symlink() or path.is_symlink() or not path.is_file():
            raise ValueError("collection protocol must be a regular file")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.digest:
            raise ValueError("collection protocol digest mismatch")
        protocol = json.loads(raw)["protocol"]
        study = protocol["study"]
        if (
            study.get("schema") != _SCHEMA
            or study.get("interval_seconds") != 60
            or study.get("terminal_grace_seconds") != 180
            or not isinstance(study.get("provenance"), dict)
        ):
            raise ValueError("unsupported public collection protocol")
        self._study = study
        self.start_at = timestamp(protocol["settings"]["start_at"])
        self.close_at = timestamp(protocol["settings"]["close_at"])
        if timestamp(study["created_at"]) >= self.start_at:
            raise ValueError("collection was not sealed before start")
        lock_path = self.root / "collector.sqlite"
        if lock_path.is_symlink() or (lock_path.exists() and not lock_path.is_file()):
            raise ValueError("collector lock must be a regular file")
        connection = sqlite3.connect(lock_path, timeout=0, isolation_level=None)
        try:
            connection.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as error:
            connection.close()
            raise RuntimeError("another collector holds this collection") from error
        self._lock = connection
        try:
            self._guard()
            self.engine = PaperEngine(self.root, expected_protocol_sha256=self.digest)
            self._control = CollectionControl(self.root, self.digest)
        except BaseException as error:
            try:
                self._halt(error)
            finally:
                self._release()
            raise
        return self

    def _release(self) -> None:
        if self._control is not None:
            self._control.close()
            self._control = None
        if self._lock is not None:
            self._lock.rollback()
            self._lock.close()
            self._lock = None

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if error is not None:
                self._halt(error)
        finally:
            self._release()

    def _guard(self) -> None:
        failure = self.root / _FAILURE
        if self._halted or failure.exists() or failure.is_symlink():
            raise RuntimeError("collection is permanently halted; inspect its failure")
        path = self.root / "protocol.json"
        if (
            self.root.is_symlink()
            or path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != self.digest
        ):
            raise ValueError("collection protocol digest mismatch")
        if build_candidate_run_provenance() != self._study["provenance"]:
            raise ValueError("collection source/runtime provenance changed")

    def _halt(self, error: BaseException) -> None:
        self._halted = True
        failure = self.root / _FAILURE
        if failure.exists() or failure.is_symlink():
            return
        record = dict(
            schema="paper_collection_failure_v1",
            protocol_sha256=self.digest,
            recorded_at=_now().isoformat(),
            error_type=type(error).__name__,
            error=str(error)[:2000],
            production_eligible=False,
        )
        with failure.open("xb") as stream:
            stream.write(canonical_json_bytes(record))
            stream.flush()
            os.fsync(stream.fileno())
        # Reconcile a possibly committed-but-unacknowledged event first. Never
        # use changed source to append financial events or invent a fresh mark.
        if (
            self.engine is not None
            and build_candidate_run_provenance() == self._study["provenance"]
        ):
            try:
                recovered = PaperEngine(self.root, expected_protocol_sha256=self.digest)
                at = self.clock()
                if at > timestamp(recovered.status()["last_command_at"]):
                    recovered.gap(
                        f"failure-{uuid.uuid4().hex}",
                        at=at,
                        reason=f"collector_failure:{record['error_type']}",
                    )
                self.engine = recovered
            except Exception:
                # The durable failure itself rejects qualification even if a
                # corrupt/unavailable journal cannot accept a gap command.
                pass

    def _capture(self, *, rules: bool) -> tuple[EvidenceRef, dict[str, Any]]:
        self._guard()
        name = f"sources/{self.clock():%Y-%m-%d}/{uuid.uuid4().hex}"
        capture = capture_forward_rules if rules else capture_forward_snapshot
        result = capture(
            self.root / name,
            transport=self.transport,
            clock=self.clock,
            monotonic=self.monotonic,
        )
        filename = "rules.json" if rules else "snapshot.json"
        digest = hashlib.sha256((self.root / name / filename).read_bytes()).hexdigest()
        return EvidenceRef(name, digest), result

    def _execute_pending(self) -> str:
        assert self.engine is not None
        decision_at = timestamp(self.engine.status()["last_command_at"])
        if (self.clock() - decision_at).total_seconds() <= 10:
            reference, _ = self._capture(rules=False)
            self._guard()
            at = self.clock()
            if self._finish_if_due(at):
                return "finished"
            if (at - decision_at).total_seconds() <= 10:
                self.engine.execute(
                    f"execute-{uuid.uuid4().hex}", market=reference, at=at
                )
                return "observed"
        self._guard()
        self.engine.gap(
            f"expired-{uuid.uuid4().hex}",
            at=self.clock(),
            reason="expired_pending_decision",
        )
        return "cancelled"

    def _finish_if_due(self, at: datetime) -> bool:
        assert self.engine is not None
        if at < self.close_at + timedelta(seconds=180):
            return False
        if self.engine.status()["pending"]:
            self.engine.gap(
                f"terminal-{uuid.uuid4().hex}", at=at, reason="terminal_window_expired"
            )
        return True

    def cycle(self) -> dict[str, Any]:
        if self._lock is None or self.engine is None:
            raise RuntimeError("collector must be open in a context manager")
        try:
            self._guard()
            at = require_aware_datetime(self.clock(), field="collector clock")
            status = self.engine.status()
            if status["events"] and at <= timestamp(status["last_command_at"]):
                raise ValueError("collector clock moved backwards")
            if at < self.start_at:
                return dict(phase="waiting", status=self.engine.status())
            if self._finish_if_due(at):
                return dict(phase="finished", status=self.engine.status())
            assert self._control is not None
            self._cycle_id = self._control.begin(at)
            if self.engine.status()["pending"]:
                phase = self._execute_pending()
                return self._complete_cycle(phase)
            if (
                self._rule is None
                or self._rule_at is None
                or (at - self._rule_at).total_seconds() >= 3300
            ):
                self._rule, rules = self._capture(rules=True)
                self._rule_at = timestamp(rules["completed_at"])
                if self._finish_if_due(self.clock()):
                    return self._complete_cycle("finished")
            reference, _ = self._capture(rules=False)
            self._guard()
            at = self.clock()
            if self._finish_if_due(at):
                return self._complete_cycle("finished")
            self.engine.decide(
                f"decide-{uuid.uuid4().hex}",
                market=reference,
                rules=self._rule,
                at=at,
            )
            phase = (
                self._execute_pending()
                if self.engine.status()["pending"]
                else "observed"
            )
            return self._complete_cycle(phase)
        except BaseException as error:
            self._halt(error)
            raise

    def _complete_cycle(self, phase: str) -> dict[str, Any]:
        assert (
            self.engine is not None
            and self._control is not None
            and self._cycle_id is not None
        )
        self._control.finish(self._cycle_id)
        self._cycle_id = None
        return dict(phase=phase, status=self.engine.status())
