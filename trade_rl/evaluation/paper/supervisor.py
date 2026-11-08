"""Single-collector public evidence acquisition bound to source and runtime."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any

from trade_rl._validation import require_aware_datetime
from trade_rl.evaluation.paper.account import timestamp
from trade_rl.evaluation.paper.collection_policy import (
    CADENCE_SECONDS,
    RETRIES_PER_SLOT,
    RETRY_DELAY_SECONDS,
    RETRY_START_WINDOW_SECONDS,
    RULES_HOLD_SECONDS,
    RULES_REFRESH_SECONDS,
    SkipBudgetExceeded,
    classify_failure,
    first_budget_breach,
)
from trade_rl.evaluation.paper.control import CollectionControl
from trade_rl.evaluation.paper.engine import EvidenceRef, PaperEngine, PaperSettings
from trade_rl.evaluation.paper.failure import FAILURE_NAME, write_collection_failure
from trade_rl.evaluation.paper.lock import acquire_collector_lock
from trade_rl.evaluation.paper.predecessor import (
    collection_lineage_problems,
    load_canonical_protocol,
    require_seal_token,
)
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.integrations.binance.forward import capture_forward_snapshot
from trade_rl.integrations.binance.forward_rules import capture_forward_rules

_SCHEMA = "paper_public_collection_v1"
_FAILURE = FAILURE_NAME


def _now() -> datetime:
    return datetime.now(UTC)


def seal_collection(
    root: str | Path,
    *,
    settings: PaperSettings,
    research_plan: dict[str, Any],
    attempt2_seal_token: object | None = None,
    clock: Callable[[], datetime] = _now,
) -> str:
    """Freeze collection identity; the caller owns the separate economic plan.

    A research plan carrying an attempt lineage v3 is refused unless the module-private
    ``attempt2`` token is given, and any lineage the shared classifier rejects is
    refused. This guards against accidental or non-canonical records through the
    supported entry points; it is not authentication against a local operator.
    """
    require_seal_token(research_plan, attempt2_seal_token)
    created = require_aware_datetime(clock(), field="collection seal")
    if created >= settings.start_at:
        raise ValueError("collection must be sealed before its future start")
    if not isinstance(research_plan, dict) or not research_plan:
        raise ValueError("collection requires an explicit research plan")
    return PaperEngine.initialize(
        root,
        settings=settings,
        attempt2_seal_token=attempt2_seal_token,
        study=dict(
            schema=_SCHEMA,
            created_at=created.isoformat(),
            interval_seconds=CADENCE_SECONDS,
            terminal_grace_seconds=180,
            provenance=build_candidate_run_provenance(),
            research_plan=research_plan,
        ),
    )


class PaperCollector:
    """Use as a context manager.

    Applies the pre-registered collection-failure policy (``collection_policy``): a
    Class T acquisition failure is retried inside its slot only before a capture is
    consumed, an unavailable slot is skipped and counted against three budgets, and
    a Class D or Class I failure, a budget breach or any other error permanently
    halts I/O.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        expected_protocol_sha256: str,
        clock: Callable[[], datetime] = _now,
        transport: Any = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.root = Path(root).absolute()
        self.digest = expected_protocol_sha256
        self.clock, self.transport, self.monotonic = clock, transport, monotonic
        self.sleep = sleep
        self.engine: PaperEngine | None = None
        self._lock: sqlite3.Connection | None = None
        self._study: dict[str, Any] = {}
        self._maximum_gap_seconds: float = 180.0
        self._rule: EvidenceRef | None = None
        self._rule_at: datetime | None = None
        self._control: CollectionControl | None = None
        self._cycle_id: int | None = None
        self._slot = 0
        self._slot_start: datetime | None = None
        self._retries_used = 0
        self._attempt_path: str | None = None
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
        protocol = load_canonical_protocol(raw)["protocol"]
        study = protocol["study"]
        if (
            study.get("schema") != _SCHEMA
            or study.get("interval_seconds") != CADENCE_SECONDS
            or study.get("terminal_grace_seconds") != 180
            or not isinstance(study.get("provenance"), dict)
        ):
            raise ValueError("unsupported public collection protocol")
        self._study = study
        self._maximum_gap_seconds = float(protocol["settings"]["maximum_gap_seconds"])
        self.start_at = timestamp(protocol["settings"]["start_at"])
        self.close_at = timestamp(protocol["settings"]["close_at"])
        if timestamp(study["created_at"]) >= self.start_at:
            raise ValueError("collection was not sealed before start")
        # A lineage v3 ``available`` record is re-read like ``attempt2 launch`` does,
        # before the lock is taken, so a refusal writes no failure marker.
        problems = collection_lineage_problems(
            study.get("research_plan"),
            root=self.root,
            start_at=self.start_at,
            now=self.clock,
        )
        if problems:
            raise ValueError(
                "collection lineage cannot be re-verified: " + "; ".join(problems)
            )
        self._lock = acquire_collector_lock(self.root)
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
        extra: dict[str, Any] = {}
        if isinstance(error, SkipBudgetExceeded):
            extra = dict(failure_class="B", budget=error.budget, slot=error.slot)
        else:
            extra = dict(failure_class="D" if classify_failure(error) == "D" else "I")
        record = write_collection_failure(
            self.root,
            protocol_sha256=self.digest,
            error_type=type(error).__name__,
            error=str(error),
            extra=extra,
        )
        if record is None:
            return
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
        self._attempt_path = None
        self._guard()
        name = f"sources/{self.clock():%Y-%m-%d}/{uuid.uuid4().hex}"
        self._attempt_path = name
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

    def _slot_of(self, at: datetime) -> int:
        return math.floor((at - self.start_at).total_seconds() / CADENCE_SECONDS)

    def _register_failure(self, purpose: str, error: BaseException) -> None:
        assert self._control is not None and self._cycle_id is not None
        assert self._attempt_path is not None
        self._control.register_attempt(
            self._cycle_id,
            slot=self._slot,
            purpose=purpose,
            path=self._attempt_path,
            outcome="failed",
            error_type=type(error).__name__,
            error=str(error),
        )

    def _acquire(
        self, *, rules: bool, retry: bool, purpose: str
    ) -> tuple[EvidenceRef, dict[str, Any]] | None:
        """Capture once, retrying a Class T failure only as the policy allows.

        Returns ``None`` when no capture was obtained and the caller must skip or
        proceed without one. A Class D or Class I failure is raised at once. Every
        failed attempt is registered durably before any further attempt, so a
        failure can never pass unregistered as an allowed skip. Only a capture is
        retried here: nothing has been consumed, and a committed decision is never
        followed by a retry (callers pass ``retry=False`` after it).
        """
        assert self._slot_start is not None
        deadline = self._slot_start + timedelta(seconds=RETRY_START_WINDOW_SECONDS)
        while True:
            try:
                return self._capture(rules=rules)
            except Exception as error:
                if classify_failure(error) != "T":
                    raise
                self._register_failure(purpose, error)
                failed_at = self.clock()
                if (
                    not retry
                    or self._retries_used >= RETRIES_PER_SLOT
                    or failed_at + timedelta(seconds=RETRY_DELAY_SECONDS) > deadline
                ):
                    return None
                self._retries_used += 1
                self.sleep(RETRY_DELAY_SECONDS)
                resumed = self.clock()
                if resumed <= failed_at:
                    raise ValueError("collector clock moved backwards") from error
                if resumed > deadline:
                    return None

    def _check_budgets(self, slots: list[int]) -> None:
        breach = first_budget_breach(slots)
        if breach is not None:
            raise SkipBudgetExceeded(breach[1], breach[0])

    def _skip(self, reason: str) -> str:
        """Skip this slot: no command is appended; the skip counts toward the budgets."""
        assert self._control is not None and self._cycle_id is not None
        self._check_budgets(
            self._control.register_skips(self._cycle_id, [self._slot], reason)
        )
        return "skipped"

    def _register_missed_slots(self, recorded_gap: bool) -> None:
        """Slots that never ran, before the first cycle or between two, are skipped.

        A cycle that records the frozen observation gap is already a permanent quality
        failure, so its missed slots are not counted a second time.
        """
        assert self._control is not None and self._cycle_id is not None
        if recorded_gap:
            return
        previous = self._control.previous_cycle_start(self._cycle_id)
        # With no earlier cycle the engine anchors the last command at the screen
        # start, so the slots between slot 0 and this one never ran either.
        first = 1 if previous is None else self._slot_of(previous) + 1
        missed = list(range(first, self._slot))
        if missed:
            self._check_budgets(
                self._control.register_skips(self._cycle_id, missed, "missed_slot")
            )

    def _rules_expired(self, at: datetime) -> bool:
        assert self._rule_at is not None
        return (at - self._rule_at).total_seconds() > RULES_HOLD_SECONDS

    def _ensure_rules(self, at: datetime) -> str | None:
        """Refresh the held rules when due; return a phase if the cycle must end.

        A refresh is due from ``RULES_REFRESH_SECONDS``. While the held rules are
        still inside the one-hour hold, a transient refresh failure proceeds on the
        held rules and consumes no retry and no skip budget; the next slot tries
        again. With no rules, or rules past the hold, the refresh is required: it may
        use the slot's retries and, if it still fails, the slot is skipped. A Class D
        or Class I failure ends the attempt in every case.
        """
        age = None if self._rule_at is None else (at - self._rule_at).total_seconds()
        if age is not None and age < RULES_REFRESH_SECONDS:
            return None
        held = age is not None and age <= RULES_HOLD_SECONDS
        captured = self._acquire(rules=True, retry=not held, purpose="rules")
        if captured is None:
            return None if held else self._skip("rules_unavailable")
        self._rule, rules = captured
        self._rule_at = timestamp(rules["completed_at"])
        return "finished" if self._finish_if_due(self.clock()) else None

    def _execute_pending(self) -> str:
        assert self.engine is not None
        decision_at = timestamp(self.engine.status()["last_command_at"])
        if (self.clock() - decision_at).total_seconds() <= 10:
            # Never retried: a decision is committed, and a retry cannot fit its
            # ten-second window. A failed capture lets the decision expire.
            captured = self._acquire(rules=False, retry=False, purpose="execution")
            if captured is None:
                return self._skip("execution_capture_failed")
            reference, _ = captured
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
            recorded_gap = (
                at - timestamp(status["last_command_at"])
            ).total_seconds() > self._maximum_gap_seconds
            if recorded_gap:
                self.engine.gap(
                    f"observation-gap-{uuid.uuid4().hex}",
                    at=at,
                    reason="observation_gap",
                )
            if self._finish_if_due(at):
                return dict(phase="finished", status=self.engine.status())
            assert self._control is not None
            self._cycle_id = self._control.begin(at)
            self._slot = self._slot_of(at)
            self._slot_start = self.start_at + timedelta(
                seconds=self._slot * CADENCE_SECONDS
            )
            self._retries_used = 0
            self._register_missed_slots(recorded_gap)
            if self.engine.status()["pending"]:
                phase = self._execute_pending()
                return self._complete_cycle(phase)
            ended = self._ensure_rules(at)
            if ended is not None:
                return self._complete_cycle(ended)
            assert self._rule is not None
            # No decision is ever made on rules older than one hour: skip before the
            # network when they already are, and again if the capture outlasts them.
            if self._rules_expired(self.clock()):
                return self._complete_cycle(self._skip("rules_expired"))
            captured = self._acquire(rules=False, retry=True, purpose="snapshot")
            if captured is None:
                return self._complete_cycle(self._skip("snapshot_unavailable"))
            reference, _ = captured
            self._guard()
            at = self.clock()
            if self._finish_if_due(at):
                return self._complete_cycle("finished")
            if self._rules_expired(at):
                assert self._control is not None and self._cycle_id is not None
                self._control.register_attempt(
                    self._cycle_id,
                    slot=self._slot,
                    purpose="snapshot",
                    path=reference.path,
                    outcome="unused",
                    error_type="RulesExpired",
                    error="held rules are older than one hour at decision time",
                )
                return self._complete_cycle(self._skip("rules_expired"))
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
