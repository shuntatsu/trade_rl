"""Content-addressed paper-trading reconciliation evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import require_aware_datetime, require_sha256

PAPER_RECONCILIATION_SCHEMA = "paper_reconciliation_evidence_v1"
PAPER_RECONCILIATION_FILE_NAME = "paper-reconciliation.json"
_RELEASE_MAXIMUM_DIFFERENCE_FRACTION = 1e-6


def _utc(value: datetime, *, field: str) -> datetime:
    return require_aware_datetime(value, field=field).astimezone(UTC)


def _non_negative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _fraction(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{field} must be finite and between 0 and 1")
    return resolved


def _conditions(
    *,
    submitted_order_count: int,
    terminal_order_count: int,
    observed_fill_count: int,
    matched_fill_count: int,
    unknown_order_fill_count: int,
    duplicate_fill_count: int,
    open_order_count: int,
    maximum_position_notional_difference_fraction: float,
    maximum_cash_difference_fraction: float,
    maximum_equity_difference_fraction: float,
    position_notional_tolerance_fraction: float,
    cash_tolerance_fraction: float,
    equity_tolerance_fraction: float,
) -> dict[str, bool]:
    return {
        "order_counts_consistent": (
            terminal_order_count + open_order_count == submitted_order_count
        ),
        "terminal_order_coverage_complete": (
            terminal_order_count == submitted_order_count and open_order_count == 0
        ),
        "fill_counts_consistent": matched_fill_count <= observed_fill_count,
        "fill_matching_complete": (
            matched_fill_count == observed_fill_count
            and unknown_order_fill_count == 0
            and duplicate_fill_count == 0
        ),
        "position_notional_reconciled": (
            maximum_position_notional_difference_fraction
            <= position_notional_tolerance_fraction
        ),
        "cash_reconciled": maximum_cash_difference_fraction <= cash_tolerance_fraction,
        "equity_reconciled": (
            maximum_equity_difference_fraction <= equity_tolerance_fraction
        ),
    }


@dataclass(frozen=True, slots=True)
class PaperReconciliationEvidence:
    """Sealed summary of normalized paper order, fill, and accounting replay."""

    evidence_digest: str
    dataset_id: str
    environment_digest: str
    policy_digest: str
    training_run_digest: str
    start_time: datetime
    end_time: datetime
    created_at: datetime
    order_log_digest: str
    fill_log_digest: str
    submitted_order_count: int
    terminal_order_count: int
    observed_fill_count: int
    matched_fill_count: int
    unknown_order_fill_count: int
    duplicate_fill_count: int
    open_order_count: int
    maximum_position_notional_difference_fraction: float
    maximum_cash_difference_fraction: float
    maximum_equity_difference_fraction: float
    position_notional_tolerance_fraction: float
    cash_tolerance_fraction: float
    equity_tolerance_fraction: float
    passed: bool
    sealed: bool = True
    schema_version: str = PAPER_RECONCILIATION_SCHEMA

    def __post_init__(self) -> None:
        for field, value in (
            ("evidence_digest", self.evidence_digest),
            ("dataset_id", self.dataset_id),
            ("environment_digest", self.environment_digest),
            ("policy_digest", self.policy_digest),
            ("training_run_digest", self.training_run_digest),
            ("order_log_digest", self.order_log_digest),
            ("fill_log_digest", self.fill_log_digest),
        ):
            require_sha256(value, field=field)
        for field in ("start_time", "end_time", "created_at"):
            object.__setattr__(self, field, _utc(getattr(self, field), field=field))
        if self.end_time <= self.start_time:
            raise ValueError("paper reconciliation end_time must follow start_time")
        if self.created_at < self.end_time:
            raise ValueError(
                "paper reconciliation created_at must not precede collection end"
            )
        if self.sealed is not True:
            raise ValueError("paper reconciliation evidence must be sealed")
        if self.schema_version != PAPER_RECONCILIATION_SCHEMA:
            raise ValueError("unsupported paper reconciliation evidence schema")
        if not isinstance(self.passed, bool):
            raise ValueError("paper reconciliation passed must be a boolean")
        for field in (
            "submitted_order_count",
            "terminal_order_count",
            "observed_fill_count",
            "matched_fill_count",
            "unknown_order_fill_count",
            "duplicate_fill_count",
            "open_order_count",
        ):
            _non_negative_integer(getattr(self, field), field=field)
        for field in (
            "maximum_position_notional_difference_fraction",
            "maximum_cash_difference_fraction",
            "maximum_equity_difference_fraction",
            "position_notional_tolerance_fraction",
            "cash_tolerance_fraction",
            "equity_tolerance_fraction",
        ):
            object.__setattr__(
                self,
                field,
                _fraction(getattr(self, field), field=field),
            )
        expected_passed = all(self.conditions().values())
        if self.passed is not expected_passed:
            raise ValueError("paper reconciliation passed does not match observations")
        if self.evidence_digest != content_digest(self.digest_payload()):
            raise ValueError("paper reconciliation evidence digest mismatch")

    @classmethod
    def create(
        cls,
        *,
        dataset_id: str,
        environment_digest: str,
        policy_digest: str,
        training_run_digest: str,
        start_time: datetime,
        end_time: datetime,
        created_at: datetime,
        order_log_digest: str,
        fill_log_digest: str,
        submitted_order_count: int,
        terminal_order_count: int,
        observed_fill_count: int,
        matched_fill_count: int,
        unknown_order_fill_count: int,
        duplicate_fill_count: int,
        open_order_count: int,
        maximum_position_notional_difference_fraction: float,
        maximum_cash_difference_fraction: float,
        maximum_equity_difference_fraction: float,
        position_notional_tolerance_fraction: float,
        cash_tolerance_fraction: float,
        equity_tolerance_fraction: float,
    ) -> PaperReconciliationEvidence:
        start = _utc(start_time, field="start_time")
        end = _utc(end_time, field="end_time")
        created = _utc(created_at, field="created_at")
        submitted_orders = _non_negative_integer(
            submitted_order_count, field="submitted_order_count"
        )
        terminal_orders = _non_negative_integer(
            terminal_order_count, field="terminal_order_count"
        )
        observed_fills = _non_negative_integer(
            observed_fill_count, field="observed_fill_count"
        )
        matched_fills = _non_negative_integer(
            matched_fill_count, field="matched_fill_count"
        )
        unknown_fills = _non_negative_integer(
            unknown_order_fill_count, field="unknown_order_fill_count"
        )
        duplicate_fills = _non_negative_integer(
            duplicate_fill_count, field="duplicate_fill_count"
        )
        open_orders = _non_negative_integer(open_order_count, field="open_order_count")
        maximum_position_difference = _fraction(
            maximum_position_notional_difference_fraction,
            field="maximum_position_notional_difference_fraction",
        )
        maximum_cash_difference = _fraction(
            maximum_cash_difference_fraction,
            field="maximum_cash_difference_fraction",
        )
        maximum_equity_difference = _fraction(
            maximum_equity_difference_fraction,
            field="maximum_equity_difference_fraction",
        )
        position_tolerance = _fraction(
            position_notional_tolerance_fraction,
            field="position_notional_tolerance_fraction",
        )
        cash_tolerance = _fraction(
            cash_tolerance_fraction,
            field="cash_tolerance_fraction",
        )
        equity_tolerance = _fraction(
            equity_tolerance_fraction,
            field="equity_tolerance_fraction",
        )
        passed = all(
            _conditions(
                submitted_order_count=submitted_orders,
                terminal_order_count=terminal_orders,
                observed_fill_count=observed_fills,
                matched_fill_count=matched_fills,
                unknown_order_fill_count=unknown_fills,
                duplicate_fill_count=duplicate_fills,
                open_order_count=open_orders,
                maximum_position_notional_difference_fraction=(
                    maximum_position_difference
                ),
                maximum_cash_difference_fraction=maximum_cash_difference,
                maximum_equity_difference_fraction=maximum_equity_difference,
                position_notional_tolerance_fraction=position_tolerance,
                cash_tolerance_fraction=cash_tolerance,
                equity_tolerance_fraction=equity_tolerance,
            ).values()
        )
        payload: dict[str, object] = {
            "cash_tolerance_fraction": cash_tolerance,
            "created_at": created,
            "dataset_id": dataset_id,
            "duplicate_fill_count": duplicate_fills,
            "end_time": end,
            "environment_digest": environment_digest,
            "equity_tolerance_fraction": equity_tolerance,
            "fill_log_digest": fill_log_digest,
            "matched_fill_count": matched_fills,
            "maximum_cash_difference_fraction": maximum_cash_difference,
            "maximum_equity_difference_fraction": maximum_equity_difference,
            "maximum_position_notional_difference_fraction": (
                maximum_position_difference
            ),
            "observed_fill_count": observed_fills,
            "open_order_count": open_orders,
            "order_log_digest": order_log_digest,
            "passed": passed,
            "policy_digest": policy_digest,
            "position_notional_tolerance_fraction": position_tolerance,
            "schema_version": PAPER_RECONCILIATION_SCHEMA,
            "sealed": True,
            "start_time": start,
            "submitted_order_count": submitted_orders,
            "terminal_order_count": terminal_orders,
            "training_run_digest": training_run_digest,
            "unknown_order_fill_count": unknown_fills,
        }
        return cls(
            evidence_digest=content_digest(payload),
            dataset_id=dataset_id,
            environment_digest=environment_digest,
            policy_digest=policy_digest,
            training_run_digest=training_run_digest,
            start_time=start,
            end_time=end,
            created_at=created,
            order_log_digest=order_log_digest,
            fill_log_digest=fill_log_digest,
            submitted_order_count=submitted_orders,
            terminal_order_count=terminal_orders,
            observed_fill_count=observed_fills,
            matched_fill_count=matched_fills,
            unknown_order_fill_count=unknown_fills,
            duplicate_fill_count=duplicate_fills,
            open_order_count=open_orders,
            maximum_position_notional_difference_fraction=(maximum_position_difference),
            maximum_cash_difference_fraction=maximum_cash_difference,
            maximum_equity_difference_fraction=maximum_equity_difference,
            position_notional_tolerance_fraction=position_tolerance,
            cash_tolerance_fraction=cash_tolerance,
            equity_tolerance_fraction=equity_tolerance,
            passed=passed,
        )

    def conditions(self) -> dict[str, bool]:
        return _conditions(
            submitted_order_count=self.submitted_order_count,
            terminal_order_count=self.terminal_order_count,
            observed_fill_count=self.observed_fill_count,
            matched_fill_count=self.matched_fill_count,
            unknown_order_fill_count=self.unknown_order_fill_count,
            duplicate_fill_count=self.duplicate_fill_count,
            open_order_count=self.open_order_count,
            maximum_position_notional_difference_fraction=(
                self.maximum_position_notional_difference_fraction
            ),
            maximum_cash_difference_fraction=self.maximum_cash_difference_fraction,
            maximum_equity_difference_fraction=self.maximum_equity_difference_fraction,
            position_notional_tolerance_fraction=(
                self.position_notional_tolerance_fraction
            ),
            cash_tolerance_fraction=self.cash_tolerance_fraction,
            equity_tolerance_fraction=self.equity_tolerance_fraction,
        )

    def digest_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("evidence_digest")
        return payload

    def require_promotable(self) -> None:
        if not self.passed:
            raise ValueError("paper reconciliation did not pass")
        for field in (
            "position_notional_tolerance_fraction",
            "cash_tolerance_fraction",
            "equity_tolerance_fraction",
        ):
            if getattr(self, field) > _RELEASE_MAXIMUM_DIFFERENCE_FRACTION:
                raise ValueError(
                    f"paper reconciliation release tolerance exceeds "
                    f"{_RELEASE_MAXIMUM_DIFFERENCE_FRACTION:g}: {field}"
                )


def write_paper_reconciliation_evidence(
    path: str | Path,
    evidence: PaperReconciliationEvidence,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(asdict(evidence))
    if output.exists():
        if output.read_bytes() != encoded:
            raise FileExistsError(
                "refusing to overwrite immutable paper reconciliation evidence"
            )
        return output
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    return output


def load_paper_reconciliation_evidence(
    path: str | Path,
) -> PaperReconciliationEvidence:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("paper reconciliation evidence must be an object")
    expected = {
        "cash_tolerance_fraction",
        "created_at",
        "dataset_id",
        "duplicate_fill_count",
        "end_time",
        "environment_digest",
        "equity_tolerance_fraction",
        "evidence_digest",
        "fill_log_digest",
        "matched_fill_count",
        "maximum_cash_difference_fraction",
        "maximum_equity_difference_fraction",
        "maximum_position_notional_difference_fraction",
        "observed_fill_count",
        "open_order_count",
        "order_log_digest",
        "passed",
        "policy_digest",
        "position_notional_tolerance_fraction",
        "schema_version",
        "sealed",
        "start_time",
        "submitted_order_count",
        "terminal_order_count",
        "training_run_digest",
        "unknown_order_fill_count",
    }
    if set(raw) != expected:
        raise ValueError("paper reconciliation evidence fields are invalid")
    try:
        return PaperReconciliationEvidence(
            evidence_digest=_string(raw, "evidence_digest"),
            dataset_id=_string(raw, "dataset_id"),
            environment_digest=_string(raw, "environment_digest"),
            policy_digest=_string(raw, "policy_digest"),
            training_run_digest=_string(raw, "training_run_digest"),
            start_time=_datetime(raw, "start_time"),
            end_time=_datetime(raw, "end_time"),
            created_at=_datetime(raw, "created_at"),
            order_log_digest=_string(raw, "order_log_digest"),
            fill_log_digest=_string(raw, "fill_log_digest"),
            submitted_order_count=_integer(raw, "submitted_order_count"),
            terminal_order_count=_integer(raw, "terminal_order_count"),
            observed_fill_count=_integer(raw, "observed_fill_count"),
            matched_fill_count=_integer(raw, "matched_fill_count"),
            unknown_order_fill_count=_integer(raw, "unknown_order_fill_count"),
            duplicate_fill_count=_integer(raw, "duplicate_fill_count"),
            open_order_count=_integer(raw, "open_order_count"),
            maximum_position_notional_difference_fraction=_number(
                raw, "maximum_position_notional_difference_fraction"
            ),
            maximum_cash_difference_fraction=_number(
                raw, "maximum_cash_difference_fraction"
            ),
            maximum_equity_difference_fraction=_number(
                raw, "maximum_equity_difference_fraction"
            ),
            position_notional_tolerance_fraction=_number(
                raw, "position_notional_tolerance_fraction"
            ),
            cash_tolerance_fraction=_number(raw, "cash_tolerance_fraction"),
            equity_tolerance_fraction=_number(raw, "equity_tolerance_fraction"),
            passed=_boolean(raw, "passed"),
            sealed=_boolean(raw, "sealed"),
            schema_version=_string(raw, "schema_version"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("paper reconciliation evidence is invalid") from error


def _string(raw: Mapping[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer(raw: Mapping[str, object], field: str) -> int:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(raw: Mapping[str, object], field: str) -> float:
    value = raw[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return float(value)


def _boolean(raw: Mapping[str, object], field: str) -> bool:
    value = raw[field]
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _datetime(raw: Mapping[str, object], field: str) -> datetime:
    try:
        return datetime.fromisoformat(_string(raw, field).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from error


__all__ = [
    "PAPER_RECONCILIATION_FILE_NAME",
    "PAPER_RECONCILIATION_SCHEMA",
    "PaperReconciliationEvidence",
    "load_paper_reconciliation_evidence",
    "write_paper_reconciliation_evidence",
]
