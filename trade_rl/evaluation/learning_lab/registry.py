"""Append-only JSONL trial registry and the deflated Sharpe ratio over all trials.

Every run of every method is one line: method id, config digest, seed, split, code commit,
metrics, status and a stop reason. Nothing is deleted: the file is only ever opened for
append, and every line carries the SHA-256 of the previous line, so :meth:`TrialRegistry.verify`
detects an edited, removed or reordered line. A method that stops is shelved with the reason
it stopped; its trials still count toward the deflated Sharpe.

The deflated Sharpe ratio is Bailey and Lopez de Prado (2014): the probability that the best
observed Sharpe exceeds the expected maximum of ``n_trials`` independent skill-less trials,
with the non-normality correction. It is a helper over the registry; it makes no claim by
itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final

SCHEMA: Final = "learning_lab_trial_v1"
GENESIS: Final = "0" * 64
STATUSES: Final = ("completed", "shelved", "invalid", "failed")
SHARPE_KEY: Final = "sharpe_per_period"
_EULER_GAMMA: Final = 0.5772156649015329
_HEX64 = re.compile(r"[0-9a-f]{64}")
_NORMAL = NormalDist()


class RegistryError(ValueError):
    """The registry file or a record is inconsistent."""


def _check_finite_metrics(value: Any, path: str = "metrics") -> None:
    """Reject non-JSON numeric sentinels at any depth before creating a receipt."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegistryError(f"metric {path!r} is not finite")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _check_finite_metrics(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _check_finite_metrics(child, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class TrialRecord:
    method_id: str
    config_digest: str
    seed: int
    split: str
    code_commit: str
    metrics: Mapping[str, Any]
    status: str
    stop_reason: str | None = None
    recorded_at: str | None = None

    def __post_init__(self) -> None:
        if not self.method_id or not self.split or not self.code_commit:
            raise RegistryError("method_id, split and code_commit are required")
        if _HEX64.fullmatch(self.config_digest) is None:
            raise RegistryError("config_digest must be a lowercase sha256")
        if type(self.seed) is not int or self.seed < 0:
            raise RegistryError("seed must be a non-negative int")
        if self.status not in STATUSES:
            raise RegistryError(f"status must be one of {STATUSES}")
        if self.status != "completed" and not self.stop_reason:
            raise RegistryError("a non-completed trial needs a stop_reason")
        _check_finite_metrics(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "config_digest": self.config_digest,
            "seed": self.seed,
            "split": self.split,
            "code_commit": self.code_commit,
            "metrics": dict(self.metrics),
            "status": self.status,
            "stop_reason": self.stop_reason,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrialRecord:
        return cls(
            method_id=payload["method_id"],
            config_digest=payload["config_digest"],
            seed=payload["seed"],
            split=payload["split"],
            code_commit=payload["code_commit"],
            metrics=dict(payload["metrics"]),
            status=payload["status"],
            stop_reason=payload.get("stop_reason"),
            recorded_at=payload.get("recorded_at"),
        )


def _line(index: int, prev: str, record: TrialRecord) -> bytes:
    body = {
        "schema": SCHEMA,
        "index": index,
        "prev_sha256": prev,
        "record": record.to_dict(),
    }
    # TrialRecord is frozen but its metrics mapping can still be mutated after init.
    _check_finite_metrics(record.metrics)
    try:
        return json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RegistryError("registry record is not canonical JSON") from error


class TrialRegistry:
    """One JSONL file; the only write is :meth:`append`."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _read_lines(self) -> list[bytes]:
        if not self.path.exists():
            return []
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise RegistryError("the registry does not end with a newline")
        return raw.split(b"\n")[:-1] if raw else []

    def verify(self) -> int:
        """Check schema, indices and the hash chain; return the number of trials."""
        prev = GENESIS
        for number, line in enumerate(self._read_lines()):
            try:
                body = json.loads(line)
            except ValueError as error:
                raise RegistryError(f"line {number} is not JSON") from error
            if body.get("schema") != SCHEMA or body.get("index") != number:
                raise RegistryError(f"line {number} has a wrong schema or index")
            if body.get("prev_sha256") != prev:
                raise RegistryError(f"line {number} breaks the hash chain")
            record = TrialRecord.from_dict(body["record"])
            if _line(number, prev, record) != line:
                raise RegistryError(f"line {number} is not in canonical form")
            prev = hashlib.sha256(line).hexdigest()
        return len(self._read_lines())

    def append(self, record: TrialRecord) -> int:
        """Append one trial after verifying the whole chain; return its index."""
        lines = self._read_lines()
        count = self.verify() if lines else 0
        prev = hashlib.sha256(lines[-1]).hexdigest() if lines else GENESIS
        data = _line(count, prev, record) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0),
        )
        try:
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return count

    def records(self) -> list[TrialRecord]:
        self.verify()
        return [
            TrialRecord.from_dict(json.loads(line)["record"])
            for line in self._read_lines()
        ]


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """``SR0``: expected maximum Sharpe of ``n_trials`` skill-less trials."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if trial_sharpe_variance < 0.0 or not math.isfinite(trial_sharpe_variance):
        raise ValueError("trial_sharpe_variance must be finite and non-negative")
    if n_trials == 1 or trial_sharpe_variance == 0.0:
        return 0.0
    return math.sqrt(trial_sharpe_variance) * (
        (1.0 - _EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
        + _EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    n_trials: int,
    trial_sharpe_variance: float,
) -> float:
    """Probability that ``sharpe`` (per period) beats the expected best of ``n_trials``."""
    if n_obs < 2:
        raise ValueError("n_obs must be at least 2")
    benchmark = expected_max_sharpe(n_trials, trial_sharpe_variance)
    variance_term = 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0.0:
        raise ValueError("the Sharpe estimator variance is not positive")
    z = (sharpe - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(variance_term)
    return _NORMAL.cdf(z)


def trial_sharpes(
    records: Iterable[TrialRecord], *, key: str = SHARPE_KEY
) -> list[float]:
    return [
        float(record.metrics[key])
        for record in records
        if key in record.metrics and math.isfinite(float(record.metrics[key]))
    ]


def registry_deflated_sharpe(
    records: Iterable[TrialRecord],
    *,
    sharpe: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    key: str = SHARPE_KEY,
) -> dict[str, float]:
    """Deflated Sharpe of one candidate against *every* registered trial.

    ``n_trials`` counts all records (failed and shelved ones too); the variance is taken
    over the per-period Sharpes the records carry (ddof 1; zero with fewer than two).
    """
    all_records = list(records)
    sharpes = trial_sharpes(all_records, key=key)
    n_trials = max(1, len(all_records))
    mean = sum(sharpes) / len(sharpes) if sharpes else 0.0
    variance = (
        sum((s - mean) ** 2 for s in sharpes) / (len(sharpes) - 1)
        if len(sharpes) > 1
        else 0.0
    )
    return {
        "n_trials": float(n_trials),
        "trial_sharpe_variance": variance,
        "expected_max_sharpe": expected_max_sharpe(n_trials, variance),
        "deflated_sharpe": deflated_sharpe_ratio(
            sharpe,
            n_obs=n_obs,
            skewness=skewness,
            kurtosis=kurtosis,
            n_trials=n_trials,
            trial_sharpe_variance=variance,
        ),
    }


__all__ = [
    "GENESIS",
    "SCHEMA",
    "SHARPE_KEY",
    "STATUSES",
    "RegistryError",
    "TrialRecord",
    "TrialRegistry",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "registry_deflated_sharpe",
    "trial_sharpes",
]
