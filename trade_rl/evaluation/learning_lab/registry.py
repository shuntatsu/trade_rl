"""Append-only JSONL trial registry and the deflated Sharpe ratio over all trials.

Every run of every method is one line: method id, config digest, seed, split, code commit,
metrics, status and a stop reason. This API never deletes existing lines: it only
appends, and each line includes the SHA-256 of its predecessor. :meth:`TrialRegistry.verify`
checks internal chain integrity, rejecting modified, reordered or missing middle lines.
It cannot detect deletion of a complete terminal suffix: without an independently pinned
head hash/count, the remaining prefix is still a valid registry. A stopped method is
shelved with the reason it stopped; its recorded trials still count toward deflated Sharpe.

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
import stat
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Final, cast

try:
    import fcntl
except ImportError:  # Windows cannot silently bypass POSIX registry locking.
    fcntl = None  # type: ignore[assignment]

SCHEMA: Final = "learning_lab_trial_v1"
CHECKPOINT_SCHEMA: Final = "learning_lab_registry_checkpoint_v1"
GENESIS: Final = "0" * 64
STATUSES: Final = ("completed", "shelved", "invalid", "failed")
SHARPE_KEY: Final = "sharpe_per_period"
_EULER_GAMMA: Final = 0.5772156649015329
_HEX64 = re.compile(r"[0-9a-f]{64}")
_NORMAL = NormalDist()


class RegistryError(ValueError):
    """The registry file or a record is inconsistent."""


@dataclass(frozen=True, slots=True)
class RegistryCheckpoint:
    """An externally retainable *claim* about one fully verified registry prefix.

    The caller must authenticate and retain this tuple independently from the
    registry. Merely producing it or storing it alongside the mutable JSONL
    file does not prove that historical attempts are complete.
    """

    trial_count: int
    head_sha256: str

    def __post_init__(self) -> None:
        if type(self.trial_count) is not int or self.trial_count < 0:
            raise RegistryError("checkpoint trial_count must be a nonnegative integer")
        if (
            type(self.head_sha256) is not str
            or _HEX64.fullmatch(self.head_sha256) is None
        ):
            raise RegistryError("checkpoint head_sha256 must be lowercase sha256")
        if (self.trial_count == 0) != (self.head_sha256 == GENESIS):
            raise RegistryError(
                "checkpoint genesis requires zero trials and vice versa"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "trial_count": self.trial_count,
            "head_sha256": self.head_sha256,
        }


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
        if any(
            type(value) is not str or not value
            for value in (self.method_id, self.split, self.code_commit)
        ):
            raise RegistryError(
                "method_id, split and code_commit must be nonempty strings"
            )
        if (
            type(self.config_digest) is not str
            or _HEX64.fullmatch(self.config_digest) is None
        ):
            raise RegistryError("config_digest must be a lowercase sha256")
        if type(self.seed) is not int or self.seed < 0:
            raise RegistryError("seed must be a non-negative int")
        if type(self.status) is not str or self.status not in STATUSES:
            raise RegistryError(f"status must be one of {STATUSES}")
        if self.stop_reason is not None and type(self.stop_reason) is not str:
            raise RegistryError("stop_reason must be a string or None")
        if self.recorded_at is not None and type(self.recorded_at) is not str:
            raise RegistryError("recorded_at must be a string or None")
        if self.status != "completed" and not self.stop_reason:
            raise RegistryError("a non-completed trial needs a stop_reason")
        if not isinstance(self.metrics, Mapping):
            raise RegistryError("metrics must be a mapping")
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
    """One append-only JSONL file with cooperative interprocess POSIX locking."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _require_current_regular_file(self, descriptor: int) -> None:
        """Never mutate a detached inode or a symlink that replaced the opened path."""
        opened = os.fstat(descriptor)
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise RegistryError(
                "registry path is not the locked regular file"
            ) from error
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise RegistryError("registry path is not the locked regular file")

    @contextmanager
    def _locked_descriptor(self, *, write: bool) -> Iterator[int | None]:
        """Serialize record appends and read snapshots on the same file descriptor.

        A separate lock-file could be unlinked/recreated behind another writer;
        locking the actual registry inode avoids that second mutable authority.
        POSIX advisory locking is required: unsupported platforms fail closed.
        """
        if fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            raise RegistryError("registry requires POSIX no-follow file locking")
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = (os.O_RDWR | os.O_APPEND | os.O_CREAT) if write else os.O_RDONLY
        # O_NONBLOCK ensures a malicious FIFO cannot block before fstat rejects it.
        flags |= os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | os.O_NONBLOCK
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileNotFoundError:
            if not write:
                yield None
                return
            raise
        except OSError as error:
            raise RegistryError("cannot safely open registry") from error
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            except OSError as error:
                raise RegistryError("registry file lock unavailable") from error
            self._require_current_regular_file(descriptor)
            yield descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_lines(descriptor: int) -> list[bytes]:
        """Read exactly one locked inode snapshot, not a path-based second read."""
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        raw = b"".join(chunks)
        if raw and not raw.endswith(b"\n"):
            raise RegistryError("the registry does not end with a newline")
        return raw.split(b"\n")[:-1] if raw else []

    @staticmethod
    def _verify_lines(lines: list[bytes]) -> int:
        prev = GENESIS
        for number, line in enumerate(lines):
            try:
                body = json.loads(line)
            except (ValueError, RecursionError) as error:
                raise RegistryError(f"line {number} is not JSON") from error
            if not isinstance(body, dict):
                raise RegistryError(f"line {number} is not a JSON object")
            if body.get("schema") != SCHEMA or body.get("index") != number:
                raise RegistryError(f"line {number} has a wrong schema or index")
            if body.get("prev_sha256") != prev:
                raise RegistryError(f"line {number} breaks the hash chain")
            record_payload = body.get("record")
            if not isinstance(record_payload, dict):
                raise RegistryError(f"line {number} has a malformed trial record")
            try:
                record = TrialRecord.from_dict(record_payload)
                canonical = _line(number, prev, record)
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise RegistryError(
                    f"line {number} has a malformed trial record"
                ) from error
            if canonical != line:
                raise RegistryError(f"line {number} is not in canonical form")
            prev = hashlib.sha256(line).hexdigest()
        return len(lines)

    def _verified_snapshot(self, descriptor: int) -> list[bytes]:
        lines = self._read_lines(descriptor)
        self._verify_lines(lines)
        self._require_current_regular_file(descriptor)
        return lines

    def verify(self) -> int:
        """Check the full chain under a shared snapshot lock."""
        with self._locked_descriptor(write=False) as descriptor:
            return 0 if descriptor is None else len(self._verified_snapshot(descriptor))

    def append(self, record: TrialRecord) -> int:
        """Atomically serialize verification and one append among cooperating writers."""
        # Invalid records must not create even an empty registry artifact.
        _line(0, GENESIS, record)
        with self._locked_descriptor(write=True) as descriptor:
            assert descriptor is not None
            lines = self._read_lines(descriptor)
            count = self._verify_lines(lines)
            prev = hashlib.sha256(lines[-1]).hexdigest() if lines else GENESIS
            payload = _line(count, prev, record) + b"\n"
            self._require_current_regular_file(descriptor)
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise RegistryError("registry append was incomplete")
            os.fsync(descriptor)
            # A renamed/unlinked registry inode cannot be reported as committed.
            self._require_current_regular_file(descriptor)
            return count

    @staticmethod
    def _checkpoint(lines: list[bytes]) -> RegistryCheckpoint:
        return RegistryCheckpoint(
            trial_count=len(lines),
            head_sha256=hashlib.sha256(lines[-1]).hexdigest() if lines else GENESIS,
        )

    def snapshot(
        self,
        *,
        expected_checkpoint: RegistryCheckpoint | None = None,
        allow_extensions: bool = False,
    ) -> tuple[list[TrialRecord], RegistryCheckpoint]:
        """Verify and decode one inode snapshot, optionally checking an external pin.

        Exact mode refuses any changed length. Explicit extension mode accepts
        a longer *valid chain* only when the pinned prefix ends at the same
        digest. Neither mode authenticates the externally supplied checkpoint.
        """
        if expected_checkpoint is not None and not isinstance(
            expected_checkpoint, RegistryCheckpoint
        ):
            raise RegistryError("expected checkpoint must be a RegistryCheckpoint")
        if type(allow_extensions) is not bool:
            raise RegistryError("checkpoint extension mode must be boolean")
        if allow_extensions and expected_checkpoint is None:
            raise RegistryError("checkpoint extensions require an expected checkpoint")
        with self._locked_descriptor(write=False) as descriptor:
            lines = [] if descriptor is None else self._verified_snapshot(descriptor)
            observed = self._checkpoint(lines)
            if expected_checkpoint is not None:
                count = expected_checkpoint.trial_count
                if observed.trial_count < count or (
                    not allow_extensions and observed.trial_count != count
                ):
                    raise RegistryError(
                        "checkpoint trial count does not match registry"
                    )
                pinned_head = (
                    hashlib.sha256(lines[count - 1]).hexdigest() if count else GENESIS
                )
                if pinned_head != expected_checkpoint.head_sha256:
                    raise RegistryError("checkpoint head hash does not match registry")
            records = [
                TrialRecord.from_dict(json.loads(line)["record"]) for line in lines
            ]
            return records, observed

    def checkpoint(self) -> RegistryCheckpoint:
        """Export one verified prefix; its subsequent trust is out-of-band."""
        return self.snapshot()[1]

    def records(self) -> list[TrialRecord]:
        """Verify and decode one consistent locked snapshot."""
        return self.snapshot()[0]


def _finite_statistic(value: object, name: str) -> float:
    """Convert only native JSON numeric scalars, refusing overflow and sentinels."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(cast("int | float", value))
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """SR0: expected maximum Sharpe over typed, finite trial statistics."""
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    variance = _finite_statistic(trial_sharpe_variance, "trial_sharpe_variance")
    if variance < 0.0:
        raise ValueError("trial_sharpe_variance must be nonnegative")
    if n_trials == 1 or variance == 0.0:
        return 0.0
    tail = 1.0 / n_trials
    # float64 loses the probability tail beyond ~1e16 independent trials;
    # NormalDist.inv_cdf(1) would throw StatisticsError or imply infinity.
    high = 1.0 - tail
    lower = 1.0 - tail / math.e
    if not (0.0 < high < 1.0 and 0.0 < lower < 1.0):
        raise ValueError("n_trials exceeds representable Gaussian tail precision")
    value = math.sqrt(variance) * (
        (1.0 - _EULER_GAMMA) * _NORMAL.inv_cdf(high)
        + _EULER_GAMMA * _NORMAL.inv_cdf(lower)
    )
    if not math.isfinite(value):
        raise ValueError("expected maximum Sharpe must be finite")
    return value


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    n_trials: int,
    trial_sharpe_variance: float,
) -> float:
    """Probability that a finite per-period Sharpe beats the expected null maximum."""
    if type(n_obs) is not int or n_obs < 2:
        raise ValueError("n_obs must be an integer of at least two")
    obs = _finite_statistic(n_obs, "n_obs")
    candidate = _finite_statistic(sharpe, "sharpe")
    skew = _finite_statistic(skewness, "skewness")
    kurt = _finite_statistic(kurtosis, "kurtosis")
    benchmark = expected_max_sharpe(n_trials, trial_sharpe_variance)
    try:
        variance_term = 1.0 - skew * candidate + (kurt - 1.0) / 4.0 * candidate**2
        if not math.isfinite(variance_term) or variance_term <= 0.0:
            raise ValueError("Sharpe estimator variance must be positive and finite")
        z = (candidate - benchmark) * math.sqrt(obs - 1.0) / math.sqrt(variance_term)
    except OverflowError as error:
        raise ValueError("deflated Sharpe numerical overflow") from error
    if not math.isfinite(z):
        raise ValueError("deflated Sharpe score must be finite")
    return _NORMAL.cdf(z)


def trial_sharpes(
    records: Iterable[TrialRecord], *, key: str = SHARPE_KEY
) -> list[float]:
    """Missing Sharpe on failures is fine; present invalid Sharpe is not."""
    values: list[float] = []
    for record in records:
        if key in record.metrics:
            values.append(_finite_statistic(record.metrics[key], key))
    return values


def registry_deflated_sharpe(
    records: Iterable[TrialRecord],
    *,
    sharpe: float,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    key: str = SHARPE_KEY,
) -> dict[str, float]:
    """Include *all* registered attempts, refusing overflowed trial dispersion."""
    all_records = list(records)
    sharpes = trial_sharpes(all_records, key=key)
    n_trials = max(1, len(all_records))
    try:
        mean = sum(sharpes) / len(sharpes) if sharpes else 0.0
        variance = (
            sum((s - mean) ** 2 for s in sharpes) / (len(sharpes) - 1)
            if len(sharpes) > 1
            else 0.0
        )
    except OverflowError as error:
        raise ValueError("trial Sharpe variance overflow") from error
    if not math.isfinite(mean) or not math.isfinite(variance):
        raise ValueError("trial Sharpe variance must be finite")
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
    "CHECKPOINT_SCHEMA",
    "GENESIS",
    "SCHEMA",
    "RegistryCheckpoint",
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
