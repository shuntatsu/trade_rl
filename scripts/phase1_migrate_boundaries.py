from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def replace_exact(path: str, old: str, new: str, *, expected: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    actual = text.count(old)
    if actual != expected:
        raise RuntimeError(
            f"{path}: expected {expected} occurrence(s) of {old!r}, found {actual}"
        )
    target.write_text(text.replace(old, new), encoding="utf-8")


def remove(path: str) -> None:
    target = ROOT / path
    if not target.is_file():
        raise RuntimeError(f"expected migration source file is missing: {path}")
    target.unlink()


VALIDATION = '''"""Standard-library validation helpers shared by the lean research core."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Final

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")


def require_non_empty(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must be non-empty")
    return normalized


def require_sha256(value: str, *, field: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def require_git_sha(value: str, *, field: str = "git_commit") -> str:
    if not _GIT_SHA_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def require_aware_datetime(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def require_unique_non_empty(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{field} must not be empty")
    normalized = tuple(require_non_empty(value, field=field) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return normalized


__all__ = [
    "require_aware_datetime",
    "require_git_sha",
    "require_non_empty",
    "require_sha256",
    "require_unique_non_empty",
]
'''

CANONICAL = '''"""Standard-library canonical JSON conversion for content identities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _datetime_value(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must be timezone-aware")
    normalized = value.astimezone(UTC).isoformat()
    return normalized.removesuffix("+00:00") + "Z"


def _mapping_value(value: Mapping[object, object]) -> dict[str, JsonValue]:
    converted: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("canonical JSON mapping keys must be strings")
        converted[key] = to_json_value(item)
    return converted


def _dataclass_value(value: object) -> dict[str, JsonValue]:
    dataclass_value = cast(Any, value)
    return {
        field.name: to_json_value(getattr(dataclass_value, field.name))
        for field in fields(dataclass_value)
    }


def to_json_value(value: object) -> JsonValue:
    """Convert a supported object into a deterministic JSON value tree."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON floats must be finite")
        return value
    if isinstance(value, datetime):
        return _datetime_value(value)
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_value(value)
    if isinstance(value, Mapping):
        return _mapping_value(value)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [to_json_value(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode a supported value as stable UTF-8 canonical JSON bytes."""

    normalized = to_json_value(value)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


__all__ = ["JsonScalar", "JsonValue", "canonical_json_bytes", "to_json_value"]
'''

GATE_MODELS = '''"""Evidence-bound evaluation gate records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from trade_rl._validation import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)

Comparator = Literal[">", ">=", "<", "<=", "=="]


def _compare(value: float, comparator: Comparator, threshold: float) -> bool:
    if comparator == ">":
        return value > threshold
    if comparator == ">=":
        return value >= threshold
    if comparator == "<":
        return value < threshold
    if comparator == "<=":
        return value <= threshold
    return value == threshold


@dataclass(frozen=True, slots=True)
class GateCheck:
    """One named gate, optionally derived from immutable metric evidence."""

    name: str
    passed: bool
    mandatory: bool = True
    detail: str | None = None
    metric_name: str | None = None
    observed_value: float | None = None
    comparator: Comparator | None = None
    threshold: float | None = None
    evidence_digest: str | None = None
    implementation_digest: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.name, field="name")
        if self.detail is not None:
            require_non_empty(self.detail, field="detail")
        evidence_fields = (
            self.metric_name,
            self.observed_value,
            self.comparator,
            self.threshold,
            self.evidence_digest,
            self.implementation_digest,
        )
        populated = tuple(value is not None for value in evidence_fields)
        if any(populated) and not all(populated):
            raise ValueError("metric gate evidence fields must be provided together")
        if all(populated):
            assert self.metric_name is not None
            assert self.observed_value is not None
            assert self.comparator is not None
            assert self.threshold is not None
            assert self.evidence_digest is not None
            assert self.implementation_digest is not None
            require_non_empty(self.metric_name, field="metric_name")
            if not math.isfinite(self.observed_value) or not math.isfinite(
                self.threshold
            ):
                raise ValueError("gate metric values must be finite")
            if self.comparator not in {">", ">=", "<", "<=", "=="}:
                raise ValueError("gate comparator is unsupported")
            require_sha256(self.evidence_digest, field="evidence_digest")
            require_sha256(self.implementation_digest, field="implementation_digest")
            expected = _compare(self.observed_value, self.comparator, self.threshold)
            if self.passed != expected:
                raise ValueError("gate passed flag does not match metric comparison")

    @property
    def evidence_bound(self) -> bool:
        return self.evidence_digest is not None

    @classmethod
    def from_metric(
        cls,
        *,
        name: str,
        metric_name: str,
        observed_value: float,
        comparator: Comparator,
        threshold: float,
        evidence_digest: str,
        implementation_digest: str,
        mandatory: bool = True,
        detail: str | None = None,
    ) -> GateCheck:
        return cls(
            name=name,
            passed=_compare(observed_value, comparator, threshold),
            mandatory=mandatory,
            detail=detail,
            metric_name=metric_name,
            observed_value=observed_value,
            comparator=comparator,
            threshold=threshold,
            evidence_digest=evidence_digest,
            implementation_digest=implementation_digest,
        )


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Gate result bound to evaluated dataset and selected policy identity."""

    dataset_id: str
    selected_policy_digest: str | None
    evaluation_digest: str
    passed: bool
    checks: tuple[GateCheck, ...]
    decided_at: datetime
    schema_version: str = "gate_decision_v3"

    def __post_init__(self) -> None:
        require_sha256(self.dataset_id, field="dataset_id")
        if self.selected_policy_digest is not None:
            require_sha256(self.selected_policy_digest, field="selected_policy_digest")
        require_sha256(self.evaluation_digest, field="evaluation_digest")
        if not self.checks:
            raise ValueError("checks must not be empty")
        names = tuple(check.name for check in self.checks)
        if len(set(names)) != len(names):
            raise ValueError("gate check names must be unique")
        require_aware_datetime(self.decided_at, field="decided_at")
        require_non_empty(self.schema_version, field="schema_version")
        mandatory_passed = all(check.passed for check in self.checks if check.mandatory)
        if self.passed != mandatory_passed:
            raise ValueError(
                "gate passed flag must equal the conjunction of mandatory checks"
            )

    @property
    def failed_mandatory_checks(self) -> tuple[GateCheck, ...]:
        return tuple(
            check for check in self.checks if check.mandatory and not check.passed
        )


__all__ = ["Comparator", "GateCheck", "GateDecision"]
'''

GATE_RESOLVE = '''"""Fail-closed resolution of evidence-bound evaluation checks."""

from __future__ import annotations

from datetime import datetime

from trade_rl.evaluation.gates.models import GateCheck, GateDecision


def resolve_gate(
    checks: tuple[GateCheck, ...],
    *,
    dataset_id: str,
    selected_policy_digest: str | None,
    evaluation_digest: str,
    decided_at: datetime,
    require_evidence: bool = False,
) -> GateDecision:
    """Resolve a gate and optionally require recomputable metric evidence."""

    if require_evidence and any(
        check.mandatory and not check.evidence_bound for check in checks
    ):
        raise ValueError("release-eligible mandatory checks require metric evidence")
    passed = all(check.passed for check in checks if check.mandatory)
    return GateDecision(
        dataset_id=dataset_id,
        selected_policy_digest=selected_policy_digest,
        evaluation_digest=evaluation_digest,
        passed=passed,
        checks=checks,
        decided_at=decided_at,
    )


__all__ = ["resolve_gate"]
'''

GATE_INIT = '''"""Evaluation gate models and fail-closed resolution."""

from trade_rl.evaluation.gates.models import GateCheck, GateDecision
from trade_rl.evaluation.gates.resolve import resolve_gate

__all__ = ["GateCheck", "GateDecision", "resolve_gate"]
'''

CANONICAL_TEST = '''from __future__ import annotations

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes


def test_canonical_json_is_stable_and_sorted() -> None:
    value = {"unicode": "日本語", "nested": {"b": 2, "a": 1}}
    expected = '{"nested":{"a":1,"b":2},"unicode":"日本語"}'.encode("utf-8")
    assert canonical_json_bytes(value) == expected


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_canonical_json_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": value})
'''


def main() -> None:
    write("trade_rl/_validation.py", VALIDATION)
    write("trade_rl/artifacts/canonical.py", CANONICAL)
    write("trade_rl/evaluation/gates/models.py", GATE_MODELS)
    write("trade_rl/evaluation/gates/resolve.py", GATE_RESOLVE)
    write("trade_rl/evaluation/gates/__init__.py", GATE_INIT)

    replacements = (
        ("trade_rl/artifacts/verified_file.py", "from trade_rl.domain.common import require_sha256", "from trade_rl._validation import require_sha256"),
        ("trade_rl/data/contracts.py", "from trade_rl.domain.common import require_aware_datetime, require_non_empty", "from trade_rl._validation import require_aware_datetime, require_non_empty"),
        ("trade_rl/data/market.py", "from trade_rl.domain.common import require_sha256, require_unique_non_empty", "from trade_rl._validation import require_sha256, require_unique_non_empty"),
        ("trade_rl/risk/inputs.py", "from trade_rl.domain.common import require_sha256", "from trade_rl._validation import require_sha256"),
        ("trade_rl/simulation/funding_evidence.py", "from trade_rl.domain.common import require_sha256", "from trade_rl._validation import require_sha256"),
        ("trade_rl/simulation/runtime_performance.py", "from trade_rl.domain.common import require_sha256", "from trade_rl._validation import require_sha256"),
        ("trade_rl/simulation/runtime_performance_io.py", "from trade_rl.domain.common import require_sha256", "from trade_rl._validation import require_sha256"),
        ("trade_rl/evaluation/walk_forward/capabilities.py", "from trade_rl.domain.common import require_non_empty, require_sha256", "from trade_rl._validation import require_non_empty, require_sha256"),
        ("trade_rl/evaluation/walk_forward/sealed_test.py", "from trade_rl.domain.common import require_non_empty, require_sha256", "from trade_rl._validation import require_non_empty, require_sha256"),
        ("trade_rl/artifacts/hashing.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("trade_rl/artifacts/__init__.py", "from trade_rl.artifacts.codec import canonical_json_bytes, to_json_value", "from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value"),
        ("trade_rl/data/identity.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("trade_rl/artifacts/store.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("trade_rl/data/artifact_codec.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("trade_rl/simulation/runtime_performance_io.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("trade_rl/simulation/orders.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("trade_rl/simulation/funding_evidence.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("tests/artifacts/test_codec.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("tests/artifacts/test_codec_store_critical_coverage.py", "import trade_rl.artifacts.codec as codec", "import trade_rl.artifacts.canonical as codec"),
        ("tests/data/test_market_artifact.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("tests/simulation/test_orders.py", "from trade_rl.artifacts.codec import canonical_json_bytes", "from trade_rl.artifacts.canonical import canonical_json_bytes"),
        ("tests/evaluation/test_gates.py", "from trade_rl.domain.evaluation import GateCheck", "from trade_rl.evaluation.gates import GateCheck"),
    )
    for path, old, new in replacements:
        replace_exact(path, old, new)

    write("tests/artifacts/test_canonical_json_shared.py", CANONICAL_TEST)

    remove("trade_rl/artifacts/codec.py")
    remove("trade_rl/evaluation/gates.py")
    remove("trade_rl/domain/__init__.py")
    remove("trade_rl/domain/canonical_json.py")
    remove("trade_rl/domain/common.py")
    remove("trade_rl/domain/evaluation.py")

    source_checkout = ROOT / "trade_rl/_source_checkout.py"
    references: list[str] = []
    for path in sorted(ROOT.rglob("*.py")):
        if path == source_checkout:
            continue
        if "source_checkout_root" in path.read_text(encoding="utf-8"):
            references.append(str(path.relative_to(ROOT)))
    if references:
        raise RuntimeError(f"source_checkout_root still has maintained callers: {references}")
    remove("trade_rl/_source_checkout.py")

    leftovers: list[str] = []
    for base in (ROOT / "trade_rl", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            if "tests/architecture" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8")
            if "trade_rl.domain" in text or "trade_rl.artifacts.codec" in text:
                leftovers.append(str(path.relative_to(ROOT)))
    if leftovers:
        raise RuntimeError(f"retired import paths remain: {leftovers}")

    transient_docs = (
        "docs/plans/2026-09-08-lean-package-boundaries-red-evidence.md",
        "docs/plans/2026-09-08-lean-package-boundaries-phase1-status.md",
        "docs/plans/2026-09-08-lean-package-boundaries-ci-note.md",
        "docs/plans/2026-09-08-lean-package-boundaries-phase1-red.md",
        "docs/plans/2026-09-08-lean-package-boundaries-phase1-red-summary.md",
        "docs/plans/2026-09-08-lean-package-boundaries-phase1-red-check.md",
        "docs/plans/2026-09-08-lean-package-boundaries-phase1-red-head.md",
    )
    for relative in transient_docs:
        path = ROOT / relative
        if path.exists():
            path.unlink()

    (ROOT / "scripts/phase1_migrate_boundaries.py").unlink()
    (ROOT / ".github/workflows/phase1-boundary-migration.yml").unlink()


if __name__ == "__main__":
    main()
