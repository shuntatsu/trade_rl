from __future__ import annotations

import re
from pathlib import Path


def _replace_once(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one exact match, found {count}")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


def _sub_once(path: str, pattern: str, replacement: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise SystemExit(f"{path}: expected one regex match, found {count}")
    source.write_text(updated, encoding="utf-8")


COMMON_ADMISSION = r'''_CAUSAL_ALPHA_TEACHER_MINIMUM_SYMBOL_NET_RETURN = -0.05


def _causal_alpha_teacher_admission_summary(
    metrics: tuple[CausalAlphaTeacherHoldoutMetric, ...],
) -> tuple[float, float, int, float, int, tuple[str, ...]]:
    values = tuple(metrics)
    if not values or len({item.symbol for item in values}) != len(values):
        raise ValueError("causal alpha teacher holdout symbols must be unique")
    aggregate_gross = float(sum(item.gross_return for item in values))
    aggregate_net = float(sum(item.net_return for item in values))
    negative_count = sum(item.gross_return < 0.0 for item in values)
    worst_symbol_net = float(min(item.net_return for item in values))
    total_trades = sum(item.trade_count for item in values)
    reasons: list[str] = []
    if aggregate_gross < 0.0:
        reasons.append("negative_aggregate_gross_return")
    if aggregate_net < 0.0:
        reasons.append("negative_aggregate_net_return")
    if negative_count > len(values) // 2:
        reasons.append("majority_negative_gross_holdouts")
    if worst_symbol_net < _CAUSAL_ALPHA_TEACHER_MINIMUM_SYMBOL_NET_RETURN:
        reasons.append("symbol_net_return_below_floor")
    if total_trades == 0:
        reasons.append("no_meaningful_trades")
    return (
        aggregate_gross,
        aggregate_net,
        negative_count,
        worst_symbol_net,
        total_trades,
        tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class CausalAlphaTeacherAdmissionEvidence:
    metrics: tuple[CausalAlphaTeacherHoldoutMetric, ...]
    aggregate_gross_return: float
    aggregate_net_return: float
    negative_gross_symbol_count: int
    worst_symbol_net_return: float
    total_trade_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    digest: str = ""

    def __post_init__(self) -> None:
        metrics = tuple(self.metrics)
        (
            expected_gross,
            expected_net,
            expected_negative_count,
            expected_worst_net,
            expected_total_trades,
            expected_reasons,
        ) = _causal_alpha_teacher_admission_summary(metrics)
        if not isinstance(self.passed, bool):
            raise ValueError("causal alpha teacher admission passed must be boolean")
        if (
            isinstance(self.negative_gross_symbol_count, bool)
            or not isinstance(self.negative_gross_symbol_count, int)
            or isinstance(self.total_trade_count, bool)
            or not isinstance(self.total_trade_count, int)
        ):
            raise ValueError("causal alpha teacher admission counts are invalid")
        if (
            self.aggregate_gross_return != expected_gross
            or self.aggregate_net_return != expected_net
            or self.negative_gross_symbol_count != expected_negative_count
            or self.worst_symbol_net_return != expected_worst_net
            or self.total_trade_count != expected_total_trades
        ):
            raise ValueError("causal alpha teacher admission summary is inconsistent")
        reasons = tuple(self.rejection_reasons)
        if reasons != expected_reasons or self.passed != (not expected_reasons):
            raise ValueError("causal alpha teacher admission reasons are inconsistent")
        expected = content_digest(
            {
                "aggregate_gross_return": self.aggregate_gross_return,
                "aggregate_net_return": self.aggregate_net_return,
                "metric_digests": tuple(item.digest for item in metrics),
                "negative_gross_symbol_count": self.negative_gross_symbol_count,
                "passed": self.passed,
                "rejection_reasons": reasons,
                "schema_version": "causal_alpha_teacher_admission_v2",
                "total_trade_count": self.total_trade_count,
                "worst_symbol_net_return": self.worst_symbol_net_return,
            }
        )
        if self.digest and self.digest != expected:
            raise ValueError("causal alpha teacher admission digest mismatch")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "digest", expected)

    def to_payload(self) -> dict[str, object]:
        return {
            "aggregate_gross_return": self.aggregate_gross_return,
            "aggregate_net_return": self.aggregate_net_return,
            "artifact_digest": self.digest,
            "metrics": [item.to_payload() for item in self.metrics],
            "negative_gross_symbol_count": self.negative_gross_symbol_count,
            "passed": self.passed,
            "rejection_reasons": list(self.rejection_reasons),
            "schema_version": "causal_alpha_teacher_admission_v2",
            "total_trade_count": self.total_trade_count,
            "worst_symbol_net_return": self.worst_symbol_net_return,
        }


def evaluate_causal_alpha_teacher_admission(
    metrics: tuple[CausalAlphaTeacherHoldoutMetric, ...],
) -> CausalAlphaTeacherAdmissionEvidence:
    """Apply the lightweight maintained pre-BC gate to untouched teacher holdouts."""

    values = tuple(metrics)
    (
        aggregate_gross,
        aggregate_net,
        negative_count,
        worst_symbol_net,
        total_trades,
        reasons,
    ) = _causal_alpha_teacher_admission_summary(values)
    return CausalAlphaTeacherAdmissionEvidence(
        metrics=values,
        aggregate_gross_return=aggregate_gross,
        aggregate_net_return=aggregate_net,
        negative_gross_symbol_count=negative_count,
        worst_symbol_net_return=worst_symbol_net,
        total_trade_count=total_trades,
        passed=not reasons,
        rejection_reasons=reasons,
    )
'''

_sub_once(
    "trade_rl/learning/causal_alpha_teacher.py",
    r"@dataclass\(frozen=True, slots=True\)\nclass CausalAlphaTeacherAdmissionEvidence:.*?(?=\n\n__all__ = \[)",
    COMMON_ADMISSION.rstrip(),
)

_replace_once(
    "trade_rl/integrations/universal_pretraining.py",
    '                != "causal_alpha_teacher_admission_v1"\n',
    '                != "causal_alpha_teacher_admission_v2"\n',
)

_replace_once(
    "trade_rl/workflows/universal_causal_alpha_v3_admission.py",
    "from trade_rl.learning.causal_alpha_teacher import CausalAlphaTeacherHoldoutMetric\n",
    "from trade_rl.learning.causal_alpha_teacher import (\n"
    "    CausalAlphaTeacherHoldoutMetric,\n"
    "    evaluate_causal_alpha_teacher_admission,\n"
    ")\n",
)
_replace_once(
    "trade_rl/workflows/universal_causal_alpha_v3_admission.py",
    '_EVIDENCE_SCHEMA: Final = "causal_alpha_v3_admission_evidence_v2"',
    '_EVIDENCE_SCHEMA: Final = "causal_alpha_v3_admission_evidence_v3"',
)

V3_EVIDENCE = r'''@dataclass(frozen=True, slots=True)
class CausalAlphaV3AdmissionEvidenceV3:
    records: tuple[CausalAlphaV3AdmissionRecordV2, ...]
    base_admission_digest: str
    aggregate_gross_return: float
    aggregate_net_return: float
    negative_gross_symbol_count: int
    worst_symbol_net_return: float
    total_trade_count: int
    hard_risk_violation_count: int
    unexplained_execution_rejection_count: int
    passed: bool
    rejection_reasons: tuple[str, ...]
    promotion_eligible: bool = False
    schema_version: str = _EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if not records or len({item.symbol for item in records}) != len(records):
            raise ValueError("V3 admission evidence requires unique symbol records")
        base = evaluate_causal_alpha_teacher_admission(
            tuple(record.to_holdout_metric() for record in records)
        )
        require_sha256(
            self.base_admission_digest,
            field="V3 admission base_admission_digest",
        )
        if self.base_admission_digest != base.digest:
            raise ValueError("V3 admission base evidence identity mismatch")
        if not math.isfinite(self.aggregate_gross_return) or not math.isfinite(
            self.aggregate_net_return
        ) or not math.isfinite(self.worst_symbol_net_return):
            raise ValueError("V3 admission aggregate returns must be finite")
        for name in (
            "negative_gross_symbol_count",
            "total_trade_count",
            "hard_risk_violation_count",
            "unexplained_execution_rejection_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"V3 admission evidence {name} is invalid")
        hard_risk_count = sum(item.hard_risk_violation for item in records)
        unexplained = sum(
            item.unexplained_execution_rejection_count for item in records
        )
        if (
            self.aggregate_gross_return != base.aggregate_gross_return
            or self.aggregate_net_return != base.aggregate_net_return
            or self.negative_gross_symbol_count != base.negative_gross_symbol_count
            or self.worst_symbol_net_return != base.worst_symbol_net_return
            or self.total_trade_count != base.total_trade_count
            or self.hard_risk_violation_count != hard_risk_count
            or self.unexplained_execution_rejection_count != unexplained
        ):
            raise ValueError("V3 admission evidence summary is inconsistent")
        expected_reasons = list(base.rejection_reasons)
        if hard_risk_count:
            expected_reasons.append("hard_risk_violation")
        if unexplained:
            expected_reasons.append("unexplained_execution_rejection")
        reasons = tuple(self.rejection_reasons)
        if not isinstance(self.passed, bool):
            raise ValueError("V3 admission passed must be boolean")
        if reasons != tuple(expected_reasons) or self.passed != (not expected_reasons):
            raise ValueError("V3 admission pass state and rejection reasons disagree")
        if self.promotion_eligible:
            raise ValueError("V3 admission evidence cannot be promotion eligible")
        if self.schema_version != _EVIDENCE_SCHEMA:
            raise ValueError("unsupported V3 admission evidence schema")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "rejection_reasons", reasons)
        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("V3 admission evidence digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "aggregate_gross_return": self.aggregate_gross_return,
            "aggregate_net_return": self.aggregate_net_return,
            "base_admission_digest": self.base_admission_digest,
            "hard_risk_violation_count": self.hard_risk_violation_count,
            "negative_gross_symbol_count": self.negative_gross_symbol_count,
            "passed": self.passed,
            "promotion_eligible": self.promotion_eligible,
            "record_digests": tuple(record.digest for record in self.records),
            "rejection_reasons": self.rejection_reasons,
            "schema_version": self.schema_version,
            "total_trade_count": self.total_trade_count,
            "unexplained_execution_rejection_count": (
                self.unexplained_execution_rejection_count
            ),
            "worst_symbol_net_return": self.worst_symbol_net_return,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def evaluate_causal_alpha_v3_admission_gate(
    records: tuple[CausalAlphaV3AdmissionRecordV2, ...],
) -> CausalAlphaV3AdmissionEvidenceV3:
    """Reuse the maintained economics gate and add V3 execution/risk checks."""

    values = tuple(records)
    if not values or len({item.symbol for item in values}) != len(values):
        raise ValueError("V3 admission requires unique symbol records")
    base = evaluate_causal_alpha_teacher_admission(
        tuple(item.to_holdout_metric() for item in values)
    )
    hard_risk_count = sum(item.hard_risk_violation for item in values)
    unexplained = sum(item.unexplained_execution_rejection_count for item in values)
    reasons = list(base.rejection_reasons)
    if hard_risk_count:
        reasons.append("hard_risk_violation")
    if unexplained:
        reasons.append("unexplained_execution_rejection")
    return CausalAlphaV3AdmissionEvidenceV3(
        records=values,
        base_admission_digest=base.digest,
        aggregate_gross_return=base.aggregate_gross_return,
        aggregate_net_return=base.aggregate_net_return,
        negative_gross_symbol_count=base.negative_gross_symbol_count,
        worst_symbol_net_return=base.worst_symbol_net_return,
        total_trade_count=base.total_trade_count,
        hard_risk_violation_count=hard_risk_count,
        unexplained_execution_rejection_count=unexplained,
        passed=not reasons,
        rejection_reasons=tuple(reasons),
    )
'''

_sub_once(
    "trade_rl/workflows/universal_causal_alpha_v3_admission.py",
    r"@dataclass\(frozen=True, slots=True\)\nclass CausalAlphaV3AdmissionEvidenceV2:.*?(?=\n\n__all__ = \[)",
    V3_EVIDENCE.rstrip(),
)

for path in (
    "trade_rl/workflows/universal_causal_alpha_v3_admission.py",
    "trade_rl/workflows/universal_causal_alpha_v3_artifacts.py",
    "trade_rl/workflows/universal_causal_alpha_v3_pipeline.py",
    "trade_rl/workflows/universal_causal_alpha_v3_replay.py",
    "trade_rl/workflows/universal_causal_alpha_v3_runner.py",
):
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if "CausalAlphaV3AdmissionEvidenceV2" in text:
        source.write_text(
            text.replace(
                "CausalAlphaV3AdmissionEvidenceV2",
                "CausalAlphaV3AdmissionEvidenceV3",
            ),
            encoding="utf-8",
        )

contracts_path = "trade_rl/workflows/universal_causal_alpha_v3_contracts.py"
_replace_once(
    contracts_path,
    "from trade_rl.learning.causal_alpha_teacher import CausalAlphaTeacherHoldoutMetric\n",
    "",
)
_replace_once(
    contracts_path,
    "from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch\n",
    "",
)
for line in (
    '_RUN_MANIFEST_SCHEMA: Final = "causal_alpha_v3_run_manifest_v1"\n',
    '_ADMISSION_RECORD_SCHEMA: Final = "causal_alpha_v3_admission_record_v1"\n',
    '_PACKAGE_SCHEMA: Final = "universal_causal_alpha_v3_teacher_package_v1"\n',
):
    _replace_once(contracts_path, line, "")
_sub_once(
    contracts_path,
    r"@dataclass\(frozen=True, slots=True\)\nclass CausalAlphaV3RunManifest:.*?(?=@dataclass\(frozen=True, slots=True\)\nclass CausalAlphaV3CandidateFreeze:)",
    "",
)
_sub_once(
    contracts_path,
    r"@dataclass\(frozen=True, slots=True\)\nclass CausalAlphaV3AdmissionRecord:.*?(?=\n\n__all__ = \[)",
    "",
)
for exported in (
    '    "CausalAlphaV3AdmissionRecord",\n',
    '    "CausalAlphaV3RunManifest",\n',
    '    "UniversalCausalAlphaV3TeacherPackage",\n',
):
    _replace_once(contracts_path, exported, "")

store_path = "trade_rl/workflows/universal_causal_alpha_v3_store.py"
_replace_once(
    store_path,
    "from trade_rl.workflows.universal_causal_alpha_v3_contracts import (\n"
    "    CausalAlphaV3AdmissionRecord,\n"
    "    CausalAlphaV3ReplayMetric,\n"
    ")\n",
    "from trade_rl.workflows.universal_causal_alpha_v3_contracts import (\n"
    "    CausalAlphaV3ReplayMetric,\n"
    ")\n",
)
_sub_once(
    store_path,
    r"    def _admission_path\(.*?(?=\n\n__all__ = \[)",
    "",
)

runner_engine = "tests/workflows/test_universal_causal_alpha_v3_runner_engine.py"
source = Path(runner_engine)
text = source.read_text(encoding="utf-8")
if "evidence.metrics" in text:
    source.write_text(text.replace("evidence.metrics", "evidence.records"), encoding="utf-8")
else:
    raise SystemExit(f"{runner_engine}: evidence.metrics call is unavailable")

_replace_once(
    "docs/UNIVERSAL_TRAINING.md",
    "Teacher admissionが失敗した場合はBCを開始せず、critic warm startやPPO updateへ進みません。既存のBC reconstruction/economic gateも緩めません。\n",
    "Teacher admissionが失敗した場合はBCを開始せず、critic warm startやPPO updateへ進みません。既存のBC reconstruction/economic gateも緩めません。\n\n"
    "Teacher admissionはBC後のbootstrap gateを複製する統計検定ではなく、未開封holdoutに対する軽量なpre-BC safety gateです。維持対象の共通判定は、aggregate gross returnとafter-cost aggregate net returnが非負、gross-negative symbolが過半数でない、worst-symbol net returnが`-0.05`以上、かつholdout全体のtrade countが正であることを要求します。V3はこの共通経済gateを再利用し、その上にhard-risk violationとunexplained execution rejectionのreject条件だけを追加します。\n",
)

for path in Path("trade_rl").rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    forbidden = (
        "CausalAlphaV3AdmissionEvidenceV2",
        "CausalAlphaV3AdmissionRecord\n",
        "CausalAlphaV3RunManifest\n",
        "UniversalCausalAlphaV3TeacherPackage\n",
    )
    for token in forbidden:
        if token in text:
            raise SystemExit(f"{path}: obsolete token remains: {token!r}")

print("teacher admission hardening patch applied")
