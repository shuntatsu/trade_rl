"""Complete V3 persistence graph and single-writer ownership."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Mapping

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_v3_admission import (
    CausalAlphaV3AdmissionRecordV2,
)
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3CandidateEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v3_diagnostics import (
    CausalAlphaV3ReplayDiagnostics,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_v2 import (
    signal_scope_metric_from_payload,
)
from trade_rl.workflows.universal_causal_alpha_v3_store import (
    CausalAlphaV3RecordStore,
)
from trade_rl.workflows.universal_causal_alpha_v3_teacher_artifacts import (
    CausalAlphaV3TeacherBatchArtifact,
    UniversalCausalAlphaV3TeacherPackageV2,
)

SignalIdentity = tuple[str, str, int]
ReplayDiagnosticsIdentity = tuple[str, str, int]


def _safe_segment(value: str, *, field: str) -> str:
    if (
        not value
        or Path(value).name != value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field} is not a safe artifact path segment")
    return value


class CausalAlphaV3RunLock:
    """Exclusive single-writer ownership for a V3 output root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.path = self.root / ".causal-alpha-v3.lock"
        self._token: str | None = None

    def acquire(self) -> CausalAlphaV3RunLock:
        if self._token is not None:
            raise RuntimeError("V3 run lock is already acquired")
        self.root.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}:{uuid.uuid4().hex}"
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise RuntimeError(
                "V3 output root already has an active or unrecovered writer lock"
            ) from error
        os.close(descriptor)
        try:
            atomic_write_bytes(self.path, token.encode("utf-8"))
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self._token = token
        return self

    def release(self) -> None:
        token = self._token
        if token is None:
            return
        try:
            current = self.path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            raise RuntimeError("V3 run lock disappeared before release") from error
        if current != token:
            raise RuntimeError("V3 run lock ownership changed before release")
        self.path.unlink()
        self._token = None

    def __enter__(self) -> CausalAlphaV3RunLock:
        return self.acquire()

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        self.release()


class CausalAlphaV3ArtifactStore(CausalAlphaV3RecordStore):
    """Replay store extended with auditable signal/admission/teacher leaves."""

    def _signal_path(self, metric: CausalAlphaV3SignalScopeMetric) -> Path:
        fit = _safe_segment(metric.fit_config_digest, field="V3 signal fit")
        symbol = _safe_segment(metric.symbol, field="V3 signal symbol")
        return (
            self.root
            / "signal"
            / "records"
            / fit
            / symbol
            / f"{metric.episode_index}.json"
        )

    def write_signal_scope_metric(self, metric: CausalAlphaV3SignalScopeMetric) -> Path:
        if not isinstance(metric, CausalAlphaV3SignalScopeMetric):
            raise TypeError("V3 signal store requires a scope metric")
        return self.write_exact_artifact(
            self._signal_path(metric).relative_to(self.root), metric.to_payload()
        )

    def load_signal_scope_metrics(
        self, *, expected: Mapping[SignalIdentity, str]
    ) -> dict[SignalIdentity, CausalAlphaV3SignalScopeMetric]:
        scopes = dict(expected)
        records_root = self.root / "signal" / "records"
        if not records_root.is_dir():
            return {}
        result: dict[SignalIdentity, CausalAlphaV3SignalScopeMetric] = {}
        for path in sorted(records_root.rglob("*.json")):
            metric = signal_scope_metric_from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
            identity = metric.identity
            if identity not in scopes:
                raise ValueError("V3 signal record is outside the expected scope")
            if metric.contract_digest != scopes[identity]:
                raise ValueError("V3 signal record contract identity drifted")
            if path != self._signal_path(metric):
                raise ValueError("V3 signal record path identity drifted")
            if identity in result:
                raise ValueError("V3 signal record scope is duplicated")
            result[identity] = metric
        return result

    def write_candidate_evidence(
        self, evidence: CausalAlphaV3CandidateEvidence
    ) -> Path:
        if not isinstance(evidence, CausalAlphaV3CandidateEvidence):
            raise TypeError("V3 candidate store requires candidate evidence")
        return self.write_exact_artifact(
            Path("selection") / "candidates" / f"{evidence.candidate.digest}.json",
            evidence.to_payload(),
        )

    def _replay_diagnostics_path(
        self, diagnostics: CausalAlphaV3ReplayDiagnostics
    ) -> Path:
        candidate = _safe_segment(
            diagnostics.candidate_digest, field="V3 diagnostics candidate"
        )
        symbol = _safe_segment(diagnostics.symbol, field="V3 diagnostics symbol")
        return (
            self.root
            / "selection"
            / "diagnostics"
            / candidate
            / symbol
            / f"{diagnostics.episode_index}.json"
        )

    def _validate_replay_diagnostics_identity(
        self, diagnostics: CausalAlphaV3ReplayDiagnostics
    ) -> None:
        if diagnostics.run_manifest_digest != self.run_manifest_digest:
            raise ValueError("V3 diagnostics run manifest identity mismatch")
        if (
            self.freeze_digest is None
            or diagnostics.freeze_digest != self.freeze_digest
        ):
            raise ValueError("V3 diagnostics freeze identity mismatch")
        _safe_segment(diagnostics.symbol, field="V3 diagnostics symbol")

    def write_replay_diagnostics(
        self, diagnostics: CausalAlphaV3ReplayDiagnostics
    ) -> Path:
        if not isinstance(diagnostics, CausalAlphaV3ReplayDiagnostics):
            raise TypeError("V3 diagnostics store requires replay diagnostics")
        self._validate_replay_diagnostics_identity(diagnostics)
        return self.write_exact_artifact(
            self._replay_diagnostics_path(diagnostics).relative_to(self.root),
            diagnostics.to_payload(),
        )

    def load_replay_diagnostics(
        self,
        *,
        expected_replay_metric_digests: Mapping[ReplayDiagnosticsIdentity, str],
    ) -> dict[ReplayDiagnosticsIdentity, CausalAlphaV3ReplayDiagnostics]:
        expected = dict(expected_replay_metric_digests)
        for identity, replay_metric_digest in expected.items():
            candidate_digest, symbol, episode_index = identity
            require_sha256(
                candidate_digest, field="V3 expected diagnostics candidate digest"
            )
            _safe_segment(symbol, field="V3 expected diagnostics symbol")
            if (
                isinstance(episode_index, bool)
                or not isinstance(episode_index, int)
                or episode_index < 0
            ):
                raise ValueError("V3 expected diagnostics episode index is invalid")
            require_sha256(
                replay_metric_digest, field="V3 expected replay metric digest"
            )
        records_root = self.root / "selection" / "diagnostics"
        if not records_root.is_dir():
            return {}
        result: dict[ReplayDiagnosticsIdentity, CausalAlphaV3ReplayDiagnostics] = {}
        for path in sorted(records_root.rglob("*.json")):
            diagnostics = CausalAlphaV3ReplayDiagnostics.from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
            self._validate_replay_diagnostics_identity(diagnostics)
            identity = diagnostics.identity
            if identity not in expected:
                raise ValueError("V3 diagnostics record is outside the expected scope")
            if diagnostics.replay_metric_digest != expected[identity]:
                raise ValueError("V3 diagnostics replay metric identity drifted")
            if path != self._replay_diagnostics_path(diagnostics):
                raise ValueError("V3 diagnostics path identity drifted")
            if identity in result:
                raise ValueError("V3 diagnostics scope is duplicated")
            result[identity] = diagnostics
        return result

    def _admission_v2_path(self, record: CausalAlphaV3AdmissionRecordV2) -> Path:
        symbol = _safe_segment(record.symbol, field="V3 admission symbol")
        return self.root / "admission" / "records" / f"{symbol}.json"

    def write_admission_record_v2(self, record: CausalAlphaV3AdmissionRecordV2) -> Path:
        if record.run_manifest_digest != self.run_manifest_digest:
            raise ValueError("V3 admission record run manifest identity mismatch")
        if self.freeze_digest is None or record.freeze_digest != self.freeze_digest:
            raise ValueError("V3 admission record freeze identity mismatch")
        return self.write_exact_artifact(
            self._admission_v2_path(record).relative_to(self.root), record.to_payload()
        )

    def load_admission_records_v2(
        self,
        *,
        expected_contract_digests: Mapping[str, str],
        selection_digest: str,
        selected_candidate_digest: str,
    ) -> dict[str, CausalAlphaV3AdmissionRecordV2]:
        if self.freeze_digest is None:
            raise ValueError("V3 admission store requires a freeze identity")
        expected = dict(expected_contract_digests)
        records_root = self.root / "admission" / "records"
        if not records_root.is_dir():
            return {}
        result: dict[str, CausalAlphaV3AdmissionRecordV2] = {}
        for path in sorted(records_root.glob("*.json")):
            record = CausalAlphaV3AdmissionRecordV2.from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if record.run_manifest_digest != self.run_manifest_digest:
                raise ValueError("V3 admission record run identity drifted")
            if record.freeze_digest != self.freeze_digest:
                raise ValueError("V3 admission record freeze identity drifted")
            if record.selection_digest != selection_digest:
                raise ValueError("V3 admission record selection identity drifted")
            if record.selected_candidate_digest != selected_candidate_digest:
                raise ValueError("V3 admission selected candidate identity drifted")
            if record.symbol not in expected:
                raise ValueError("V3 admission record is outside expected scope")
            if record.contract_digest != expected[record.symbol]:
                raise ValueError("V3 admission record contract identity drifted")
            if path != self._admission_v2_path(record):
                raise ValueError("V3 admission record path identity drifted")
            if record.symbol in result:
                raise ValueError("V3 admission record symbol is duplicated")
            result[record.symbol] = record
        return result

    def _teacher_batch_path(self, symbol: str) -> Path:
        safe = _safe_segment(symbol, field="V3 teacher symbol")
        return self.root / "teacher" / "batches" / f"{safe}.json"

    def write_teacher_package(
        self, package: UniversalCausalAlphaV3TeacherPackageV2
    ) -> Path:
        if package.run_manifest_digest != self.run_manifest_digest:
            raise ValueError("V3 teacher package run identity mismatch")
        if self.freeze_digest is None or package.freeze_digest != self.freeze_digest:
            raise ValueError("V3 teacher package freeze identity mismatch")
        for symbol in package.train_symbols:
            artifact = package.batch_artifact(symbol)
            self.write_exact_artifact(
                self._teacher_batch_path(symbol).relative_to(self.root),
                artifact.to_payload(),
            )
        return self.write_exact_artifact("teacher/package.json", package.to_payload())

    def load_teacher_package(self) -> UniversalCausalAlphaV3TeacherPackageV2:
        package_path = self.root / "teacher" / "package.json"
        if not package_path.is_file():
            raise ValueError("V3 teacher package artifact is missing")
        raw = json.loads(package_path.read_text(encoding="utf-8"))
        symbols_raw = raw.get("train_symbols")
        artifact_digests = raw.get("batch_artifact_digests")
        if not isinstance(symbols_raw, list | tuple):
            raise ValueError("V3 teacher package train_symbols are invalid")
        if not isinstance(artifact_digests, Mapping):
            raise ValueError("V3 teacher batch artifact digests are invalid")
        symbols = tuple(str(item) for item in symbols_raw)
        batches = {}
        for symbol in symbols:
            path = self._teacher_batch_path(symbol)
            if not path.is_file():
                raise ValueError(f"V3 teacher batch artifact missing for {symbol}")
            artifact = CausalAlphaV3TeacherBatchArtifact.from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if artifact.symbol != symbol:
                raise ValueError("V3 teacher batch symbol identity drifted")
            if artifact.run_manifest_digest != self.run_manifest_digest:
                raise ValueError("V3 teacher batch run identity drifted")
            if (
                self.freeze_digest is None
                or artifact.freeze_digest != self.freeze_digest
            ):
                raise ValueError("V3 teacher batch freeze identity drifted")
            if artifact_digests.get(symbol) != artifact.digest:
                raise ValueError("V3 teacher batch artifact digest drifted")
            batches[symbol] = artifact.batch
        return UniversalCausalAlphaV3TeacherPackageV2.from_payload(raw, batches=batches)


__all__ = [
    "CausalAlphaV3ArtifactStore",
    "CausalAlphaV3RunLock",
    "ReplayDiagnosticsIdentity",
    "SignalIdentity",
]
