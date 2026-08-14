"""Fail-closed atomic persistence for causal alpha V3 research evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_causal_alpha_v3_contracts import (
    CausalAlphaV3AdmissionRecord,
    CausalAlphaV3ReplayMetric,
)

ReplayIdentity = tuple[str, str, int]


class CausalAlphaV3RecordStore:
    """Persist immutable research records and reconstruct resume state from disk."""

    def __init__(
        self,
        root: Path,
        *,
        run_manifest_digest: str,
        freeze_digest: str | None = None,
    ) -> None:
        require_sha256(run_manifest_digest, field="V3 record store run manifest digest")
        if freeze_digest is not None:
            require_sha256(freeze_digest, field="V3 record store freeze digest")
        self.root = Path(root)
        self.run_manifest_digest = run_manifest_digest
        self.freeze_digest = freeze_digest

    def write_exact_artifact(
        self,
        relative_path: Path | str,
        payload: Mapping[str, object],
    ) -> Path:
        """Write once or verify an already-existing immutable JSON artifact."""

        destination = self.root / Path(relative_path)
        expected = dict(payload)
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    f"V3 immutable artifact identity drifted at {relative_path}"
                )
            return destination
        if destination.exists():
            raise ValueError(f"V3 immutable artifact path is not a file: {relative_path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(destination, canonical_json_bytes(expected) + b"\n")
        return destination

    def _replay_path(self, metric: CausalAlphaV3ReplayMetric) -> Path:
        return (
            self.root
            / "selection"
            / "records"
            / metric.candidate_digest
            / metric.symbol
            / f"{metric.episode_index}.json"
        )

    def _validate_replay_store_identity(self, metric: CausalAlphaV3ReplayMetric) -> None:
        if metric.run_manifest_digest != self.run_manifest_digest:
            raise ValueError("V3 replay record run manifest identity mismatch")
        if self.freeze_digest is None:
            raise ValueError("V3 replay record store requires a freeze identity")
        if metric.freeze_digest != self.freeze_digest:
            raise ValueError("V3 replay record freeze identity mismatch")

    def write_replay_metric(self, metric: CausalAlphaV3ReplayMetric) -> Path:
        if not isinstance(metric, CausalAlphaV3ReplayMetric):
            raise TypeError("V3 replay store requires CausalAlphaV3ReplayMetric")
        self._validate_replay_store_identity(metric)
        destination = self._replay_path(metric)
        if destination.is_file():
            raw = json.loads(destination.read_text(encoding="utf-8"))
            existing = CausalAlphaV3ReplayMetric.from_payload(raw)
            self._validate_replay_store_identity(existing)
            if existing != metric:
                raise ValueError("V3 replay record conflicts with completed scope")
            return destination
        if destination.exists():
            raise ValueError("V3 replay record path is not a file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            destination,
            canonical_json_bytes(metric.to_payload()) + b"\n",
        )
        return destination

    def load_replay_metrics(
        self,
        *,
        expected_contract_digests: Mapping[ReplayIdentity, str],
    ) -> dict[ReplayIdentity, CausalAlphaV3ReplayMetric]:
        expected = dict(expected_contract_digests)
        for contract_digest in expected.values():
            require_sha256(contract_digest, field="V3 expected replay contract digest")
        records_root = self.root / "selection" / "records"
        if not records_root.is_dir():
            return {}
        result: dict[ReplayIdentity, CausalAlphaV3ReplayMetric] = {}
        for path in sorted(records_root.rglob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            metric = CausalAlphaV3ReplayMetric.from_payload(raw)
            self._validate_replay_store_identity(metric)
            identity = metric.identity
            if identity not in expected:
                raise ValueError("V3 replay record is outside the expected scope")
            if metric.contract_digest != expected[identity]:
                raise ValueError("V3 replay record contract identity drifted")
            if identity in result:
                raise ValueError("V3 replay record scope is duplicated")
            result[identity] = metric
        return result

    def _admission_path(self, record: CausalAlphaV3AdmissionRecord) -> Path:
        return self.root / "admission" / "records" / f"{record.symbol}.json"

    def write_admission_record(self, record: CausalAlphaV3AdmissionRecord) -> Path:
        if not isinstance(record, CausalAlphaV3AdmissionRecord):
            raise TypeError("V3 admission store requires CausalAlphaV3AdmissionRecord")
        if record.run_manifest_digest != self.run_manifest_digest:
            raise ValueError("V3 admission record run manifest identity mismatch")
        if self.freeze_digest is None or record.freeze_digest != self.freeze_digest:
            raise ValueError("V3 admission record freeze identity mismatch")
        destination = self._admission_path(record)
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing != record.to_payload():
                raise ValueError("V3 admission record conflicts with completed symbol")
            return destination
        if destination.exists():
            raise ValueError("V3 admission record path is not a file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(
            destination,
            canonical_json_bytes(record.to_payload()) + b"\n",
        )
        return destination


__all__ = ["CausalAlphaV3RecordStore", "ReplayIdentity"]
