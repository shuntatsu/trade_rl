from __future__ import annotations

from pathlib import Path


def _replace(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one patch location in {path}")
    source.write_text(text.replace(old, new, 1), encoding="utf-8")


_replace(
    "trade_rl/workflows/universal_causal_alpha_v3_contracts.py",
    "from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch\n",
    """from trade_rl.learning.causal_alpha_teacher import CausalAlphaTeacherHoldoutMetric
from trade_rl.learning.episode_oracle_teacher import EpisodeOracleBatch
""",
)

_replace(
    "trade_rl/workflows/universal_causal_alpha_v3_contracts.py",
    """        if include_digest:
            payload[\"artifact_digest\"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaV3TeacherPackage:
""",
    """        if include_digest:
            payload[\"artifact_digest\"] = self.digest
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> CausalAlphaV3AdmissionRecord:
        return cls(
            run_manifest_digest=str(raw[\"run_manifest_digest\"]),
            freeze_digest=str(raw[\"freeze_digest\"]),
            selection_digest=str(raw[\"selection_digest\"]),
            selected_candidate_digest=str(raw[\"selected_candidate_digest\"]),
            symbol=str(raw[\"symbol\"]),
            contract_digest=str(raw[\"contract_digest\"]),
            gross_return=float(raw[\"gross_return\"]),
            net_return=float(raw[\"net_return\"]),
            turnover_per_day=float(raw[\"turnover_per_day\"]),
            total_execution_cost=float(raw[\"total_execution_cost\"]),
            trade_count=int(raw[\"trade_count\"]),
            maximum_drawdown=float(raw[\"maximum_drawdown\"]),
            digest=str(raw[\"artifact_digest\"]),
        )

    def to_holdout_metric(self) -> CausalAlphaTeacherHoldoutMetric:
        return CausalAlphaTeacherHoldoutMetric(
            symbol=self.symbol,
            gross_return=self.gross_return,
            net_return=self.net_return,
            turnover_per_day=self.turnover_per_day,
            total_execution_cost=self.total_execution_cost,
            trade_count=self.trade_count,
            maximum_drawdown=self.maximum_drawdown,
        )


@dataclass(frozen=True, slots=True)
class UniversalCausalAlphaV3TeacherPackage:
""",
)

_replace(
    "trade_rl/workflows/universal_causal_alpha_v3_store.py",
    """        atomic_write_bytes(
            destination,
            canonical_json_bytes(record.to_payload()) + b\"\\n\",
        )
        return destination


__all__ = [\"CausalAlphaV3RecordStore\", \"ReplayIdentity\"]
""",
    """        atomic_write_bytes(
            destination,
            canonical_json_bytes(record.to_payload()) + b\"\\n\",
        )
        return destination

    def load_admission_records(
        self,
        *,
        expected_contract_digests: Mapping[str, str],
        selection_digest: str,
        selected_candidate_digest: str,
    ) -> dict[str, CausalAlphaV3AdmissionRecord]:
        if self.freeze_digest is None:
            raise ValueError(\"V3 admission record store requires a freeze identity\")
        require_sha256(selection_digest, field=\"V3 admission selection digest\")
        require_sha256(
            selected_candidate_digest,
            field=\"V3 admission selected candidate digest\",
        )
        expected = dict(expected_contract_digests)
        if not expected or any(not symbol for symbol in expected):
            raise ValueError(\"V3 admission expected scope must be non-empty\")
        for digest in expected.values():
            require_sha256(digest, field=\"V3 expected admission contract digest\")
        records_root = self.root / \"admission\" / \"records\"
        if not records_root.is_dir():
            return {}
        result: dict[str, CausalAlphaV3AdmissionRecord] = {}
        for path in sorted(records_root.glob(\"*.json\")):
            raw = json.loads(path.read_text(encoding=\"utf-8\"))
            record = CausalAlphaV3AdmissionRecord.from_payload(raw)
            if record.run_manifest_digest != self.run_manifest_digest:
                raise ValueError(\"V3 admission record run manifest identity mismatch\")
            if record.freeze_digest != self.freeze_digest:
                raise ValueError(\"V3 admission record freeze identity mismatch\")
            if record.selection_digest != selection_digest:
                raise ValueError(\"V3 admission record selection identity mismatch\")
            if record.selected_candidate_digest != selected_candidate_digest:
                raise ValueError(\"V3 admission selected candidate identity mismatch\")
            if record.symbol not in expected:
                raise ValueError(\"V3 admission record is outside the expected scope\")
            if record.contract_digest != expected[record.symbol]:
                raise ValueError(\"V3 admission record contract identity drifted\")
            if path.name != f\"{record.symbol}.json\":
                raise ValueError(\"V3 admission record path identity drifted\")
            if record.symbol in result:
                raise ValueError(\"V3 admission record symbol is duplicated\")
            result[record.symbol] = record
        return result


__all__ = [\"CausalAlphaV3RecordStore\", \"ReplayIdentity\"]
""",
)
