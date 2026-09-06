from pathlib import Path

path = Path("trade_rl/workflows/universal_trade_rl_u2_selection.py")
text = path.read_text(encoding="utf-8")
if "def build_universal_trade_rl_u2_development_selection_evidence(" in text:
    raise SystemExit("final Development Selection API already exists; refusing duplicate patch")

text = text.replace("from typing import Final\n", "from typing import Final, Protocol\n", 1)

import_anchor = (
    "from trade_rl.workflows import universal_trade_rl_u2_contract as u2_contract\n"
    "from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS\n"
)
if import_anchor not in text:
    raise SystemExit("Task4 import anchor missing")
import_block = '''from trade_rl.workflows import universal_trade_rl_u2_contract as u2_contract
from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS
from trade_rl.workflows.universal_trade_rl_u2_development_closure import (
    UniversalTradeRLU2AuthoritativeDevelopmentLock,
    UniversalTradeRLU2FinalCheckpointClosure,
)
from trade_rl.workflows.universal_trade_rl_u2_predevelopment import (
    U2_ADMISSION_STATUS,
    U2_PRODUCTION_STATUS,
    UniversalTradeRLU2DevelopmentLock,
)
'''
text = text.replace(import_anchor, import_block, 1)

anchor = "\n\n__all__ = [\n"
if anchor not in text:
    raise SystemExit("Task4 insertion anchor missing")

block = r'''

U2_DEVELOPMENT_SELECTION_EVIDENCE_SCHEMA: Final = (
    "universal_trade_rl_u2_development_selection_evidence_v1"
)
_U2_FINAL_SELECTION_PRIMARY_CELLS: Final = ("B", "C1", "C2", "D1", "D2")
_U2_FINAL_SELECTION_ROBUSTNESS_SCOPES: Final = ("D1", "D2", "D1+D2")


class _DigestBoundArtifact(Protocol):
    @property
    def digest(self) -> str: ...

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]: ...


def _require_current_artifact_digest(
    artifact: _DigestBoundArtifact,
    *,
    field_name: str,
) -> None:
    require_sha256(artifact.digest, field=f"{field_name} digest")
    expected = content_digest(artifact.to_payload(include_digest=False))
    if artifact.digest != expected:
        raise ValueError(f"{field_name} digest/identity drifted")


def _u2_final_selection_summary_identity_map(
    evidence: UniversalTradeRLU2SeedRobustnessEvidence,
) -> dict[tuple[int, str], str]:
    return {
        (summary.training_seed, summary.cell): summary.digest
        for summary in evidence.summaries
    }


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentSelectionEvidence:
    """Canonical all-AND Development Selection evidence for one frozen U2 generation."""

    u2_contract: u2_contract.UniversalTradeRLU2Contract
    base_lock: UniversalTradeRLU2DevelopmentLock
    development_lock: UniversalTradeRLU2AuthoritativeDevelopmentLock
    checkpoint_closure: UniversalTradeRLU2FinalCheckpointClosure
    primary_cell_gates: tuple[UniversalTradeRLU2PrimarySelectionCellGateEvidence, ...]
    seed_robustness_gates: tuple[UniversalTradeRLU2SeedRobustnessEvidence, ...]
    passed: bool = field(init=False)
    admission_eligible: bool = field(init=False)
    selected_checkpoint_digest: str | None = field(init=False)
    production_eligible: bool = field(init=False)
    schema_version: str = U2_DEVELOPMENT_SELECTION_EVIDENCE_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_DEVELOPMENT_SELECTION_EVIDENCE_SCHEMA:
            raise ValueError("unsupported U2 Development Selection evidence schema")
        if not isinstance(self.u2_contract, u2_contract.UniversalTradeRLU2Contract):
            raise TypeError("U2 final Selection requires the frozen U2 contract")
        if not isinstance(self.base_lock, UniversalTradeRLU2DevelopmentLock):
            raise TypeError("U2 final Selection base Development lock is invalid")
        if not isinstance(
            self.development_lock,
            UniversalTradeRLU2AuthoritativeDevelopmentLock,
        ):
            raise TypeError("U2 final Selection authoritative Development lock is invalid")
        if not isinstance(
            self.checkpoint_closure,
            UniversalTradeRLU2FinalCheckpointClosure,
        ):
            raise TypeError("U2 final Selection checkpoint closure is invalid")

        for artifact, field_name in (
            (self.u2_contract, "U2 final Selection U2 contract"),
            (self.base_lock, "U2 final Selection base Development lock"),
            (self.development_lock, "U2 final Selection Development lock"),
            (self.checkpoint_closure, "U2 final Selection checkpoint closure"),
        ):
            _require_current_artifact_digest(artifact, field_name=field_name)

        gates = self.primary_cell_gates
        if not isinstance(gates, tuple):
            raise TypeError("U2 final Selection primary gates must be an immutable tuple")
        if any(
            not isinstance(gate, UniversalTradeRLU2PrimarySelectionCellGateEvidence)
            for gate in gates
        ):
            raise TypeError("U2 final Selection primary gate is invalid")
        if tuple(gate.cell for gate in gates) != _U2_FINAL_SELECTION_PRIMARY_CELLS:
            raise ValueError(
                "U2 final Selection requires ordered B/C1/C2/D1/D2 primary cell closure"
            )

        current_selection_thresholds_digest = content_digest(
            u2_contract._selection_thresholds_payload()
        )
        if self.u2_contract.selection_thresholds_digest != current_selection_thresholds_digest:
            raise ValueError("U2 final Selection threshold preregistration identity drifted")
        for gate in gates:
            _require_current_artifact_digest(
                gate,
                field_name=f"U2 final Selection primary gate {gate.cell}",
            )
            _require_current_artifact_digest(
                gate.summary,
                field_name=f"U2 final Selection primary summary {gate.cell}",
            )
            if gate.training_seed != u2_contract.U2_PRIMARY_CANDIDATE_SEED:
                raise ValueError("U2 final Selection primary gates must use seed 0")
            if gate.selection_thresholds_digest != self.u2_contract.selection_thresholds_digest:
                raise ValueError("U2 final Selection primary gate threshold identity drifted")

        robustness = self.seed_robustness_gates
        if not isinstance(robustness, tuple):
            raise TypeError(
                "U2 final Selection robustness gates must be an immutable tuple"
            )
        if any(
            not isinstance(result, UniversalTradeRLU2SeedRobustnessEvidence)
            for result in robustness
        ):
            raise TypeError("U2 final Selection robustness gate is invalid")
        if tuple(result.scope for result in robustness) != (
            _U2_FINAL_SELECTION_ROBUSTNESS_SCOPES
        ):
            raise ValueError(
                "U2 final Selection requires ordered D1/D2/D1+D2 robustness closure"
            )

        current_cross_seed_thresholds_digest = content_digest(
            _u2_cross_seed_robustness_thresholds()
        )
        for result in robustness:
            _require_current_artifact_digest(
                result,
                field_name=f"U2 final Selection robustness {result.scope}",
            )
            for summary in result.summaries:
                _require_current_artifact_digest(
                    summary,
                    field_name=(
                        "U2 final Selection robustness summary "
                        f"{result.scope}/{summary.training_seed}/{summary.cell}"
                    ),
                )
            _require_current_artifact_digest(
                result.bootstrap_result,
                field_name=f"U2 final Selection robustness bootstrap {result.scope}",
            )
            if result.training_seeds != U2_TRAINING_SEEDS:
                raise ValueError("U2 final Selection robustness seed closure is invalid")
            if result.robustness_thresholds_digest != current_cross_seed_thresholds_digest:
                raise ValueError(
                    "U2 final Selection robustness threshold identity drifted"
                )

        robustness_by_scope = {result.scope: result for result in robustness}
        d1 = robustness_by_scope["D1"]
        d2 = robustness_by_scope["D2"]
        aggregate = robustness_by_scope["D1+D2"]
        standalone_summary_digests = {
            **_u2_final_selection_summary_identity_map(d1),
            **_u2_final_selection_summary_identity_map(d2),
        }
        aggregate_summary_digests = _u2_final_selection_summary_identity_map(aggregate)
        if aggregate_summary_digests != standalone_summary_digests:
            raise ValueError(
                "U2 final Selection D1+D2 aggregate summary identity does not match D1/D2"
            )
        expected_aggregate_scope_closure = tuple(
            sorted((*d1.scope_closure, *d2.scope_closure))
        )
        if aggregate.scope_closure != expected_aggregate_scope_closure:
            raise ValueError(
                "U2 final Selection D1+D2 aggregate tile identity does not match D1/D2"
            )
        if aggregate.bootstrap_segment_digests != (
            *d1.bootstrap_segment_digests,
            *d2.bootstrap_segment_digests,
        ):
            raise ValueError(
                "U2 final Selection D1+D2 aggregate bootstrap identity does not match D1/D2"
            )

        primary_by_cell = {gate.cell: gate for gate in gates}
        for cell in ("D1", "D2"):
            standalone_digest = standalone_summary_digests.get(
                (u2_contract.U2_PRIMARY_CANDIDATE_SEED, cell)
            )
            if standalone_digest is None:
                raise ValueError(
                    f"U2 final Selection {cell} robustness summary identity is missing"
                )
            if primary_by_cell[cell].summary_digest != standalone_digest:
                raise ValueError(
                    f"U2 final Selection {cell} primary summary identity does not match robustness"
                )

        contract_digest = self.u2_contract.digest
        if not (
            self.base_lock.u2_contract_digest
            == self.development_lock.u2_contract_digest
            == self.checkpoint_closure.u2_contract_digest
            == contract_digest
        ):
            raise ValueError("U2 final Selection U2 contract identity closure mismatch")
        if not (
            self.base_lock.universe_manifest_digest
            == self.development_lock.universe_manifest_digest
            == self.checkpoint_closure.universe_manifest_digest
            == self.u2_contract.universe_manifest_digest
        ):
            raise ValueError("U2 final Selection universe identity closure mismatch")
        if not (
            self.base_lock.u1_contract_digest
            == self.development_lock.u1_contract_digest
            == self.checkpoint_closure.u1_contract_digest
            == self.u2_contract.u1_contract_digest
        ):
            raise ValueError("U2 final Selection U1 contract identity closure mismatch")
        if not (
            self.base_lock.u1_normalizer_digest
            == self.development_lock.u1_normalizer_digest
            == self.checkpoint_closure.normalizer_digest
            == self.u2_contract.u1_normalizer_digest
        ):
            raise ValueError("U2 final Selection normalizer identity closure mismatch")
        if self.checkpoint_closure.time_partition_digest != self.u2_contract.time_partition_digest:
            raise ValueError("U2 final Selection time-partition identity mismatch")
        if self.checkpoint_closure.training_config_digest != self.u2_contract.training_config_digest:
            raise ValueError("U2 final Selection training-config identity mismatch")
        if not (
            self.base_lock.predevelopment_contract_digest
            == self.development_lock.predevelopment_contract_digest
            == self.checkpoint_closure.predevelopment_contract_digest
        ):
            raise ValueError("U2 final Selection pre-development identity closure mismatch")
        if self.development_lock.base_lock_digest != self.base_lock.digest:
            raise ValueError("U2 final Selection Development/base lock identity mismatch")
        if (
            self.development_lock.final_checkpoint_closure_digest
            != self.checkpoint_closure.digest
        ):
            raise ValueError(
                "U2 final Selection Development lock checkpoint closure identity mismatch"
            )
        if self.base_lock.checkpoint_digests != self.checkpoint_closure.checkpoint_digests:
            raise ValueError("U2 final Selection checkpoint mapping identity mismatch")

        checkpoint_seeds = tuple(
            seed for seed, _digest in self.checkpoint_closure.checkpoint_digests
        )
        if checkpoint_seeds != U2_TRAINING_SEEDS:
            raise ValueError("U2 final Selection checkpoint seed closure is not canonical")
        if self.base_lock.development_numeric_open_count != 0:
            raise ValueError("U2 final Selection Development lock open count drifted")
        if self.base_lock.admission_numeric_open_count != 0:
            raise ValueError("U2 final Selection requires Admission numeric open count zero")
        if not (
            self.base_lock.admission_status
            == self.development_lock.admission_status
            == U2_ADMISSION_STATUS
        ):
            raise ValueError("U2 final Selection requires Admission to remain SEALED")
        if not (
            self.u2_contract.production_status
            == self.base_lock.production_status
            == self.development_lock.production_status
            == self.checkpoint_closure.production_status
            == U2_PRODUCTION_STATUS
        ):
            raise ValueError("U2 final Selection requires Production NO-GO")

        passed = all(gate.passed for gate in gates) and all(
            result.passed for result in robustness
        )
        checkpoint_by_seed = dict(self.checkpoint_closure.checkpoint_digests)
        primary_checkpoint_digest = checkpoint_by_seed.get(
            u2_contract.U2_PRIMARY_CANDIDATE_SEED
        )
        if primary_checkpoint_digest is None:
            raise ValueError("U2 final Selection seed-0 final checkpoint is missing")
        require_sha256(
            primary_checkpoint_digest,
            field="U2 final Selection seed-0 checkpoint digest",
        )

        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "admission_eligible", passed)
        object.__setattr__(
            self,
            "selected_checkpoint_digest",
            primary_checkpoint_digest if passed else None,
        )
        object.__setattr__(self, "production_eligible", False)

        expected_digest = content_digest(self.to_payload(include_digest=False))
        if self.digest:
            require_sha256(self.digest, field="U2 Development Selection evidence digest")
            if self.digest != expected_digest:
                raise ValueError("U2 Development Selection evidence digest mismatch")
        object.__setattr__(self, "digest", expected_digest)

    @property
    def primary_cells(self) -> tuple[str, ...]:
        return tuple(gate.cell for gate in self.primary_cell_gates)

    @property
    def robustness_scopes(self) -> tuple[str, ...]:
        return tuple(result.scope for result in self.seed_robustness_gates)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "u2_contract_digest": self.u2_contract.digest,
            "base_lock_digest": self.base_lock.digest,
            "development_lock_digest": self.development_lock.digest,
            "checkpoint_closure_digest": self.checkpoint_closure.digest,
            "primary_cells": self.primary_cells,
            "primary_cell_gate_digests": tuple(
                gate.digest for gate in self.primary_cell_gates
            ),
            "robustness_scopes": self.robustness_scopes,
            "seed_robustness_gate_digests": tuple(
                result.digest for result in self.seed_robustness_gates
            ),
            "passed": self.passed,
            "admission_eligible": self.admission_eligible,
            "selected_checkpoint_digest": self.selected_checkpoint_digest,
            "production_eligible": self.production_eligible,
            "admission_status": U2_ADMISSION_STATUS,
            "production_status": U2_PRODUCTION_STATUS,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def build_universal_trade_rl_u2_development_selection_evidence(
    *,
    u2_contract: u2_contract.UniversalTradeRLU2Contract,
    base_lock: UniversalTradeRLU2DevelopmentLock,
    development_lock: UniversalTradeRLU2AuthoritativeDevelopmentLock,
    checkpoint_closure: UniversalTradeRLU2FinalCheckpointClosure,
    primary_cell_gates: tuple[UniversalTradeRLU2PrimarySelectionCellGateEvidence, ...],
    seed_robustness_gates: tuple[UniversalTradeRLU2SeedRobustnessEvidence, ...],
) -> UniversalTradeRLU2DevelopmentSelectionEvidence:
    """Build final U2 Development Selection evidence without opening Admission."""

    return UniversalTradeRLU2DevelopmentSelectionEvidence(
        u2_contract=u2_contract,
        base_lock=base_lock,
        development_lock=development_lock,
        checkpoint_closure=checkpoint_closure,
        primary_cell_gates=primary_cell_gates,
        seed_robustness_gates=seed_robustness_gates,
    )
'''

text = text.replace(anchor, block + anchor, 1)
exports = (
    '    "U2_DEVELOPMENT_SELECTION_EVIDENCE_SCHEMA",\n'
    '    "UniversalTradeRLU2DevelopmentSelectionEvidence",\n'
    '    "build_universal_trade_rl_u2_development_selection_evidence",\n'
)
text = text.replace("__all__ = [\n", "__all__ = [\n" + exports, 1)
path.write_text(text, encoding="utf-8")
