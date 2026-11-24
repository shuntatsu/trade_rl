"""Durable native completed-arm receipts for an explicitly artificial attempt.

The four recovery file slots match the stopped-bundle contract. The synthetic
receipt recipe is deliberately distinct from the fixed historical recipe. A
source report remains a supplied claim, not producer authentication or a
research/market execution permission. Publication assumes an exclusive owner.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tools.tier_d_nonrl.task18_native import (
    NativeArmObservation,
    SyntheticNativeAttempt,
)
from tools.tier_d_nonrl.task18_native_decode import (
    RestoredNativeEvidence,
    TerminalNativeFacts,
    closed_record,
    decode_ledger,
    decode_returns,
    parse_json,
)
from tools.tier_d_nonrl.task18_native_oracle import (
    NativeArmEvidence,
    NativeArmVerification,
    verify_native_arm,
)
from tools.tier_d_nonrl.task18_receipts import (
    ArmReceipt,
    ExternalArmVerificationReport,
    ExternalSourceVerificationReport,
    RunReceipt,
    SourceBinding,
    _finite_number,
    evaluate_run_receipt,
)
from tools.tier_d_nonrl.task18_recipe import Task18Protocol, build_task18_protocol
from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_rename_directory, atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import read_verified_bytes

_PROTOCOL = build_task18_protocol()
_ARMS = {arm.arm_id: arm for arm in _PROTOCOL.arms}
_FILES = ("native-ledger.json", "result.json", "verification.json", "receipt.json")
_MAX_LEDGER = 64 * 1024 * 1024
_MAX_METADATA = 4 * 1024 * 1024
_SCOPE = "SYNTHETIC_NATIVE_CONFORMANCE_ONLY"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class NativeArmFilePin:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_sha256(self.sha256, field="native artifact digest")
        if type(self.path) is not str or not self.path:
            raise ValueError("native artifact path must be nonempty")
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= _MAX_LEDGER:
            raise ValueError("native artifact pin size exceeds limit")


@dataclass(frozen=True, slots=True)
class VerifiedNativeArmArtifact:
    receipt: ArmReceipt
    verification: NativeArmVerification
    pins: tuple[NativeArmFilePin, ...]

    @property
    def execution_authorized(self) -> bool:
        return False


def synthetic_receipt_protocol(
    attempt: SyntheticNativeAttempt, source: SourceBinding
) -> Task18Protocol:
    """Derive a distinct fixed-roster receipt recipe for the pinned artificial input.

    The historical-reference field is the existing receipt evaluator's reference
    slot. In this synthetic recipe it names the artificial fixture, never the
    archived Dataset/protocol. The default historical recipe remains unchanged.
    """
    if type(attempt) is not SyntheticNativeAttempt or type(source) is not SourceBinding:
        raise ValueError(
            "native receipts require exact typed artificial attempt and source"
        )
    binding = attempt.frozen.binding
    if (
        binding.scope != "SYNTHETIC_ONLY"
        or source.commit_sha != binding.source_commit
        or source.manifest_sha256 != binding.source_manifest_sha256
    ):
        raise ValueError("native receipt source does not match the frozen binding")
    source_files = tuple(sorted(source.files, key=lambda x: x.path))
    first_pin = source_files[0]
    reference_digest = content_digest(
        {
            "schema": "task18_artificial_receipt_reference_v1",
            "base_recipe": _PROTOCOL.adapted_protocol_digest,
            "binding": asdict(binding),
            "plan_pin": asdict(attempt.plan_pin),
        }
    )
    return replace(
        _PROTOCOL,
        recipe_schema="task18_synthetic_native_receipts_v1",
        scope=_SCOPE,
        historical_reference=replace(
            _PROTOCOL.historical_reference,
            source_commit=source.commit_sha,
            protocol_source_path=first_pin.path,
            protocol_source_sha256=first_pin.sha256,
            protocol_digest=reference_digest,
            dataset_id=binding.dataset_id,
            manifest_sha256=binding.manifest_sha256,
            arrays_sha256=binding.arrays_sha256,
            source_file_pins=source_files,
        ),
        dataset=replace(_PROTOCOL.dataset, consumed_development_only=False),
        gates=replace(
            _PROTOCOL.gates,
            claim_boundary="Artificial software conformance only; no research result or execution authorization.",
        ),
    )


def _validate_run(attempt: SyntheticNativeAttempt, run: RunReceipt) -> Task18Protocol:
    if type(run) is not RunReceipt or not run.run_id.strip() or run.runtime_completed:
        raise ValueError("native artifact requires a typed incomplete run declaration")
    protocol = synthetic_receipt_protocol(attempt, run.source_binding)
    evaluation = evaluate_run_receipt(run, protocol)
    if not (
        evaluation.declaration_consistent
        and evaluation.source_report_matches
        and evaluation.roster_complete
        and tuple(a.arm_id for a in run.arms) == tuple(a.arm_id for a in protocol.arms)
        and evaluation.required_scenarios_covered
        and evaluation.declared_metrics_well_formed
        and (
            not any(a.status == "COMPLETED" for a in run.arms)
            or evaluation.arm_reports_match
        )
    ):
        raise ValueError(
            "native run declaration or source report differs from pinned artificial recipe"
        )
    # A partially observed initial balance is never an arbitrary number.
    for arm in run.arms:
        if arm.initial_equity_usdt is not None and (
            not _finite_number(arm.initial_equity_usdt)
            or float(arm.initial_equity_usdt)
            != _PROTOCOL.economics.initial_capital_per_symbol_usdt
        ):
            raise ValueError("native run initial equity differs from the recipe")
    return protocol


def new_synthetic_run_receipt(
    attempt: SyntheticNativeAttempt,
    *,
    run_id: str,
    source_binding: SourceBinding,
    source_report: ExternalSourceVerificationReport,
) -> RunReceipt:
    """Declare all 48 untouched artificial slots; no execution is implied."""
    protocol = synthetic_receipt_protocol(attempt, source_binding)
    ref = protocol.historical_reference
    protocol_digest = protocol.adapted_protocol_digest
    source_digest = source_binding.manifest_sha256
    run = RunReceipt(
        run_id=run_id,
        protocol_digest=protocol_digest,
        historical_protocol_digest=ref.protocol_digest,
        dataset_id=ref.dataset_id,
        manifest_sha256=ref.manifest_sha256,
        arrays_sha256=ref.arrays_sha256,
        source_binding=source_binding,
        source_report=source_report,
        runtime_completed=False,
        arms=tuple(
            ArmReceipt(
                arm_id=arm.arm_id,
                pair_id=arm.pair_id,
                role=arm.role,
                configuration=arm.configuration,
                symbol=arm.symbol,
                scenario=arm.scenario,
                protocol_digest=protocol_digest,
                dataset_id=ref.dataset_id,
                manifest_sha256=ref.manifest_sha256,
                arrays_sha256=ref.arrays_sha256,
                source_binding_sha256=source_digest,
                status="NOT_ATTEMPTED",
                initial_equity_usdt=None,
                terminal_equity_usdt=None,
                max_drawdown=None,
                terminal_flat=None,
                pending_order_count=None,
                reason_code="NOT_STARTED",
            )
            for arm in protocol.arms
        ),
    )
    _validate_run(attempt, run)
    return run


def _slot(run: RunReceipt, arm_id: str) -> ArmReceipt:
    if type(arm_id) is not str or arm_id not in _ARMS:
        raise ValueError("unknown native arm identity")
    return next(arm for arm in run.arms if arm.arm_id == arm_id)


def _context(
    attempt: SyntheticNativeAttempt, run: RunReceipt, arm_id: str
) -> dict[str, object]:
    report = run.source_report
    if report is None:
        raise ValueError("missing declared source report")
    return {
        "scope": _SCOPE,
        "run_id": run.run_id,
        "arm_id": arm_id,
        "protocol_digest": run.protocol_digest,
        "dataset_id": run.dataset_id,
        "plan_pin": asdict(attempt.plan_pin),
        "source_report_binding_sha256": report.binding_sha256,
        "execution_authorized": False,
    }


def _result_payload(
    attempt: SyntheticNativeAttempt,
    run: RunReceipt,
    arm_id: str,
    evidence: NativeArmEvidence,
) -> dict[str, object]:
    return {
        "schema": "task18_artificial_native_result_v1",
        **_context(attempt, run, arm_id),
        "terminal": asdict(TerminalNativeFacts.observe(evidence.book)),
        "returns": asdict(evidence.returns),
    }


def _verification_bytes(
    attempt: SyntheticNativeAttempt,
    run: RunReceipt,
    arm_id: str,
    report: NativeArmVerification,
    *,
    ledger_sha256: str,
    result_sha256: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "schema": "task18_artificial_native_verification_v1",
            **_context(attempt, run, arm_id),
            "native_ledger_sha256": ledger_sha256,
            "arm_result_sha256": result_sha256,
            "verification": asdict(report),
            "producer_authenticated": False,
        }
    )


def _receipt(
    run: RunReceipt,
    old: ArmReceipt,
    evidence: NativeArmEvidence,
    report: NativeArmVerification,
    *,
    ledger_sha256: str,
    result_sha256: str,
    verification_sha256: str,
) -> ArmReceipt:
    if (
        not report.passed
        or not report.terminal_flat
        or report.pending_orders != 0
        or not report.cost_oracle_passed
    ):
        raise ValueError("refuse a completion receipt for failed native verification")
    source_report = run.source_report
    if source_report is None:
        raise ValueError("missing source report")
    initial = _PROTOCOL.economics.initial_capital_per_symbol_usdt
    book = evidence.book
    external = ExternalArmVerificationReport(
        run_id=run.run_id,
        protocol_digest=run.protocol_digest,
        source_report_binding_sha256=source_report.binding_sha256,
        arm_id=old.arm_id,
        pair_id=old.pair_id,
        role=old.role,
        configuration=old.configuration,
        symbol=old.symbol,
        scenario=old.scenario,
        dataset_id=run.dataset_id,
        native_ledger_sha256=ledger_sha256,
        arm_result_sha256=result_sha256,
        verification_report_sha256=verification_sha256,
        expected_advances=_PROTOCOL.index_plan.native_advances_per_arm,
        observed_advances=report.advances,
        initial_equity_usdt=initial,
        terminal_equity_usdt=float(book.portfolio_value),
        max_drawdown=float(book.max_drawdown),
        terminal_settlement_complete=True,
        terminal_flat=True,
        pending_order_count=0,
        cost_oracle_passed=True,
    )
    return replace(
        old,
        status="COMPLETED",
        initial_equity_usdt=initial,
        terminal_equity_usdt=external.terminal_equity_usdt,
        max_drawdown=external.max_drawdown,
        terminal_flat=True,
        pending_order_count=0,
        native_ledger_sha256=ledger_sha256,
        arm_result_sha256=result_sha256,
        external_report=external,
        reason_code=None,
        error_code=None,
        progress=None,
    )


def _regular_dir(path: Path) -> Path:
    if ".." in path.parts:
        raise ValueError("native artifact path must not traverse parents")
    path = path.absolute()
    if not path.is_dir() or any(
        p.is_symlink() or p.is_junction() for p in (*path.parents, path)
    ):
        raise ValueError("native artifact directory must be regular and non-aliased")
    return path


def _pins(arm_id: str, raw: dict[str, bytes]) -> tuple[NativeArmFilePin, ...]:
    result = []
    for name in _FILES:
        limit = _MAX_LEDGER if name == "native-ledger.json" else _MAX_METADATA
        if not 0 < len(raw[name]) <= limit:
            raise ValueError("native artifact file exceeds size limit")
        result.append(
            NativeArmFilePin(f"arms/{arm_id}/{name}", _sha(raw[name]), len(raw[name]))
        )
    return tuple(sorted(result, key=lambda pin: pin.path))


def publish_completed_native_arm(
    root: Path,
    *,
    attempt: SyntheticNativeAttempt,
    run: RunReceipt,
    observation: NativeArmObservation,
) -> VerifiedNativeArmArtifact:
    """Publish all four immutable files together, after actual source verification."""
    _validate_run(attempt, run)
    if type(observation) is not NativeArmObservation:
        raise ValueError("expected actual typed native arm observation")
    old = _slot(run, observation.arm_id)
    if old.status != "RUNNING":
        raise ValueError("only an already enrolled RUNNING slot may publish completion")
    root = _regular_dir(root)
    arms = _regular_dir(root / "arms")
    destination = arms / old.arm_id
    if destination.exists() or destination.is_symlink() or destination.is_junction():
        raise FileExistsError("native arm artifact already exists")
    result = observation.result
    report = verify_native_arm(
        attempt.dataset, old.arm_id, result, frozen=attempt.frozen
    )
    if (
        report != observation.verification
        or not report.passed
        or result.ledger_evidence is None
    ):
        raise ValueError("native observation failed source verification")
    ledger_raw = canonical_json_bytes(result.ledger_evidence.to_mapping())
    result_raw = canonical_json_bytes(_result_payload(attempt, run, old.arm_id, result))
    verification_raw = _verification_bytes(
        attempt,
        run,
        old.arm_id,
        report,
        ledger_sha256=_sha(ledger_raw),
        result_sha256=_sha(result_raw),
    )
    receipt = _receipt(
        run,
        old,
        result,
        report,
        ledger_sha256=_sha(ledger_raw),
        result_sha256=_sha(result_raw),
        verification_sha256=_sha(verification_raw),
    )
    raw = {
        "native-ledger.json": ledger_raw,
        "result.json": result_raw,
        "verification.json": verification_raw,
        "receipt.json": canonical_json_bytes(receipt.to_mapping()),
    }
    pins = _pins(old.arm_id, raw)
    staging = Path(tempfile.mkdtemp(prefix=f".{old.arm_id}.staging-", dir=arms))
    try:
        for name in _FILES:
            atomic_write_bytes(staging / name, raw[name])
        atomic_rename_directory(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return VerifiedNativeArmArtifact(receipt, report, pins)


def load_completed_native_arm(
    root: Path,
    *,
    attempt: SyntheticNativeAttempt,
    run: RunReceipt,
    arm_id: str,
    expected_pins: tuple[NativeArmFilePin, ...],
) -> VerifiedNativeArmArtifact:
    """Read pinned bytes and re-run the offline source oracle, not the arm."""
    _validate_run(attempt, run)
    old = _slot(run, arm_id)
    if old.status not in {"RUNNING", "INTERRUPTED", "COMPLETED"}:
        raise ValueError("completed artifact has no eligible original slot")
    expected_paths = {f"arms/{arm_id}/{name}" for name in _FILES}
    if (
        type(expected_pins) is not tuple
        or len(expected_pins) != len(_FILES)
        or any(type(p) is not NativeArmFilePin for p in expected_pins)
        or {p.path for p in expected_pins} != expected_paths
    ):
        raise ValueError("native artifact pin roster differs")
    pins_by_path = {pin.path: pin for pin in expected_pins}
    root = _regular_dir(root)
    directory = _regular_dir(root / "arms" / arm_id)
    if {p.name for p in directory.iterdir()} != set(_FILES):
        raise ValueError("native artifact directory contains foreign or missing files")
    raw = {}
    for name in _FILES:
        pin = pins_by_path[f"arms/{arm_id}/{name}"]
        limit = _MAX_LEDGER if name == "native-ledger.json" else _MAX_METADATA
        if pin.size_bytes > limit:
            raise ValueError("native artifact file exceeds size limit")
        raw[name] = read_verified_bytes(
            root / pin.path,
            expected_digest=pin.sha256,
            expected_size_bytes=pin.size_bytes,
            field=f"native artifact {name}",
        )
    # Every file has been externally verified before any JSON decode.
    try:
        ledger = decode_ledger(raw["native-ledger.json"])
        context = _context(attempt, run, arm_id)
        result_record = closed_record(
            parse_json(raw["result.json"]),
            set(context) | {"schema", "terminal", "returns"},
            field="native result",
        )
        restored = RestoredNativeEvidence(
            TerminalNativeFacts.from_mapping(result_record["terminal"]),
            decode_returns(result_record["returns"]),
            ledger.decisions,
            ledger,
        )
        if (
            canonical_json_bytes(_result_payload(attempt, run, arm_id, restored))
            != raw["result.json"]
        ):
            raise ValueError("native result differs from exact bound declaration")
        report = verify_native_arm(
            attempt.dataset, arm_id, restored, frozen=attempt.frozen
        )
        if not report.passed:
            raise ValueError(f"restored native evidence failed: {report.reason}")
        ledger_sha = _sha(raw["native-ledger.json"])
        result_sha = _sha(raw["result.json"])
        report_raw = _verification_bytes(
            attempt,
            run,
            arm_id,
            report,
            ledger_sha256=ledger_sha,
            result_sha256=result_sha,
        )
        if raw["verification.json"] != report_raw:
            raise ValueError(
                "native verification report differs from recomputed claims"
            )
        receipt = _receipt(
            run,
            old,
            restored,
            report,
            ledger_sha256=ledger_sha,
            result_sha256=result_sha,
            verification_sha256=_sha(report_raw),
        )
        if raw["receipt.json"] != canonical_json_bytes(receipt.to_mapping()):
            raise ValueError("native completion receipt differs from verified evidence")
        if old.status == "COMPLETED" and old != receipt:
            raise ValueError("previously completed native receipt changed")
    except (
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
        ArithmeticError,
        RecursionError,
    ) as exc:
        raise ValueError("invalid native artifact evidence") from exc
    return VerifiedNativeArmArtifact(
        receipt, report, tuple(sorted(expected_pins, key=lambda pin: pin.path))
    )
