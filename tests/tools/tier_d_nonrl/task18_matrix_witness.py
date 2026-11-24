"""Closed artificial 48-arm witness. Explicit invocation only, never market input.

A single exclusive local parent enrolls a bounded batch, gives disjoint arm
directories to workers and promotes only verified native completion receipts.
No remote data, arbitrary Dataset, research permission or distributed lease.
Interrupted batches remain stopped evidence; this witness does not retry them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from pathlib import Path

from tests.tools.tier_d_nonrl.test_task18_native_recovery import _fixture
from tools.tier_d_nonrl import task18_recovery as recovery
from tools.tier_d_nonrl.task18_frozen_inputs import (
    FrozenInputsPin,
    prepare_frozen_inputs,
    publish_frozen_inputs,
)
from tools.tier_d_nonrl.task18_native import (
    SyntheticNativeAttempt,
    open_synthetic_native_attempt,
)
from tools.tier_d_nonrl.task18_native_artifact import (
    NativeArmFilePin,
    VerifiedNativeArmArtifact,
    _validate_run,
    load_completed_native_arm,
    new_synthetic_run_receipt,
    publish_completed_native_arm,
    synthetic_receipt_protocol,
)
from tools.tier_d_nonrl.task18_receipts import (
    ArmReceipt,
    RunReceipt,
    evaluate_run_receipt,
)
from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl._validation import require_sha256
from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.verified_file import file_digest_and_size, open_regular_binary

_ROOT = Path(__file__).resolve().parents[3]
_PROTOCOL = build_task18_protocol()
_IDS = tuple(a.arm_id for a in _PROTOCOL.arms)
_MAX_METADATA = 4 * 1024 * 1024
_MAX_BATCH = 12
_MAX_WORKERS = 4
_SCOPE = "SYNTHETIC_NATIVE_MATRIX_WITNESS_ONLY"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _clean_head() -> str:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()

    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("matrix witness requires an exact clean source checkout")
    return git("rev-parse", "HEAD")


def _read_pinned(root: Path, name: str, digest: str) -> bytes:
    require_sha256(digest, field=name)
    with open_regular_binary(root / name, field=name) as reader:
        raw = reader.read(_MAX_METADATA + 1)
    if not raw or len(raw) > _MAX_METADATA or _sha(raw) != digest:
        raise ValueError(f"{name} external pin mismatch or size limit")
    return raw


def _write(root: Path, name: str, value: object) -> str:
    raw = canonical_json_bytes(value)
    if not raw or len(raw) > _MAX_METADATA:
        raise ValueError(f"{name} exceeds witness metadata bound")
    atomic_write_bytes(root / name, raw)
    return _sha(raw)


def _batch_ids(run: RunReceipt, count: int, workers: int) -> tuple[str, ...]:
    if type(count) is not int or not 1 <= count <= _MAX_BATCH:
        raise ValueError("witness batch size must be 1..12")
    if type(workers) is not int or not 1 <= workers <= _MAX_WORKERS:
        raise ValueError("witness worker count must be 1..4")
    if run.runtime_completed or tuple(a.arm_id for a in run.arms) != _IDS:
        raise ValueError("witness requires the exact incomplete roster")
    if any(a.status not in ("NOT_ATTEMPTED", "COMPLETED") for a in run.arms):
        raise ValueError(
            "stopped or failed witness requires explicit recovery; no implicit retry"
        )
    ids = tuple(a.arm_id for a in run.arms if a.status == "NOT_ATTEMPTED")[:count]
    if not ids:
        raise ValueError("witness has no untouched arms")
    return ids


def _set_arm(run: RunReceipt, arm: ArmReceipt) -> RunReceipt:
    return replace(
        run, arms=tuple(arm if a.arm_id == arm.arm_id else a for a in run.arms)
    )


def _prepare(root: Path) -> dict[str, object]:
    head = _clean_head()
    if ".." in root.parts:
        raise ValueError("witness destination must not traverse parents")
    root = recovery._plain_directory(root.parent) / root.name
    if root.exists() or root.is_symlink() or root.is_junction():
        raise FileExistsError("witness destination already exists")
    d, binding, source, report, source_raw = _fixture()
    root.mkdir()
    bundle = root / "attempt"
    start = time.perf_counter()
    pin = publish_frozen_inputs(bundle, prepare_frozen_inputs(d, binding=binding))
    prepare_seconds = time.perf_counter() - start
    attempt = open_synthetic_native_attempt(
        d, bundle / "frozen-plan.bin", expected_binding=binding, expected_pin=pin
    )
    run = new_synthetic_run_receipt(
        attempt,
        run_id="task18-artificial-matrix:" + uuid.uuid4().hex,
        source_binding=source,
        source_report=report,
    )
    atomic_write_bytes(bundle / "source-verification.json", source_raw)
    context = {
        "schema": "task18_artificial_matrix_witness_v1",
        "scope": _SCOPE,
        "source_head": head,
        "dataset_id": d.dataset_id,
        "plan_pin": asdict(pin),
        "run_id": run.run_id,
        "initial_run_sha256": _sha(canonical_json_bytes(run.to_mapping())),
        "prepare_seconds": prepare_seconds,
        "execution_authorized": False,
    }
    context_sha = _write(root, "matrix-context.json", context)
    run_sha = _write(bundle, "stopped-run.json", run.to_mapping())
    (bundle / "arms").mkdir()
    (root / "batch-metrics").mkdir()
    return {
        "root": str(root),
        "context_sha256": context_sha,
        "run_sha256": run_sha,
        **context,
    }


def _open(root: Path, context_sha: str) -> tuple[SyntheticNativeAttempt, dict]:
    context = recovery._decode_metadata(
        _read_pinned(root, "matrix-context.json", context_sha)
    )
    if set(context) != {
        "schema",
        "scope",
        "source_head",
        "dataset_id",
        "plan_pin",
        "run_id",
        "initial_run_sha256",
        "prepare_seconds",
        "execution_authorized",
    }:
        raise ValueError("witness context field closure differs")
    d, binding, source, report, source_raw = _fixture()
    bundle = recovery._plain_directory(root / "attempt")
    if (
        _read_pinned(
            bundle, "source-verification.json", report.verification_report_sha256
        )
        != source_raw
    ):
        raise ValueError("witness artificial source bytes differ")
    pin = FrozenInputsPin(**context["plan_pin"])
    if (
        context.get("scope") != _SCOPE
        or context.get("schema") != "task18_artificial_matrix_witness_v1"
        or context.get("execution_authorized") is not False
        or context.get("dataset_id") != d.dataset_id
        or context.get("source_head") != _clean_head()
    ):
        raise ValueError("witness context differs from exact artificial source")
    return open_synthetic_native_attempt(
        d,
        bundle / "frozen-plan.bin",
        expected_binding=binding,
        expected_pin=pin,
    ), context


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    artifact: VerifiedNativeArmArtifact
    native_seconds: float
    publish_seconds: float
    reload_seconds: float
    peak_rss_kib: int | None


def _worker(
    root: Path, context_sha: str, run: RunReceipt, ids: tuple[str, ...]
) -> tuple[_WorkerResult, ...]:
    # Workers receive only locally generated, in-memory typed declarations.
    # Each owns a disjoint subset of the already durably enrolled batch.
    attempt, _ = _open(root, context_sha)
    _validate_run(attempt, run)
    result = []
    bundle = recovery._plain_directory(root / "attempt")
    for arm_id in ids:
        if next(a.status for a in run.arms if a.arm_id == arm_id) != "RUNNING":
            raise ValueError("worker arm has no RUNNING enrollment")
        t = time.perf_counter()
        observation = next(attempt.iter_arms((arm_id,)))
        native_seconds = time.perf_counter() - t
        t = time.perf_counter()
        published = publish_completed_native_arm(
            bundle, attempt=attempt, run=run, observation=observation
        )
        publish_seconds = time.perf_counter() - t
        del observation
        t = time.perf_counter()
        verified = load_completed_native_arm(
            bundle,
            attempt=attempt,
            run=run,
            arm_id=arm_id,
            expected_pins=published.pins,
        )
        if verified != published:
            raise ValueError(
                "worker completion readback differs from published receipt"
            )
        rss = None
        if os.name == "posix":
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.uname().sysname == "Darwin":
                rss //= 1024
        result.append(
            _WorkerResult(
                verified, native_seconds, publish_seconds, time.perf_counter() - t, rss
            )
        )
    return tuple(result)


def _batch(
    root: Path, context_sha: str, run_sha: str, count: int, workers: int
) -> dict[str, object]:
    root = recovery._plain_directory(root)
    attempt, context = _open(root, context_sha)
    bundle = recovery._plain_directory(root / "attempt")
    raw = _read_pinned(bundle, "stopped-run.json", run_sha)
    run = recovery._parse_run(recovery._decode_metadata(raw))
    _validate_run(attempt, run)
    if run.run_id != context["run_id"]:
        raise ValueError("witness run identity differs from context")
    ids = _batch_ids(run, count, workers)
    previous = tuple(a.arm_id for a in run.arms if a.status == "COMPLETED")
    if {p.name for p in (bundle / "arms").iterdir()} != set(previous):
        raise ValueError("witness arm directory roster differs from pinned run")
    run = replace(
        run,
        arms=tuple(
            replace(
                a,
                status="RUNNING",
                reason_code="RUN_IN_PROGRESS",
                initial_equity_usdt=_PROTOCOL.economics.initial_capital_per_symbol_usdt,
            )
            if a.arm_id in ids
            else a
            for a in run.arms
        ),
    )
    enrollment_sha = _write(bundle, "stopped-run.json", run.to_mapping())
    start = time.perf_counter()
    n = min(workers, len(ids))
    groups = tuple(ids[len(ids) * i // n : len(ids) * (i + 1) // n] for i in range(n))
    if n == 1:
        outputs = _worker(root, context_sha, run, groups[0])
    else:
        with ProcessPoolExecutor(
            max_workers=n, mp_context=get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(_worker, root, context_sha, run, group)
                for group in groups
            ]
            outputs = tuple(item for f in futures for item in f.result())
    by_id = {v.artifact.receipt.arm_id: v for v in outputs}
    if len(outputs) != len(ids) or set(by_id) != set(ids):
        raise ValueError("worker completion roster differs from enrollment")
    for arm_id in ids:
        value = by_id[arm_id]
        if (
            not value.artifact.verification.passed
            or value.artifact.receipt.status != "COMPLETED"
        ):
            raise ValueError("worker completion was not verified")
        run = _set_arm(run, value.artifact.receipt)
    _validate_run(attempt, run)
    completed = tuple(a.arm_id for a in run.arms if a.status == "COMPLETED")
    if len(completed) == len(_IDS):
        run = replace(run, runtime_completed=True)
        evaluation = evaluate_run_receipt(
            run, synthetic_receipt_protocol(attempt, run.source_binding)
        )
        if not (
            evaluation.declared_arm_statuses_complete
            and evaluation.arm_reports_match
            and evaluation.terminal_claims_pass_diagnostic
            and evaluation.declaration_consistent
        ):
            raise ValueError(
                "matrix completion declaration failed receipt reconciliation"
            )
    new_sha = _write(bundle, "stopped-run.json", run.to_mapping())
    metrics = {
        "scope": _SCOPE,
        "source_head": context["source_head"],
        "plan_pin": context["plan_pin"],
        "previous_run_sha256": run_sha,
        "enrollment_sha256": enrollment_sha,
        "run_sha256": new_sha,
        "batch_ids": ids,
        "completed_arm_ids": completed,
        "pending_arm_ids": tuple(a for a in _IDS if a not in completed),
        "runtime_completed": run.runtime_completed,
        "worker_count": n,
        "batch_seconds": time.perf_counter() - start,
        "verified_native_advances": sum(
            by_id[a].artifact.verification.advances for a in ids
        ),
        "arms": [
            {
                "arm_id": a,
                "native_seconds": by_id[a].native_seconds,
                "publish_seconds": by_id[a].publish_seconds,
                "reload_seconds": by_id[a].reload_seconds,
                "peak_rss_kib": by_id[a].peak_rss_kib,
                "verification": asdict(by_id[a].artifact.verification),
                "pins": [asdict(p) for p in by_id[a].artifact.pins],
            }
            for a in ids
        ],
        "execution_authorized": False,
    }
    metrics_sha = _write(root / "batch-metrics", f"{len(completed):02d}.json", metrics)
    return {**metrics, "metrics_sha256": metrics_sha}


def _finalize(
    root: Path, context_sha: str, run_sha: str, batch_pins: tuple[tuple[str, str], ...]
) -> dict[str, object]:
    """Read back all independently verified arm bytes and the exact batch chain."""
    root = recovery._plain_directory(root)
    attempt, context = _open(root, context_sha)
    bundle = recovery._plain_directory(root / "attempt")
    run = recovery._parse_run(
        recovery._decode_metadata(_read_pinned(bundle, "stopped-run.json", run_sha))
    )
    protocol = synthetic_receipt_protocol(attempt, run.source_binding)
    evaluation = evaluate_run_receipt(run, protocol)
    if (
        run.run_id != context["run_id"]
        or not run.runtime_completed
        or not evaluation.declaration_consistent
        or not evaluation.source_report_matches
        or not evaluation.declared_arm_statuses_complete
        or not evaluation.arm_reports_match
        or not evaluation.terminal_claims_pass_diagnostic
        or tuple(a.arm_id for a in run.arms) != _IDS
    ):
        raise ValueError("matrix witness is not complete")
    metrics_dir = recovery._plain_directory(root / "batch-metrics")
    if (
        not batch_pins
        or len(batch_pins) > 48
        or {p.name for p in metrics_dir.iterdir()} != {name for name, _ in batch_pins}
    ):
        raise ValueError("matrix batch evidence roster differs")
    seen: list[str] = []
    total = 0
    native_seconds = 0.0
    publish_seconds = 0.0
    reload_seconds = 0.0
    batch_seconds = 0.0
    peak = 0
    previous = context["initial_run_sha256"]
    receipts = {a.arm_id: a for a in run.arms}
    all_pins: list[dict[str, object]] = []
    for name, sha in batch_pins:
        if name not in {f"{n:02d}.json" for n in range(1, 49)}:
            raise ValueError("invalid witness batch metric path")
        entry = recovery._decode_metadata(_read_pinned(metrics_dir, name, sha))
        ids = tuple(entry["batch_ids"])
        if (
            entry["previous_run_sha256"] != previous
            or entry["scope"] != _SCOPE
            or entry["execution_authorized"] is not False
            or entry["source_head"] != context["source_head"]
            or entry["plan_pin"] != context["plan_pin"]
            or not ids
            or ids != _IDS[len(seen) : len(seen) + len(ids)]
            or tuple(item["arm_id"] for item in entry["arms"]) != ids
        ):
            raise ValueError("witness batch lineage differs")
        seen.extend(ids)
        if name != f"{len(seen):02d}.json" or tuple(
            entry["completed_arm_ids"]
        ) != tuple(seen):
            raise ValueError("witness completion frontier differs")
        previous = entry["run_sha256"]
        batch_seconds += entry["batch_seconds"]
        subtotal = 0
        for item in entry["arms"]:
            arm_id = item["arm_id"]
            receipt = receipts[arm_id]
            evidence = receipt.external_report
            if evidence is None:
                raise ValueError("completed arm missing verification report")
            verification = item["verification"]
            if (
                verification["passed"] is not True
                or verification["terminal_flat"] is not True
                or verification["cost_oracle_passed"] is not True
                or verification["pending_orders"] != 0
                or verification["advances"]
                != _PROTOCOL.index_plan.native_advances_per_arm
                or verification["actor_decisions"]
                != _PROTOCOL.index_plan.actor_stop_exclusive
                - _PROTOCOL.index_plan.actor_start_inclusive
                or verification["terminal_advances"]
                != _PROTOCOL.index_plan.terminal_advance_count
            ):
                raise ValueError("matrix arm coverage differs")
            pins = tuple(NativeArmFilePin(**p) for p in item["pins"])
            prefix = f"arms/{arm_id}/"
            expected = {
                prefix + "native-ledger.json": evidence.native_ledger_sha256,
                prefix + "result.json": evidence.arm_result_sha256,
                prefix + "verification.json": evidence.verification_report_sha256,
                prefix + "receipt.json": _sha(
                    canonical_json_bytes(receipt.to_mapping())
                ),
            }
            if len(pins) != 4 or {p.path: p.sha256 for p in pins} != expected:
                raise ValueError("matrix arm pin differs from its receipt")
            directory = recovery._plain_directory(bundle / "arms" / arm_id)
            if {p.name for p in directory.iterdir()} != {
                Path(path).name for path in expected
            }:
                raise ValueError("matrix arm file roster differs")
            for pin in pins:
                if (bundle / pin.path).stat().st_size != pin.size_bytes:
                    raise ValueError(
                        "matrix arm size differs from verified publication"
                    )
                if file_digest_and_size(bundle / pin.path) != (
                    pin.sha256,
                    pin.size_bytes,
                ):
                    raise ValueError(
                        "matrix arm bytes differ from verified publication"
                    )
            all_pins.extend(asdict(p) for p in pins)
            subtotal += verification["advances"]
            native_seconds += item["native_seconds"]
            publish_seconds += item["publish_seconds"]
            reload_seconds += item["reload_seconds"]
            peak = max(peak, item["peak_rss_kib"] or 0)
        if subtotal != entry["verified_native_advances"]:
            raise ValueError("matrix batch native count differs")
        total += subtotal
    if (
        tuple(seen) != _IDS
        or previous != run_sha
        or total != _PROTOCOL.index_plan.total_native_advances
    ):
        raise ValueError("matrix total roster or final digest differs")
    if {p.name for p in (bundle / "arms").iterdir()} != set(_IDS):
        raise ValueError("matrix final arm directory roster differs")
    summary = {
        "schema": "task18_artificial_matrix_completed_witness_v1",
        "scope": _SCOPE,
        "context_sha256": context_sha,
        "run_sha256": run_sha,
        "source_head": context["source_head"],
        "dataset_id": run.dataset_id,
        "plan_pin": context["plan_pin"],
        "run_id": run.run_id,
        "completed_arm_ids": _IDS,
        "verified_native_advances": total,
        "all_terminal_flat": True,
        "pending_order_count": 0,
        "arm_file_pins": all_pins,
        "batch_metrics_pins": batch_pins,
        "prepare_seconds": context["prepare_seconds"],
        "batch_wall_seconds": batch_seconds,
        "sum_native_worker_seconds": native_seconds,
        "sum_publication_worker_seconds": publish_seconds,
        "sum_reload_worker_seconds": reload_seconds,
        "peak_single_worker_rss_kib": peak or None,
        "runtime_completed": True,
        "execution_authorized": False,
    }
    if (root / "matrix-witness.json").exists():
        raise FileExistsError("matrix witness summary already exists")
    digest = _write(root, "matrix-witness.json", summary)
    return {k: v for k, v in summary.items() if k != "arm_file_pins"} | {
        "summary_sha256": digest
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("root", type=Path)
    batch = sub.add_parser("batch")
    batch.add_argument("root", type=Path)
    batch.add_argument("--context-sha256", required=True)
    batch.add_argument("--run-sha256", required=True)
    batch.add_argument("--count", type=int, default=4)
    batch.add_argument("--workers", type=int, default=2)
    final = sub.add_parser("finalize")
    final.add_argument("root", type=Path)
    final.add_argument("--context-sha256", required=True)
    final.add_argument("--run-sha256", required=True)
    final.add_argument(
        "--batch-pin",
        action="append",
        required=True,
        help="exact metrics filename=SHA256 in execution order",
    )
    args = parser.parse_args()
    if args.command == "prepare":
        value = _prepare(args.root)
    elif args.command == "batch":
        value = _batch(
            args.root, args.context_sha256, args.run_sha256, args.count, args.workers
        )
    else:
        pairs = tuple(tuple(raw.split("=", 1)) for raw in args.batch_pin)
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("batch pin requires filename=SHA256")
        value = _finalize(
            args.root,
            args.context_sha256,
            args.run_sha256,
            tuple((p[0], p[1]) for p in pairs),
        )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
