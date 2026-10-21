"""A sampled resource supervisor; ordinary retries retain independent attempts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .progress import validate_progress_policy
from .progress_io import ProgressReader
from .tier_d_process_tree import ProcessTree, ProcessTreeSample, terminate_owned_tree
from .tier_d_process_tree import ProcessUsage as ProcessUsage
from .tier_d_protocol import (
    IMPLEMENTATION_MODEL,
    REPO,
    ROOT,
    TASK_ID,
    json_sha256,
    market_protocol,
    source_head,
    source_pins,
    validate_committed_source,
    worker_count,
)
from .tier_d_records import (
    append_log,
    arm_state,
    output_manifest,
    read_json,
    write_json,
)


def process_rss(process: subprocess.Popen[Any]) -> int:
    try:
        fields = (Path("/proc") / str(process.pid) / "status").read_text().splitlines()
        return (
            int(next(row for row in fields if row.startswith("VmRSS:")).split()[1])
            * 1024
        )
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        if process.poll() is not None:
            return 0
        raise


def mark_attempt_failure(
    root: Path, *, failure: str | None, returncode: int | None
) -> None:
    if failure is None and returncode == 0:
        return
    reason = failure or f"worker exited nonzero ({returncode}); retain partial attempt"
    state_path = root / "attempt.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state.update(status="FAILED", supervisor_failure=reason)
        for slot in state["arms"]:
            if slot["state"] in {"RUNNING", "SUBMITTED"}:
                slot["state"] = "INTERRUPTED"
        write_json(state_path, state, replace=True)


def _publish_stopped_receipts(
    root: Path,
    protocol: dict[str, Any],
    provenance: dict[str, Any],
    *,
    owned_tree_stopped: bool,
) -> dict[str, Any]:
    """Finalize only this supervisor's stopped attempt; never repair an old source."""
    if not owned_tree_stopped:
        return {"state": "SKIPPED_NO_VERIFIED_OWNED_TREE_STOP"}
    from .resume import reconcile_stopped_receipts

    try:
        state_path = root / "attempt.json"
        if root.is_symlink() or state_path.is_symlink() or not state_path.is_file():
            raise ValueError("stopped attempt state must be a regular file")
        original_state = state_path.read_bytes()
        updates = reconcile_stopped_receipts(root, protocol, provenance)
        if state_path.read_bytes() != original_state:
            raise ValueError("stopped attempt state changed before reconciliation")
        state = read_json(state_path)
        for update in updates:
            arm_state(
                state, update["account"], update["scenario"], update["role"]
            ).update(update)
        if updates:
            write_json(state_path, state, replace=True)
        return {"state": "RECONCILED", "completed_arms": len(updates)}
    except Exception as error:
        return {
            "state": "REFUSED",
            "failure": {"type": type(error).__name__, "message": str(error)},
        }


def worker_command(
    mode: str,
    root: Path,
    review_path: Path | None,
    *,
    protocol_path: Path | None = None,
    resume_from: Path | None = None,
    max_workers: int = 1,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.tier_d_nonrl",
        "--worker",
        "--mode",
        mode,
        "--attempt-root",
        str(root),
        "--max-workers",
        str(worker_count(max_workers)),
    ]
    for flag, path in (
        ("--review-record", review_path),
        ("--protocol", protocol_path),
        ("--resume-from", resume_from),
    ):
        if path is not None:
            command += [flag, str(path.resolve())]
    return command


def supervised_run(
    mode: str,
    root: Path,
    review_path: Path | None,
    *,
    protocol_path: Path | None = None,
    resume_from: Path | None = None,
    max_workers: int | None = None,
) -> int:
    if mode == "market" and os.name != "nt":
        raise ValueError(
            "market supervision requires Windows aggregate process-tree monitoring"
        )
    from .tier_d_runner import (
        build_driver_run_provenance,
        load_market_protocol,
        validate_external_review,
    )

    protocol = (
        load_market_protocol(protocol_path, max_workers=max_workers)
        if mode == "market"
        else market_protocol(max_workers=max_workers or 1)
    )
    if mode == "market":
        validate_external_review(review_path, protocol)
        validate_committed_source(protocol)
    launch_head, launch_driver = protocol["source_head"], source_pins()
    root = root.resolve()
    if not root.is_relative_to(ROOT) or root == ROOT:
        raise ValueError("attempt outputs must stay below the assigned ignored scope")
    root.mkdir(parents=True, exist_ok=False)
    artifact_root = root / "attempt"
    command = worker_command(
        mode,
        artifact_root,
        review_path,
        protocol_path=protocol_path,
        resume_from=resume_from,
        max_workers=protocol["concurrency"]["max_workers"],
    )
    environment = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPO),
    }
    budget = protocol["resource_proposal"]
    progress_policy = validate_progress_policy(protocol.get("progress_policy"))
    reader = ProgressReader(
        artifact_root,
        expected_protocol=protocol if mode == "market" else None,
        expected_policy=progress_policy,
        clock=time.monotonic,
    )
    stalled_arms: list[dict[str, Any]] = []
    owned_tree_stopped = False
    started, peak_rss, failure = time.monotonic(), 0, None
    next_resource_sample = started + budget["sample_seconds"]
    with (root / "worker.log").open("w", encoding="utf-8") as output:
        with subprocess.Popen(
            command, cwd=REPO, env=environment, stdout=output, stderr=subprocess.STDOUT
        ) as process:
            tree = ProcessTree(process.pid) if os.name == "nt" else None
            try:
                if tree is not None:
                    initial_sample = tree.sample()
                    peak_rss = initial_sample.rss_bytes
                    append_log(
                        root,
                        "resource_sample",
                        rss_bytes=initial_sample.rss_bytes,
                        root_rss_bytes=initial_sample.root_rss_bytes,
                        rss_attributed_process_count=len(initial_sample.members),
                        owned_process_count=len(initial_sample.owned_members),
                        ambiguous_rss_process_count=len(
                            initial_sample.ambiguous_members
                        ),
                        disk_bytes=sum(
                            path.stat().st_size
                            for path in root.rglob("*")
                            if path.is_file()
                        ),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    if initial_sample.rss_bytes > budget["rss_bytes"]:
                        failure = "sampled process-tree RSS budget exceeded; retain partial attempt"
                while failure is None and process.poll() is None:
                    for _ in reader.poll():
                        pass  # Child parent owns canonical progress publication.
                    stalled_arms = reader.stalled()
                    if stalled_arms:
                        failure = "native progress stalled; retain partial attempt"
                        break
                    try:
                        process.wait(
                            timeout=min(
                                progress_policy["sample_seconds"],
                                max(0, next_resource_sample - time.monotonic()),
                            )
                        )
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if time.monotonic() < next_resource_sample:
                        continue
                    next_resource_sample = time.monotonic() + budget["sample_seconds"]
                    sample = tree.sample() if tree is not None else None
                    rss = (
                        sample.rss_bytes if sample is not None else process_rss(process)
                    )
                    peak_rss = max(peak_rss, rss)
                    disk = sum(
                        path.stat().st_size
                        for path in root.rglob("*")
                        if path.is_file()
                    )
                    elapsed = time.monotonic() - started
                    append_log(
                        root,
                        "resource_sample",
                        rss_bytes=rss,
                        root_rss_bytes=sample.root_rss_bytes
                        if sample is not None
                        else rss,
                        owned_process_count=len(sample.owned_members)
                        if sample is not None
                        else 1,
                        rss_attributed_process_count=len(sample.members)
                        if sample is not None
                        else 1,
                        ambiguous_rss_process_count=len(sample.ambiguous_members)
                        if sample is not None
                        else 0,
                        disk_bytes=disk,
                        elapsed_seconds=elapsed,
                    )
                    if (
                        rss > budget["rss_bytes"]
                        or disk > budget["disk_bytes"]
                        or elapsed > budget["wall_seconds"]
                    ):
                        failure = (
                            "sampled resource budget exceeded; retain partial attempt"
                        )
                        break
                if failure is None:
                    for _ in reader.poll():
                        pass  # Validate the final committed journal prefix too.
            except Exception as error:
                failure = (
                    f"execution supervision failed: {type(error).__name__}: {error}"
                )
            except KeyboardInterrupt:
                failure = "operator interruption; retain partial attempt"
            finally:
                try:
                    owned_survivors: tuple[ProcessUsage, ...] = ()
                    ambiguous_survivors: tuple[ProcessUsage, ...] = ()
                    final_sample: ProcessTreeSample | None = None
                    final_sample_is_fresh = False
                    if tree is not None:
                        if failure is None and process.returncode in (None, 0):
                            try:
                                final_sample = tree.sample()
                                final_sample_is_fresh = True
                            except Exception as sampling_error:
                                failure = f"final ownership sample failed: {type(sampling_error).__name__}: {sampling_error}"
                                final_sample = tree.last_sample
                        else:
                            final_sample = tree.last_sample
                        if final_sample is not None:
                            owned_survivors = final_sample.owned_members
                            ambiguous_survivors = final_sample.ambiguous_members
                            if final_sample_is_fresh:
                                peak_rss = max(peak_rss, final_sample.rss_bytes)
                                append_log(
                                    root,
                                    "resource_sample",
                                    rss_bytes=final_sample.rss_bytes,
                                    root_rss_bytes=final_sample.root_rss_bytes,
                                    rss_attributed_process_count=len(
                                        final_sample.members
                                    ),
                                    owned_process_count=len(final_sample.owned_members),
                                    ambiguous_rss_process_count=len(
                                        final_sample.ambiguous_members
                                    ),
                                    disk_bytes=sum(
                                        path.stat().st_size
                                        for path in root.rglob("*")
                                        if path.is_file()
                                    ),
                                    elapsed_seconds=time.monotonic() - started,
                                )
                    if ambiguous_survivors and failure is None:
                        failure = (
                            "ambiguous RSS-attributed processes remained after worker exit; "
                            "retain partial attempt"
                        )
                    elif (
                        owned_survivors
                        and process.poll() is not None
                        and failure is None
                    ):
                        failure = "verified owned descendants remained after worker exit; retain partial attempt"
                    if (
                        failure is not None
                        or process.returncode not in (None, 0)
                        or owned_survivors
                        or ambiguous_survivors
                    ):
                        if tree is not None:
                            terminate_owned_tree(process, tree)
                            owned_tree_stopped = (
                                getattr(tree, "root_identity", None) is not None
                                and bool(getattr(tree, "owned", {}))
                                and process.poll() is not None
                            )
                            owned_survivors = ()
                            ambiguous_survivors = ()
                        elif process.poll() is None:
                            process.terminate()
                            process.wait(timeout=30)
                except Exception as error:
                    cleanup_error = f"owned process cleanup or verification failed: {type(error).__name__}: {error}"
                    failure = (
                        f"{failure}; {cleanup_error}" if failure else cleanup_error
                    )
                    if tree is not None and tree.last_sample is not None:
                        owned_survivors = tree.last_sample.owned_members
                        ambiguous_survivors = tree.last_sample.ambiguous_members
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)
            returncode = process.returncode
    final_disk = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    final_elapsed = time.monotonic() - started
    if final_disk > budget["disk_bytes"]:
        failure = (
            failure
            or "final output disk budget exceeded; retain full failure artifacts"
        )
    if final_elapsed > budget["wall_seconds"]:
        failure = (
            failure or "final wall-time budget exceeded; retain full attempt artifacts"
        )
    if failure is None and returncode != 0:
        failure = f"worker exited nonzero ({returncode}); retain partial attempt"
    report = {
        "status": "FAILED" if failure or returncode else "COMPLETED",
        "failure": failure,
        "child_returncode": returncode,
        "elapsed_seconds": final_elapsed,
        "sampled_peak_rss_bytes": peak_rss or None,
        "rss_attributed_processes_remaining": len(
            owned_survivors + ambiguous_survivors
        ),
        "ambiguous_rss_processes_remaining": len(ambiguous_survivors),
        "verified_owned_processes_remaining": len(owned_survivors),
        "final_disk_bytes": final_disk,
        "limits": budget,
        "progress_policy": progress_policy,
        "native_progress_states": reader.states,
        "stalled_arms": stalled_arms,
        "measurement": "observed for this attempt only; no market benchmark",
        "attempt_root": str(artifact_root),
    }
    write_json(root / "supervisor.json", report)
    mark_attempt_failure(artifact_root, failure=failure, returncode=returncode)
    if failure is not None or returncode != 0:
        report["receipt_reconciliation"] = {
            "state": "SKIPPED_NO_VERIFIED_OWNED_TREE_STOP"
        }
        if owned_tree_stopped:
            try:
                retained_protocol = read_json(artifact_root / "protocol.json")
                if (
                    source_head() != launch_head
                    or source_pins() != launch_driver
                    or retained_protocol["source_head"] != launch_head
                ):
                    raise ValueError("supervised source changed before finalization")
                if mode == "market":
                    if json_sha256(retained_protocol) != json_sha256(protocol):
                        raise ValueError("stopped market protocol differs")
                    final_review = validate_external_review(
                        review_path.resolve() if review_path is not None else None,
                        protocol,
                    )
                else:
                    if (
                        retained_protocol["scope"]
                        != "ARTIFICIAL_SOFTWARE_INTEGRATION_ONLY"
                        or retained_protocol["concurrency"] != protocol["concurrency"]
                        or retained_protocol["progress_policy"] != progress_policy
                    ):
                        raise ValueError("stopped artificial protocol differs")
                    final_review = None
                report["receipt_reconciliation"] = _publish_stopped_receipts(
                    artifact_root,
                    retained_protocol,
                    build_driver_run_provenance(retained_protocol, final_review),
                    owned_tree_stopped=True,
                )
            except Exception as error:
                report["receipt_reconciliation"] = {
                    "state": "REFUSED",
                    "failure": {"type": type(error).__name__, "message": str(error)},
                }
        write_json(root / "supervisor.json", report, replace=True)
    if artifact_root.exists():
        write_json(
            artifact_root / "output-manifest.json",
            output_manifest(artifact_root),
            replace=True,
        )
    write_json(root / "output-manifest.json", output_manifest(root))
    print(f"{report['status']}: {root}", flush=True)
    return 1 if report["status"] == "FAILED" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("source", "synthetic", "market"), default="source"
    )
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--review-record", type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        help="exact frozen source declaration; required for market",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="read-only prior worker attempt, not its supervisor container",
    )
    parser.add_argument("--max-workers", type=int, choices=range(1, 7), default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.mode == "source":
        root = arguments.attempt_root or ROOT / f"source-{uuid4().hex[:12]}"
        root.mkdir(parents=True, exist_ok=False)
        protocol = market_protocol(max_workers=arguments.max_workers or 1)
        write_json(root / "protocol.json", protocol)
        write_json(
            root / "implementation-harness-provenance.json",
            {
                "task_id": TASK_ID,
                "source_head": source_head(),
                "configured_model": IMPLEMENTATION_MODEL,
                "configured_reasoning_effort": "max",
                "evidence": "coordinator harness configuration only; provider weights are not observable",
                "protocol_digest": json_sha256(protocol),
                "source_sha256": source_pins(),
                "candidate_execution": "NOT_STARTED; fresh exact-source review required",
                "history": "Prior failed R1/R2 attempts remain preserved separately; this source declaration does not reclassify or authorize them",
            },
        )
        print(f"Result-blind source only: {root / 'protocol.json'}")
        return
    if arguments.worker:
        from .tier_d_runner import run_market, run_synthetic

        if arguments.attempt_root is None:
            raise ValueError("worker needs an explicit attempt root")
        if arguments.mode == "market":
            run_market(
                arguments.attempt_root,
                review_path=arguments.review_record,
                protocol_path=arguments.protocol,
                resume_from=arguments.resume_from,
                max_workers=arguments.max_workers,
            )
        else:
            run_synthetic(
                arguments.attempt_root,
                resume_from=arguments.resume_from,
                max_workers=arguments.max_workers or 1,
            )
        return
    attempt_root = arguments.attempt_root or ROOT / (
        f"{arguments.mode}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    sys.exit(
        supervised_run(
            arguments.mode,
            attempt_root,
            arguments.review_record,
            protocol_path=arguments.protocol,
            resume_from=arguments.resume_from,
            max_workers=arguments.max_workers,
        )
    )
