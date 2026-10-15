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

from tier_d_process_tree import ProcessTree, terminate_owned_tree
from tier_d_process_tree import ProcessUsage as ProcessUsage
from tier_d_protocol import (
    IMPLEMENTATION_MODEL,
    REPO,
    ROOT,
    SOURCE_HEAD,
    TASK_ID,
    json_sha256,
    market_protocol,
    source_pins,
)
from tier_d_records import append_log, output_manifest, write_json


def process_rss(process: subprocess.Popen[Any]) -> int:
    fields = (Path("/proc") / str(process.pid) / "status").read_text().splitlines()
    return (
        int(next(row for row in fields if row.startswith("VmRSS:")).split()[1]) * 1024
    )


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
            if slot["state"] == "RUNNING":
                slot["state"] = "INTERRUPTED"
        write_json(state_path, state)


def supervised_run(mode: str, root: Path, review_path: Path | None) -> int:
    from tier_d_runner import validate_external_review

    protocol = market_protocol()
    if mode == "market":
        validate_external_review(review_path, protocol)
    root = root.resolve()
    if not root.is_relative_to(ROOT) or root == ROOT:
        raise ValueError("attempt outputs must stay below the assigned ignored scope")
    root.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-B",
        str(ROOT / "tier_d_runner.py"),
        "--worker",
        "--mode",
        mode,
        "--attempt-root",
        str(root),
    ]
    if review_path is not None:
        command += ["--review-record", str(review_path.resolve())]
    environment = os.environ | {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(REPO))),
    }
    budget = protocol["resource_proposal"]
    started, peak_rss, failure = time.monotonic(), 0, None
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
                        owned_process_count=len(initial_sample.members),
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
                    try:
                        process.wait(timeout=budget["sample_seconds"])
                        break
                    except subprocess.TimeoutExpired:
                        pass
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
                        owned_process_count=len(sample.members)
                        if sample is not None
                        else 1,
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
            except Exception as error:
                failure = (
                    f"resource supervision failed: {type(error).__name__}: {error}"
                )
            except KeyboardInterrupt:
                failure = "operator interruption; retain partial attempt"
            finally:
                try:
                    survivors: tuple[ProcessUsage, ...] = ()
                    if tree is not None:
                        if failure is None and process.returncode in (None, 0):
                            try:
                                survivors = tree.sample().members
                            except Exception as sampling_error:
                                failure = f"final ownership sample failed: {type(sampling_error).__name__}: {sampling_error}"
                                survivors = (
                                    tree.last_sample.members if tree.last_sample else ()
                                )
                        else:
                            survivors = (
                                tree.last_sample.members if tree.last_sample else ()
                            )
                    if survivors and process.poll() is not None and failure is None:
                        failure = "owned descendants remained after worker exit; retain partial attempt"
                    if (
                        failure is not None
                        or process.returncode not in (None, 0)
                        or survivors
                    ):
                        if tree is not None:
                            terminate_owned_tree(process, tree)
                        elif process.poll() is None:
                            process.terminate()
                            process.wait(timeout=30)
                except Exception as error:
                    cleanup_error = f"owned process cleanup or verification failed: {type(error).__name__}: {error}"
                    failure = (
                        f"{failure}; {cleanup_error}" if failure else cleanup_error
                    )
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)
            returncode = process.returncode
    final_disk = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    final_elapsed = time.monotonic() - started
    if final_disk > budget["disk_bytes"]:
        failure = "final output disk budget exceeded; retain full failure artifacts"
    if final_elapsed > budget["wall_seconds"]:
        failure = "final wall-time budget exceeded; retain full attempt artifacts"
    if failure is None and returncode != 0:
        failure = f"worker exited nonzero ({returncode}); retain partial attempt"
    report = {
        "status": "FAILED" if failure or returncode else "COMPLETED",
        "failure": failure,
        "child_returncode": returncode,
        "elapsed_seconds": final_elapsed,
        "sampled_peak_rss_bytes": peak_rss or None,
        "final_disk_bytes": final_disk,
        "limits": budget,
        "measurement": "observed for this attempt only; no market benchmark",
    }
    write_json(root / "supervisor.json", report)
    mark_attempt_failure(root, failure=failure, returncode=returncode)
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
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.mode == "source":
        protocol = market_protocol()
        write_json(ROOT / "protocol.json", protocol)
        write_json(
            ROOT / "implementation-harness-provenance.json",
            {
                "task_id": TASK_ID,
                "source_head": SOURCE_HEAD,
                "configured_model": IMPLEMENTATION_MODEL,
                "configured_reasoning_effort": "max",
                "evidence": "coordinator harness configuration only; provider weights are not observable",
                "protocol_digest": json_sha256(protocol),
                "source_sha256": source_pins(),
                "economic_run": "NEW_CONTROL_SOURCE_ATTEMPT_NOT_STARTED; ROOT different-model review prerequisite",
                "preserved_prior_actual_attempt": {
                    "attempt": "market-20261014T1305Z",
                    "status": "FAILED_RESOURCE_SUPERVISION",
                    "cost_producers_completed": 4,
                    "public_preparations_completed": 4,
                    "ridge_vintages_completed": 12,
                    "native_arms": "48 UNATTEMPTED; no replay/PnL",
                },
            },
        )
        print(f"Result-blind source only: {ROOT / 'protocol.json'}")
        return
    if arguments.worker:
        from tier_d_runner import run_market, run_synthetic

        if arguments.attempt_root is None:
            raise ValueError("worker needs an explicit attempt root")
        if arguments.mode == "market":
            run_market(arguments.attempt_root, review_path=arguments.review_record)
        else:
            run_synthetic(arguments.attempt_root)
        return
    attempt_root = arguments.attempt_root or ROOT / (
        f"{arguments.mode}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    sys.exit(supervised_run(arguments.mode, attempt_root, arguments.review_record))
