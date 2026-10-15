"""Windows-only sampled resource control for one artificial scaling worker."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from time import monotonic
from typing import Any

from tools.tier_d_nonrl.tier_d_process_tree import ProcessTree, terminate_owned_tree
from tools.tier_d_nonrl.tier_d_records import append_log, write_json

REPO = Path(__file__).resolve().parents[2]
LIMITS = {
    "wall_seconds": 1200,
    "rss_bytes": 2 * 1024**3,
    "disk_bytes": 4 * 1024**3,
    "sample_seconds": 10,
}


def _platform_name() -> str:
    return os.name


def _disk_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _violations(usage: dict[str, Any], limits: dict[str, int]) -> list[str]:
    return [
        limit
        for limit, field in (
            ("rss_bytes", "rss_bytes"),
            ("disk_bytes", "disk_bytes"),
            ("wall_seconds", "elapsed_seconds"),
        )
        if usage[field] > limits[limit]
    ]


def supervised(command: list[str], root: Path) -> int:
    """Run one owned worker; retain child bytes and return control status 0/1.

    The caller builds the fixed internal command and gives the child
    ``root/attempt``. Only this parent creates the fresh supervisor root.
    Sampled caps permit overshoot; unavailable ownership verification is failure.
    """
    if _platform_name() != "nt":
        raise ValueError("artificial scaling supervision requires Windows")
    if (
        type(command) is not list
        or not command
        or any(type(item) is not str or not item or "\0" in item for item in command)
    ):
        raise ValueError("supervision requires a nonempty command list")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    limits, started = dict(LIMITS), monotonic()
    process: subprocess.Popen[Any] | None = None
    tree: ProcessTree | None = None
    peak_rss = peak_disk = 0
    failure = final_sample_error = None
    cleanup_errors: list[str] = []
    final_violation: list[str] = []
    owned_remaining: int | None = None
    reaped = False

    def sample(stage: str) -> dict[str, Any]:
        nonlocal peak_rss, peak_disk
        if tree is None:
            raise RuntimeError("owned process tree is unavailable")
        observed = tree.sample()
        usage: dict[str, Any] = {
            "rss_bytes": observed.rss_bytes,
            "root_rss_bytes": observed.root_rss_bytes,
            "owned_process_count": len(observed.members),
            "disk_bytes": _disk_bytes(root),
            "elapsed_seconds": monotonic() - started,
        }
        peak_rss = max(peak_rss, observed.rss_bytes)
        peak_disk = max(peak_disk, usage["disk_bytes"])
        append_log(root, "resource_sample", stage=stage, **usage)
        return usage

    with (root / "worker.log").open("x", encoding="utf-8") as output:
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=os.environ
                | {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=output,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            tree = ProcessTree(process.pid)
            usage = sample("initial")
            while True:
                if exceeded := _violations(usage, limits):
                    raise RuntimeError(f"sampled resource budget exceeded: {exceeded}")
                if process.poll() is not None:
                    break
                remaining = limits["wall_seconds"] - (monotonic() - started)
                if remaining <= 0:
                    raise RuntimeError("sampled resource budget exceeded: wall_seconds")
                try:
                    process.wait(timeout=min(limits["sample_seconds"], remaining))
                    break
                except subprocess.TimeoutExpired:
                    usage = sample("periodic")
        except (Exception, KeyboardInterrupt) as error:
            failure = f"{type(error).__name__}: {error}"
        finally:
            if process is not None:
                returncode = process.poll()
                if returncode not in (None, 0):
                    failure = failure or f"worker exited nonzero ({returncode})"
                try:
                    usage = sample("final")
                    final_violation = _violations(usage, limits)
                    owned_remaining = usage["owned_process_count"]
                except Exception as error:
                    final_sample_error = f"{type(error).__name__}: {error}"
                if owned_remaining and returncode is not None:
                    failure = failure or "owned descendants remained after worker exit"
                if failure or final_violation or final_sample_error or owned_remaining:
                    try:
                        if tree is None:
                            raise RuntimeError("owned process tree is unavailable")
                        # This helper uses retained identities if fresh CIM fails.
                        terminate_owned_tree(process, tree)
                    except Exception as error:
                        cleanup_errors.append(f"{type(error).__name__}: {error}")
                try:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=30)
                    reaped = True
                    if tree is None:
                        raise RuntimeError("owned process verification is unavailable")
                    owned_remaining = None
                    owned_remaining = len(tree.sample().members)
                    if owned_remaining:
                        failure = (
                            failure or "owned descendants remained after worker exit"
                        )
                        owned_remaining = None
                        terminate_owned_tree(process, tree)
                        owned_remaining = len(tree.sample().members)
                        if owned_remaining:
                            raise RuntimeError(
                                "owned process descendants survived cleanup"
                            )
                except Exception as error:
                    cleanup_errors.append(f"{type(error).__name__}: {error}")
    elapsed = monotonic() - started
    final_disk = None
    try:
        final_disk = _disk_bytes(root)
        peak_disk = max(peak_disk, final_disk)
        if final_disk > limits["disk_bytes"] and "disk_bytes" not in final_violation:
            final_violation.append("disk_bytes")
    except Exception as error:
        final_sample_error = final_sample_error or f"{type(error).__name__}: {error}"
    if elapsed > limits["wall_seconds"] and "wall_seconds" not in final_violation:
        final_violation.append("wall_seconds")
    returncode = None if process is None else process.returncode
    failed = bool(
        failure
        or cleanup_errors
        or final_violation
        or final_sample_error
        or returncode != 0
        or owned_remaining != 0
    )
    summary: dict[str, Any] = {
        "scope": "ARTIFICIAL_SCALING_SUPERVISION_ONLY",
        "status": "FAILED" if failed else "COMPLETED",
        "command": command,
        "returncode": returncode,
        "failure": failure,
        "cleanup_error": "; ".join(cleanup_errors) or None,
        "final_violation": final_violation,
        "final_sample_error": final_sample_error,
        "owned_members_remaining": owned_remaining,
        "worker_reaped": reaped,
        "peak_rss_bytes": peak_rss,
        "peak_disk_bytes": peak_disk,
        "disk_bytes_before_summary": final_disk,
        "elapsed_seconds": elapsed,
        "limits": limits,
    }
    path = root / "supervision.json"
    write_json(path, summary)
    changed = False
    try:
        if _disk_bytes(root) > limits["disk_bytes"]:
            if "disk_bytes" not in final_violation:
                final_violation.append("disk_bytes")
                changed = True
            failed = True
    except Exception as error:
        if summary["final_sample_error"] is None:
            summary["final_sample_error"] = f"{type(error).__name__}: {error}"
            changed = True
        failed = True
    if changed:
        summary["status"] = "FAILED"
        write_json(path, summary, replace=True)
    return int(failed)
