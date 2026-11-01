"""Command line entry: ``python -m trade_rl.evaluation.residual_rl``.

Every invocation that executes code appends one entry to the provenance JSON,
including calibration and smoke runs. The ``full`` mode refuses a dirty working
tree (outside the report files it writes) so the recorded commit is the exact code.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from trade_rl.evaluation.residual_rl.protocol import (
    FROZEN_PROTOCOL_SHA256,
    PROTOCOL_V1,
    protocol_digest,
    protocol_payload,
)
from trade_rl.evaluation.residual_rl.study import (
    run_calibration,
    run_full_study,
    run_smoke,
)

REPORT_PREFIX = "issue810-residual-rl-synthetic-20261030"
DEFAULT_REPORT_DIR = Path("report")
_REPORT_EXCLUDE = f":(exclude)report/{REPORT_PREFIX}*"


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--", ".", _REPORT_EXCLUDE)
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_outside_report_files": bool(status),
    }


def write_json(path: Path, value: object) -> str:
    """Write LF-only pretty JSON and return its SHA-256."""

    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    with path.open("wb") as handle:
        handle.write(data)
    return sha256(data).hexdigest()


def append_provenance(path: Path, entry: dict[str, Any]) -> None:
    runs: list[dict[str, Any]] = []
    if path.exists():
        runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    runs.append(entry)
    write_json(
        path,
        {
            "schema": "issue810_residual_ppo_synthetic_provenance_v1",
            "protocol_id": PROTOCOL_V1.protocol_id,
            "note": (
                "Every execution of the study code is listed, including "
                "calibration and smoke runs. SYNTHETIC data only."
            ),
            "runs": runs,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m trade_rl.evaluation.residual_rl")
    parser.add_argument(
        "mode", choices=("write-protocol", "calibration", "smoke", "full")
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke-timesteps", type=int, default=1_008)
    parser.add_argument("--smoke-seeds", type=int, default=2)
    parser.add_argument("--note", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report_dir: Path = args.report_dir
    protocol_path = report_dir / f"{REPORT_PREFIX}-protocol.json"
    provenance_path = report_dir / f"{REPORT_PREFIX}-provenance.json"
    results_path = report_dir / f"{REPORT_PREFIX}-results.json"
    digest = protocol_digest()

    if args.mode == "write-protocol":
        sha = write_json(
            protocol_path,
            {
                "schema": "issue810_residual_ppo_synthetic_protocol_v1",
                "protocol_sha256": digest,
                "frozen_protocol_sha256_in_source": FROZEN_PROTOCOL_SHA256,
                "protocol": protocol_payload(),
            },
        )
        print(f"protocol digest {digest}; file sha256 {sha}")
        return 0

    state = git_state()
    if args.mode == "full" and state["dirty_outside_report_files"]:
        print("refusing a full run on a dirty working tree", file=sys.stderr)
        return 2

    started = datetime.now(UTC)
    clock = time.monotonic()
    entry: dict[str, Any] = {
        "mode": args.mode,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "note": args.note,
        "started_utc": started.isoformat(),
        "git": state,
        "protocol_sha256_computed": digest,
        "protocol_sha256_frozen_in_source": FROZEN_PROTOCOL_SHA256,
        "synthetic_only": True,
    }
    if args.mode == "calibration":
        entry["evidence"] = False
        entry["result"] = run_calibration(PROTOCOL_V1)
    elif args.mode == "smoke":
        entry["evidence"] = False
        entry["result"] = run_smoke(
            PROTOCOL_V1,
            total_timesteps=args.smoke_timesteps,
            n_seeds=args.smoke_seeds,
        )
    else:
        result = run_full_study(PROTOCOL_V1, workers=args.workers)
        entry["evidence"] = True
        entry["label"] = result["label"]
        entry["label_reasons"] = result["label_reasons"]
        entry["results_file"] = results_path.as_posix()
        entry["results_file_sha256"] = write_json(
            results_path, {**result, "git": state}
        )
    entry["finished_utc"] = datetime.now(UTC).isoformat()
    entry["elapsed_seconds"] = round(time.monotonic() - clock, 1)
    append_provenance(provenance_path, entry)
    print(json.dumps({k: v for k, v in entry.items() if k != "result"}, indent=2))
    return 0


__all__ = ["REPORT_PREFIX", "append_provenance", "git_state", "main", "write_json"]
