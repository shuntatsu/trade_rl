#!/usr/bin/env python3
"""Re-evaluate a completed research generation with fresh confirmation evidence."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any


def _runner_namespace() -> dict[str, Any]:
    runner = Path(__file__).with_name("run_full_research.py")
    return runpy.run_path(str(runner))


def _load_existing_context(
    work_root: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, Any], Path, str]:
    root = work_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"research generation is missing: {root}")
    namespace = _runner_namespace()
    load_json = namespace["_load_json"]
    policy_digest = namespace["_training_policy_digest"]
    summary = load_json(root / "summary.json")
    training = summary.get("training")
    digest = policy_digest(training)
    walk_forward = summary.get("walk_forward")
    if not isinstance(walk_forward, dict):
        raise ValueError("summary walk_forward result is missing")
    raw_path = walk_forward.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("summary walk_forward artifact_path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repository_root / path
    if not path.is_dir():
        raise FileNotFoundError(f"walk-forward artifact is missing: {path}")
    return summary, path, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    summary, walk_forward_path, policy_digest = _load_existing_context(
        args.work_root,
        repository_root=repository_root,
    )
    namespace = _runner_namespace()
    finalize = namespace["_finalize_research_run"]
    exit_code = finalize(
        work_root=args.work_root.resolve(),
        walk_forward_path=walk_forward_path,
        summary=summary,
        strict=True,
        require_confirmation=True,
        expected_policy_digest=policy_digest,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
