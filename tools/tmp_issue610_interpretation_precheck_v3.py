"""Strict pre-interpretation termination validation for Issue #610.

The candidate is never trained here. This layer exists only to apply the already
verified fail-closed termination oracle before the frozen v2 economic decision is
computed from an immutable candidate artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import trade_rl.evaluation.experiments.evidence as evidence_module
from tools.tmp_issue610_exact_once_orchestrator import (
    SEEDS,
    SYMBOLS,
    _strict_source,
    install_artifact_bridge,
)
from tools.tmp_issue610_ppo_global_btc_regime_termination import (
    validate_no_new_ppo_terminations,
)

NEW_TERMINATION_PREFIX = "new PPO termination:"


def partition_termination_violations(
    violations: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split valid new-termination gate failures from invalid evidence failures."""

    new_termination = tuple(
        violation
        for violation in violations
        if violation.startswith(NEW_TERMINATION_PREFIX)
    )
    invalid = tuple(
        violation
        for violation in violations
        if not violation.startswith(NEW_TERMINATION_PREFIX)
    )
    return new_termination, invalid


def classify_strict_termination_evidence(
    baseline: Any,
    candidate: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (new termination gate failures, malformed/roster violations)."""

    violations = validate_no_new_ppo_terminations(
        baseline.runs,
        candidate.runs,
        seeds=SEEDS,
        symbols=SYMBOLS,
    )
    return partition_termination_violations(violations)


def precheck_candidate_artifact(
    *,
    source_root: Path,
    candidate_root: Path,
    artifact_module_path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Load immutable evidence and apply the strict termination oracle only."""

    install_artifact_bridge(artifact_module_path)
    _dataset, _snapshot, baseline = _strict_source(source_root)
    candidate = evidence_module.load_evidence_set(candidate_root / "evidence")
    return classify_strict_termination_evidence(baseline, candidate)


def self_check() -> None:
    if NEW_TERMINATION_PREFIX != "new PPO termination:":
        raise RuntimeError("new-termination prefix drift")
    if SEEDS != (0, 1, 2, 3, 4):
        raise RuntimeError("seed roster drift")
    if SYMBOLS != ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"):
        raise RuntimeError("symbol roster drift")
    print("ISSUE610_INTERPRETATION_PRECHECK_V3_SELF_CHECK=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    precheck = sub.add_parser("precheck")
    precheck.add_argument("--source-root", type=Path, required=True)
    precheck.add_argument("--candidate-root", type=Path, required=True)
    precheck.add_argument("--artifact-module", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "self-check":
        self_check()
        return

    new_termination, invalid = precheck_candidate_artifact(
        source_root=args.source_root,
        candidate_root=args.candidate_root,
        artifact_module_path=args.artifact_module,
    )
    if invalid:
        raise RuntimeError("Issue 610 candidate termination evidence is invalid")
    print("ISSUE610_INTERPRETATION_PRECHECK_V3=PASS")
    print(f"NEW_TERMINATION_PRESENT={'true' if new_termination else 'false'}")
    print("MALFORMED_TERMINATION_EVIDENCE=false")


if __name__ == "__main__":
    main()
