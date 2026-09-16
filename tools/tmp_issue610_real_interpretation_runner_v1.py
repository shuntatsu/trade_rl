"""Deterministic real-artifact interpretation wrapper for Issue #610.

This module is fixed before the recovery candidate artifact is published. It never
trains PPO. It applies the verified strict termination precheck first and invokes
the frozen v2 economic interpretation only when termination evidence is structurally
valid. A canonical decision envelope is emitted for independent reconstruction.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from tools.tmp_issue610_interpretation_precheck_v3 import precheck_candidate_artifact
from tools.tmp_issue610_interpretation_v2 import interpret_candidate_v2

ISSUE_NUMBER = 610
STRICT_PRECHECK_SCHEMA = "issue610_strict_termination_precheck_v1"
DECISION_ENVELOPE_SCHEMA = "issue610_interpretation_decision_v3"
VALID_DECISIONS = frozenset({"ACCEPT_CANDIDATE", "KEEP_BASELINE", "INVALID"})


def _with_digest(payload: Mapping[str, object]) -> dict[str, object]:
    resolved = dict(payload)
    resolved["content_digest"] = content_digest(resolved)
    return resolved


def build_strict_precheck_report(
    *,
    interpretation_run_id: int,
    precompute_artifact_id: int,
    precompute_artifact_digest: str,
    candidate_artifact_id: int,
    candidate_artifact_digest: str,
    new_termination: tuple[str, ...],
    invalid_termination_evidence: tuple[str, ...],
) -> dict[str, object]:
    """Build the canonical strict termination precheck report."""

    return _with_digest(
        {
            "schema_version": STRICT_PRECHECK_SCHEMA,
            "issue_number": ISSUE_NUMBER,
            "interpretation_run_id": interpretation_run_id,
            "precompute_artifact_id": precompute_artifact_id,
            "precompute_artifact_api_digest": precompute_artifact_digest,
            "candidate_artifact_id": candidate_artifact_id,
            "candidate_artifact_api_digest": candidate_artifact_digest,
            "new_termination_violations": list(new_termination),
            "invalid_termination_evidence": list(invalid_termination_evidence),
            "termination_evidence_valid": not invalid_termination_evidence,
            "candidate_retrained_during_interpretation": False,
            "economic_values_interpreted": False,
            "final_test_accessed": False,
            "production_eligible": False,
            "live_trading_authorized": False,
            "merge_authorized": False,
        }
    )


def build_decision_envelope(
    *,
    strict_precheck: Mapping[str, object],
    v2_result: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the final v3 envelope while fail-closing classifier disagreement."""

    strict_invalid = strict_precheck.get("invalid_termination_evidence")
    strict_new = strict_precheck.get("new_termination_violations")
    if not isinstance(strict_invalid, list) or not all(
        isinstance(value, str) for value in strict_invalid
    ):
        raise RuntimeError("strict invalid-termination payload malformed")
    if not isinstance(strict_new, list) or not all(
        isinstance(value, str) for value in strict_new
    ):
        raise RuntimeError("strict new-termination payload malformed")

    if strict_invalid:
        if v2_result is not None:
            raise RuntimeError("v2 interpretation forbidden for invalid termination evidence")
        decision = "INVALID"
        v2_digest: object = None
        economic_values_interpreted = False
    else:
        if v2_result is None:
            raise RuntimeError("valid termination evidence requires frozen v2 interpretation")
        v2_new = v2_result.get("new_termination_violations")
        v2_validity = v2_result.get("termination_validity_violations")
        if not isinstance(v2_new, list) or not all(
            isinstance(value, str) for value in v2_new
        ):
            raise RuntimeError("v2 new-termination payload malformed")
        if v2_validity != []:
            raise RuntimeError("strict/v2 termination validity disagreement")
        if bool(strict_new) != bool(v2_new):
            raise RuntimeError("strict/v2 new-termination classification disagreement")
        raw_decision = v2_result.get("decision")
        if not isinstance(raw_decision, str) or raw_decision not in VALID_DECISIONS:
            raise RuntimeError("v2 decision malformed")
        decision = raw_decision
        v2_digest = v2_result.get("content_digest")
        if not isinstance(v2_digest, str):
            raise RuntimeError("v2 result content digest missing")
        unsigned_v2 = dict(v2_result)
        unsigned_v2.pop("content_digest", None)
        if content_digest(unsigned_v2) != v2_digest:
            raise RuntimeError("v2 result content digest mismatch")
        economic_values_interpreted = True

    strict_digest = strict_precheck.get("content_digest")
    if not isinstance(strict_digest, str):
        raise RuntimeError("strict precheck content digest missing")
    unsigned_strict = dict(strict_precheck)
    unsigned_strict.pop("content_digest", None)
    if content_digest(unsigned_strict) != strict_digest:
        raise RuntimeError("strict precheck content digest mismatch")

    return _with_digest(
        {
            "schema_version": DECISION_ENVELOPE_SCHEMA,
            "issue_number": ISSUE_NUMBER,
            "interpretation_run_id": strict_precheck["interpretation_run_id"],
            "precompute_artifact_id": strict_precheck["precompute_artifact_id"],
            "precompute_artifact_api_digest": strict_precheck[
                "precompute_artifact_api_digest"
            ],
            "candidate_artifact_id": strict_precheck["candidate_artifact_id"],
            "candidate_artifact_api_digest": strict_precheck[
                "candidate_artifact_api_digest"
            ],
            "strict_precheck_content_digest": strict_digest,
            "v2_result_content_digest": v2_digest,
            "decision": decision,
            "strict_new_termination_present": bool(strict_new),
            "strict_termination_evidence_valid": not strict_invalid,
            "candidate_retrained_during_interpretation": False,
            "economic_values_interpreted": economic_values_interpreted,
            "final_test_accessed": False,
            "operational_eligibility_established": False,
            "production_eligible": False,
            "live_trading_authorized": False,
            "merge_authorized": False,
        }
    )


def interpret_real_candidate_v1(
    *,
    source_root: Path,
    candidate_root: Path,
    precompute_path: Path,
    output_root: Path,
    artifact_module_path: Path,
    interpretation_run_id: int,
    precompute_artifact_id: int,
    precompute_artifact_digest: str,
    candidate_artifact_id: int,
    candidate_artifact_digest: str,
) -> dict[str, object]:
    """Interpret one immutable candidate without any training or rule adaptation."""

    new_termination, invalid_termination = precheck_candidate_artifact(
        source_root=source_root,
        candidate_root=candidate_root,
        artifact_module_path=artifact_module_path,
    )
    strict_report = build_strict_precheck_report(
        interpretation_run_id=interpretation_run_id,
        precompute_artifact_id=precompute_artifact_id,
        precompute_artifact_digest=precompute_artifact_digest,
        candidate_artifact_id=candidate_artifact_id,
        candidate_artifact_digest=candidate_artifact_digest,
        new_termination=new_termination,
        invalid_termination_evidence=invalid_termination,
    )

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "strict-precheck.json").write_bytes(
        canonical_json_bytes(strict_report)
    )

    v2_result: Mapping[str, object] | None = None
    if not invalid_termination:
        resolved_v2 = interpret_candidate_v2(
            source_root=source_root,
            candidate_root=candidate_root,
            precompute_path=precompute_path,
            output_root=output_root / "v2",
            artifact_module_path=artifact_module_path,
            interpretation_run_id=interpretation_run_id,
            precompute_artifact_id=precompute_artifact_id,
            precompute_artifact_digest=precompute_artifact_digest,
            candidate_artifact_id=candidate_artifact_id,
            candidate_artifact_digest=candidate_artifact_digest,
        )
        v2_result = cast(Mapping[str, object], resolved_v2)

    envelope = build_decision_envelope(
        strict_precheck=strict_report,
        v2_result=v2_result,
    )
    (output_root / "decision.json").write_bytes(canonical_json_bytes(envelope))
    print("ISSUE610_REAL_INTERPRETATION_RUNNER_V1=PASS")
    print("FINAL_TEST_ACCESSED=false")
    print("PRODUCTION_ELIGIBLE=false")
    print("LIVE_TRADING_AUTHORIZED=false")
    print("MERGE_AUTHORIZED=false")
    return envelope


def self_check() -> None:
    if STRICT_PRECHECK_SCHEMA != "issue610_strict_termination_precheck_v1":
        raise RuntimeError("strict precheck schema drift")
    if DECISION_ENVELOPE_SCHEMA != "issue610_interpretation_decision_v3":
        raise RuntimeError("decision envelope schema drift")
    if VALID_DECISIONS != frozenset({"ACCEPT_CANDIDATE", "KEEP_BASELINE", "INVALID"}):
        raise RuntimeError("decision roster drift")
    print("ISSUE610_REAL_INTERPRETATION_RUNNER_V1_SELF_CHECK=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    interpret = sub.add_parser("interpret")
    interpret.add_argument("--source-root", type=Path, required=True)
    interpret.add_argument("--candidate-root", type=Path, required=True)
    interpret.add_argument("--precompute", type=Path, required=True)
    interpret.add_argument("--output-root", type=Path, required=True)
    interpret.add_argument("--artifact-module", type=Path, required=True)
    interpret.add_argument("--interpretation-run-id", type=int, required=True)
    interpret.add_argument("--precompute-artifact-id", type=int, required=True)
    interpret.add_argument("--precompute-artifact-digest", required=True)
    interpret.add_argument("--candidate-artifact-id", type=int, required=True)
    interpret.add_argument("--candidate-artifact-digest", required=True)
    args = parser.parse_args()

    if args.command == "self-check":
        self_check()
        return
    if args.command == "interpret":
        interpret_real_candidate_v1(
            source_root=args.source_root,
            candidate_root=args.candidate_root,
            precompute_path=args.precompute,
            output_root=args.output_root,
            artifact_module_path=args.artifact_module,
            interpretation_run_id=args.interpretation_run_id,
            precompute_artifact_id=args.precompute_artifact_id,
            precompute_artifact_digest=args.precompute_artifact_digest,
            candidate_artifact_id=args.candidate_artifact_id,
            candidate_artifact_digest=args.candidate_artifact_digest,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
