"""Independent Artifact-only audit for Issue 663 normalization comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.directional_selection import passes_screen, passes_stress

ISSUE = 663
PROTOCOL_SHA = "de6179570dae339c4a3fe3d62d076edc811c332372185efc17d1566e7602fd55"
SEEDS = (0, 1, 2, 3, 4)


def _canonical_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"canonical JSON object required: {path}")
    return raw, value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_package(root: Path, seed: int) -> Path:
    return root / f"issue663-normalization-seed{seed}-v1"


def _fresh_package(root: Path, seed: int) -> Path:
    return root / f"issue663-normalization-seed{seed}-fresh-v1"


def _control_package(root: Path, seed: int) -> Path:
    return root / f"issue645-corrected-directional-arm-ppo{seed}-v1"


def _verify_bundle(package: Path, expected_digest: str) -> None:
    bundle = package / "bundle"
    manifest_raw, manifest = _canonical_object(bundle / "manifest.json")
    if manifest.get("schema") != "ppo_normalized_model_v1":
        raise ValueError("candidate normalized model schema drifted")
    if manifest.get("normalizer", {}).get("schema") != "ppo_feature_standardization_v1":
        raise ValueError("candidate normalizer schema drifted")
    policy = bundle / "policy.zip"
    if _sha(policy) != manifest.get("policy_sha256"):
        raise ValueError("candidate policy bytes differ from bundle manifest")
    from trade_rl.artifacts import content_digest

    if content_digest(manifest) != expected_digest:
        raise ValueError("candidate bundle digest drifted")
    if canonical_json_bytes(manifest) != manifest_raw:
        raise ValueError("candidate bundle manifest is noncanonical")


def audit(
    *,
    protocol_path: Path,
    candidates: Path,
    fresh_root: Path,
    controls: Path,
    comparison_path: Path,
    output: Path,
) -> dict[str, Any]:
    protocol_raw, protocol = _canonical_object(protocol_path)
    if hashlib.sha256(protocol_raw).hexdigest() != PROTOCOL_SHA:
        raise ValueError("Issue 663 protocol SHA differs from sealed v2")
    if (
        protocol.get("issue") != ISSUE
        or protocol["relative_gate"]["required_seed_wins"] != 4
        or protocol["absolute_gate"]["ppo_required_qualifying_seeds"] != 4
        or protocol["absolute_gate"]["terminal_flatness_semantics"]
        != "issue645_reporting_quantity_abs_le_1e-10"
    ):
        raise ValueError("Issue 663 audit protocol semantics drifted")

    comparison_raw, comparison = _canonical_object(comparison_path)
    if comparison.get("schema") != "issue663_normalization_comparison_v1":
        raise ValueError("Issue 663 comparison schema drifted")
    if comparison.get("protocol_sha256") != PROTOCOL_SHA:
        raise ValueError("Issue 663 comparison protocol drifted")

    failed_seeds: list[int] = []
    rows: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    control_results: list[dict[str, Any]] = []
    fresh_records: list[dict[str, Any]] = []

    for seed in SEEDS:
        candidate_package = _candidate_package(candidates, seed)
        _, attempt = _canonical_object(candidate_package / "attempt.json")
        if (
            attempt.get("issue") != ISSUE
            or attempt.get("seed") != seed
            or attempt.get("unused_data_accessed") is not False
            or attempt.get("production_eligible") is not False
            or attempt.get("live_trading_authorized") is not False
        ):
            raise ValueError(f"candidate seed {seed} attempt identity drifted")

        fresh_package = _fresh_package(fresh_root, seed)
        _, fresh = _canonical_object(fresh_package / "fresh.json")
        if (
            fresh.get("seed") != seed
            or fresh.get("policy_model_refit_performed") is not False
        ):
            raise ValueError(f"candidate seed {seed} fresh record drifted")
        fresh_records.append(fresh)

        control_package = _control_package(controls, seed)
        control_raw, control = _canonical_object(control_package / "result.json")
        _, control_sha = _canonical_object(control_package / "result.sha256.json")
        if control_sha != {"sha256": hashlib.sha256(control_raw).hexdigest()}:
            raise ValueError(f"control seed {seed} result digest drifted")
        if control.get("arm") != f"ppo{seed}":
            raise ValueError(f"control seed {seed} identity drifted")
        control_results.append(control)

        exit_code = attempt.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError(f"candidate seed {seed} exit code is invalid")
        if exit_code != 0:
            failed_seeds.append(seed)
            if (
                fresh.get("candidate_failed") is not True
                or fresh.get("verified") is not False
            ):
                raise ValueError(
                    f"failed candidate seed {seed} fresh disposition drifted"
                )
            continue

        candidate_raw, candidate_wrapper = _canonical_object(
            candidate_package / "result.json"
        )
        _, candidate_sha = _canonical_object(candidate_package / "result.sha256.json")
        if candidate_sha != {"sha256": hashlib.sha256(candidate_raw).hexdigest()}:
            raise ValueError(f"candidate seed {seed} result digest drifted")
        if (
            candidate_wrapper.get("issue") != ISSUE
            or candidate_wrapper.get("seed") != seed
            or candidate_wrapper.get("protocol_sha256") != PROTOCOL_SHA
            or candidate_wrapper.get("unused_data_accessed") is not False
            or candidate_wrapper.get("production_eligible") is not False
            or candidate_wrapper.get("live_trading_authorized") is not False
        ):
            raise ValueError(f"candidate seed {seed} result identity drifted")
        if (
            fresh.get("verified") is not True
            or fresh.get("candidate_failed") is not False
        ):
            raise ValueError(f"candidate seed {seed} fresh verification is not Green")
        if (
            fresh.get("candidate_result_sha256")
            != hashlib.sha256(candidate_raw).hexdigest()
        ):
            raise ValueError(f"candidate seed {seed} fresh result hash drifted")
        _verify_bundle(candidate_package, candidate_wrapper["bundle_digest"])

        candidate = candidate_wrapper["result"]
        diagnostics = candidate_wrapper["diagnostics"]
        if not math.isclose(
            float(diagnostics["net_total_return"]),
            float(candidate["metrics"]["total_return"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"candidate seed {seed} net diagnostic drifted")
        if diagnostics["turnover_total"] != candidate["metrics"]["turnover_total"]:
            raise ValueError(f"candidate seed {seed} turnover diagnostic drifted")
        if diagnostics["total_cost"] != candidate["metrics"]["total_cost"]:
            raise ValueError(f"candidate seed {seed} cost diagnostic drifted")
        if diagnostics["funding_pnl"] != candidate["metrics"]["funding_pnl"]:
            raise ValueError(f"candidate seed {seed} funding diagnostic drifted")
        if diagnostics["gross_total_return"] is not None:
            raise ValueError("gross return must not be invented by Issue 663 harness")
        if diagnostics["gross_total_return_status"] != (
            "not_exposed_by_frozen_issue645_evaluator"
        ):
            raise ValueError("gross-return evidence boundary drifted")
        intent_counts = diagnostics["intent_counts"]
        if (
            sum(int(value) for value in intent_counts.values())
            != diagnostics["decision_count"]
        ):
            raise ValueError(f"candidate seed {seed} intent diagnostics drifted")

        base_pass = passes_screen(candidate, require_positive_years=True)
        stress_pass = bool(base_pass and passes_stress(candidate))
        if (
            fresh.get("base_pass") is not base_pass
            or fresh.get("stress_pass") is not stress_pass
        ):
            raise ValueError(f"candidate seed {seed} fresh screen drifted")

        delta = float(candidate["metrics"]["total_return"]) - float(
            control["metrics"]["total_return"]
        )
        rows.append(
            {
                "seed": seed,
                "control_total_return": control["metrics"]["total_return"],
                "candidate_total_return": candidate["metrics"]["total_return"],
                "paired_delta": delta,
                "candidate_base_pass": base_pass,
                "candidate_stress_pass": stress_pass,
                "control_termination_count": len(control["termination_reasons"]),
                "candidate_termination_count": len(candidate["termination_reasons"]),
                "candidate_turnover_total": candidate["metrics"]["turnover_total"],
                "control_turnover_total": control["metrics"]["turnover_total"],
                "candidate_total_cost": candidate["metrics"]["total_cost"],
                "control_total_cost": control["metrics"]["total_cost"],
                "diagnostics": diagnostics,
            }
        )
        candidate_results.append(candidate)

    complete = not failed_seeds and len(candidate_results) == len(SEEDS)
    if not complete:
        if comparison.get("complete") is not False:
            raise ValueError("incomplete candidate roster was not marked incomplete")
        if comparison.get("failed_seeds") != failed_seeds:
            raise ValueError("incomplete comparison failed-seed roster drifted")
        if comparison.get("decision") is not None:
            raise ValueError("incomplete comparison must not publish a formal decision")
        decision = None
        relative = False
        absolute = False
    else:
        deltas = [float(row["paired_delta"]) for row in rows]
        wins = sum(delta > 0.0 for delta in deltas)
        relative = bool(
            all(record.get("verified") is True for record in fresh_records)
            and wins >= protocol["relative_gate"]["required_seed_wins"]
            and median(deltas) > 0.0
            and all(len(row["returns"]) == 17_544 for row in candidate_results)
            and all(
                0.0
                <= float(row["ledger_max_drawdown"])
                <= protocol["relative_gate"]["max_candidate_drawdown"]
                for row in candidate_results
            )
            and all(
                len(candidate["termination_reasons"])
                <= len(control["termination_reasons"])
                for candidate, control in zip(
                    candidate_results, control_results, strict=True
                )
            )
        )
        base_passes = [
            passes_screen(row, require_positive_years=True) for row in candidate_results
        ]
        stress_passes = [
            base and passes_stress(row)
            for base, row in zip(base_passes, candidate_results, strict=True)
        ]
        median_total = float(
            np.median([row["metrics"]["total_return"] for row in candidate_results])
        )
        median_years = {
            year: float(
                np.median([row["year_returns"][year] for row in candidate_results])
            )
            for year in ("2023", "2024")
        }
        absolute = bool(
            sum(stress_passes)
            >= protocol["absolute_gate"]["ppo_required_qualifying_seeds"]
            and median_total > 0.0
            and all(value > 0.0 for value in median_years.values())
        )
        decision = (
            "KEEP_BASELINE"
            if not relative
            else (
                "PROSPECTIVE_PAPER_REQUIRED"
                if absolute
                else "RELATIVE_IMPROVEMENT_ONLY"
            )
        )
        if comparison.get("complete") is not True:
            raise ValueError("complete comparison was not marked complete")
        if comparison.get("failed_seeds") != []:
            raise ValueError("complete comparison contains failed seeds")
        if comparison.get("seed_rows") != rows:
            raise ValueError(
                "comparison seed rows differ from independent reconstruction"
            )
        if comparison.get("paired_win_count") != wins:
            raise ValueError("comparison paired-win count drifted")
        comparison_median = comparison.get("median_paired_delta")
        if isinstance(comparison_median, bool) or not isinstance(
            comparison_median, (int, float)
        ):
            raise ValueError("comparison median paired delta is not numeric")
        if not math.isclose(
            float(comparison_median),
            float(median(deltas)),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("comparison median paired delta drifted")
        if comparison.get("robust_relative_improvement") is not relative:
            raise ValueError("comparison relative gate drifted")
        if comparison.get("absolute_family_qualified") is not absolute:
            raise ValueError("comparison absolute gate drifted")
        if comparison.get("decision") != decision:
            raise ValueError("comparison decision drifted")

    if comparison.get("unused_data_accessed") is not False:
        raise ValueError("comparison unused-data boundary drifted")
    if comparison.get("final_test_accessed") is not False:
        raise ValueError("comparison final-test boundary drifted")
    if comparison.get("production_eligible") is not False:
        raise ValueError("comparison production boundary drifted")
    if comparison.get("live_trading_authorized") is not False:
        raise ValueError("comparison live boundary drifted")

    output.mkdir(parents=True, exist_ok=False)
    audit_payload = {
        "schema": "issue663_normalization_decision_audit_v1",
        "issue": ISSUE,
        "protocol_sha256": PROTOCOL_SHA,
        "comparison_sha256": hashlib.sha256(comparison_raw).hexdigest(),
        "complete": complete,
        "failed_seeds": failed_seeds,
        "decision": decision,
        "robust_relative_improvement": relative,
        "absolute_family_qualified": absolute,
        "policy_model_refit_performed": False,
        "artifact_only_decision_reconstruction": True,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    with (output / "audit.json").open("xb") as stream:
        stream.write(canonical_json_bytes(audit_payload))
    return audit_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(
        protocol_path=args.protocol,
        candidates=args.candidates,
        fresh_root=args.fresh_root,
        controls=args.controls,
        comparison_path=args.comparison,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
