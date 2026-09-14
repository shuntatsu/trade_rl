"""Independent verifier for Issue #541 Stage A legacy-baseline bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


class StageABridgeError(RuntimeError):
    """Raised when Stage A differs economically from the frozen source baseline."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StageABridgeError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise StageABridgeError(f"JSON object required: {path}")
    return payload


def _require_equal(label: str, left: object, right: object) -> None:
    if left != right:
        raise StageABridgeError(f"{label} differs")


def _evidence_root(root: Path) -> Path:
    return root / "study" / "baseline" / "evidence"


def _plan_without_implementation(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("implementation_digest", None)
    return result


def _provenance_without_bridge_context(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("implementation", None)
    result.pop("implementation_digest", None)
    result.pop("research_context_digest", None)
    return result


def _verify_cash_and_costs(summary: dict[str, Any]) -> None:
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list) or not by_symbol:
        raise StageABridgeError("summary by_symbol is invalid")
    for symbol in by_symbol:
        if not isinstance(symbol, dict):
            raise StageABridgeError("summary symbol entry is invalid")
        strategies = symbol.get("strategies")
        if not isinstance(strategies, list) or not strategies:
            raise StageABridgeError("summary strategy roster is invalid")
        for strategy in strategies:
            if not isinstance(strategy, dict):
                raise StageABridgeError("summary strategy entry is invalid")
            name = strategy.get("name")
            metrics = strategy.get("metrics")
            diagnostics = strategy.get("diagnostics")
            if (
                not isinstance(name, str)
                or not isinstance(metrics, dict)
                or not isinstance(diagnostics, dict)
            ):
                raise StageABridgeError("summary strategy payload is invalid")
            if name == "cash":
                zeros = (
                    strategy.get("fill_count"),
                    metrics.get("n_trades"),
                    metrics.get("rebalance_events"),
                    metrics.get("total_cost"),
                    metrics.get("turnover_total"),
                    metrics.get("total_return"),
                    diagnostics.get("n_trades"),
                    diagnostics.get("rebalance_events"),
                    diagnostics.get("total_cost"),
                    diagnostics.get("turnover_total"),
                )
                if any(value != 0 for value in zeros):
                    raise StageABridgeError("cash zero semantics differ")
                initial = (
                    summary.get("evaluation", {}).get("initial_capital")
                    if isinstance(summary.get("evaluation"), dict)
                    else None
                )
                if (
                    initial is not None
                    and strategy.get("final_portfolio_value") != initial
                ):
                    raise StageABridgeError("cash final portfolio value differs")
            else:
                if (
                    metrics.get("n_trades", 0) <= 0
                    or diagnostics.get("n_trades", 0) <= 0
                ):
                    raise StageABridgeError(f"{name} has no trading activity")
                if (
                    metrics.get("total_cost", 0.0) <= 0.0
                    or diagnostics.get("total_cost", 0.0) <= 0.0
                ):
                    raise StageABridgeError(f"{name} lacks positive trading cost")


def _strategy_payloads(
    summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    by_symbol = summary.get("by_symbol")
    if not isinstance(by_symbol, list):
        raise StageABridgeError("summary by_symbol is invalid")
    for symbol in by_symbol:
        if not isinstance(symbol, dict) or not isinstance(symbol.get("symbol"), str):
            raise StageABridgeError("summary symbol is invalid")
        strategies = symbol.get("strategies")
        if not isinstance(strategies, list):
            raise StageABridgeError("summary strategies are invalid")
        for strategy in strategies:
            if not isinstance(strategy, dict) or not isinstance(
                strategy.get("name"), str
            ):
                raise StageABridgeError("summary strategy is invalid")
            result[(symbol["symbol"], strategy["name"])] = strategy
    return result


def verify_stage_a_bridge(
    original_root: str | Path, replay_root: str | Path
) -> dict[str, object]:
    """Verify that integrated-code legacy replay is economically identical to source."""

    original = Path(original_root)
    replay = Path(replay_root)
    original_plan = _read_json(original / "study" / "plan.json")
    replay_plan = _read_json(replay / "study" / "plan.json")
    _require_equal(
        "Study plan except implementation provenance",
        _plan_without_implementation(original_plan),
        _plan_without_implementation(replay_plan),
    )
    original_plan_impl = original_plan.get("implementation_digest")
    replay_plan_impl = replay_plan.get("implementation_digest")

    original_evidence = _evidence_root(original)
    replay_evidence = _evidence_root(replay)
    original_manifest = _read_json(original_evidence / "manifest.json")
    replay_manifest = _read_json(replay_evidence / "manifest.json")
    for key in (
        "schema_version",
        "ppo_seeds",
        "semantic_config",
        "semantic_config_digest",
    ):
        _require_equal(
            f"evidence manifest {key}",
            original_manifest.get(key),
            replay_manifest.get(key),
        )
    original_context = original_manifest.get("research_context_digest")
    replay_context = replay_manifest.get("research_context_digest")
    if not isinstance(original_context, str) or not isinstance(replay_context, str):
        raise StageABridgeError("evidence research context digest is invalid")
    seeds = original_manifest.get("ppo_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise StageABridgeError("frozen PPO seed roster is invalid")

    raw_arrays_checked = 0
    summaries_checked = 0
    original_implementations: set[str] = set()
    replay_implementations: set[str] = set()
    runtime_digests: set[str] = set()
    deterministic_reference: dict[tuple[str, str], dict[str, Any]] | None = None

    for seed in seeds:
        original_run = original_evidence / "runs" / f"seed-{seed}"
        replay_run = replay_evidence / "runs" / f"seed-{seed}"
        original_summary = _read_json(original_run / "summary.json")
        replay_summary = _read_json(replay_run / "summary.json")
        if original_summary != replay_summary:
            raise StageABridgeError(f"economic summary differs for seed {seed}")
        _verify_cash_and_costs(replay_summary)
        summaries_checked += 1

        with (
            np.load(original_run / "returns.npz", allow_pickle=False) as left,
            np.load(replay_run / "returns.npz", allow_pickle=False) as right,
        ):
            if left.files != right.files:
                raise StageABridgeError(
                    f"raw return key roster differs for seed {seed}"
                )
            for key in left.files:
                left_array = left[key]
                right_array = right[key]
                if (
                    left_array.dtype != right_array.dtype
                    or left_array.shape != right_array.shape
                    or left_array.tobytes() != right_array.tobytes()
                ):
                    raise StageABridgeError(
                        f"raw return array differs for seed {seed}: {key}"
                    )
                raw_arrays_checked += 1

        original_provenance = _read_json(original_run / "provenance.json")
        replay_provenance = _read_json(replay_run / "provenance.json")
        if _provenance_without_bridge_context(
            original_provenance
        ) != _provenance_without_bridge_context(replay_provenance):
            if original_provenance.get(
                "runtime_environment_digest"
            ) != replay_provenance.get("runtime_environment_digest"):
                raise StageABridgeError(f"runtime environment differs for seed {seed}")
            raise StageABridgeError(f"unexplained provenance drift for seed {seed}")
        if original_provenance.get("research_context_digest") != original_context:
            raise StageABridgeError(f"source research context mismatch for seed {seed}")
        if replay_provenance.get("research_context_digest") != replay_context:
            raise StageABridgeError(f"replay research context mismatch for seed {seed}")
        old_impl = original_provenance.get("implementation_digest")
        new_impl = replay_provenance.get("implementation_digest")
        runtime = replay_provenance.get("runtime_environment_digest")
        if (
            not isinstance(old_impl, str)
            or not isinstance(new_impl, str)
            or not isinstance(runtime, str)
        ):
            raise StageABridgeError("provenance digest fields are invalid")
        original_implementations.add(old_impl)
        replay_implementations.add(new_impl)
        runtime_digests.add(runtime)

        payloads = _strategy_payloads(replay_summary)
        deterministic = {
            key: value for key, value in payloads.items() if key[1] != "ppo"
        }
        if deterministic_reference is None:
            deterministic_reference = deterministic
        elif deterministic != deterministic_reference:
            raise StageABridgeError("deterministic strategy seed invariance failed")

    if len(original_implementations) != 1 or len(replay_implementations) != 1:
        raise StageABridgeError(
            "implementation provenance is inconsistent across seeds"
        )
    if len(runtime_digests) != 1:
        raise StageABridgeError("runtime environment is inconsistent across seeds")

    original_analysis = _read_json(original / "study" / "baseline" / "analysis.json")
    replay_analysis = _read_json(replay / "study" / "baseline" / "analysis.json")
    if original_analysis.get("analysis") != replay_analysis.get("analysis"):
        raise StageABridgeError("economic analysis differs")

    original_impl = next(iter(original_implementations))
    replay_impl = next(iter(replay_implementations))
    if original_plan_impl is not None and original_plan_impl != original_impl:
        raise StageABridgeError("source Study/run implementation binding differs")
    if replay_plan_impl is not None and replay_plan_impl != replay_impl:
        raise StageABridgeError("replay Study/run implementation binding differs")

    return {
        "schema_version": "issue541_stage_a_bridge_verification_v2",
        "status": "PASS",
        "raw_return_arrays_checked": raw_arrays_checked,
        "summaries_checked": summaries_checked,
        "ppo_seeds": list(seeds),
        "original_evidence_fingerprint": original_manifest.get("fingerprint"),
        "replay_evidence_fingerprint": replay_manifest.get("fingerprint"),
        "original_implementation_digest": original_impl,
        "replay_implementation_digest": replay_impl,
        "implementation_digest_changed": original_impl != replay_impl,
        "study_plan_implementation_changed": original_plan_impl != replay_plan_impl,
        "research_context_changed": original_context != replay_context,
        "runtime_environment_digest": next(iter(runtime_digests)),
        "runtime_environment_digest_match": True,
        "raw_returns_exact": True,
        "economic_summaries_exact": True,
        "economic_analysis_exact": True,
        "deterministic_seed_invariance": True,
        "cash_zero_semantics": True,
        "positive_trading_costs": True,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify_stage_a_bridge(args.original, args.replay)
    Path(args.output).write_text(
        json.dumps(report, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
