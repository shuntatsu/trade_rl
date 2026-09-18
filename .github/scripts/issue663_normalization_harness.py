"""Exactly-once Issue 663 normalized-PPO research harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.artifacts import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.data.features.price_channels import with_price_channels
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_selection import passes_screen, passes_stress
from trade_rl.evaluation.directional_study import development_indices
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.metrics import compound_return
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import PPOIntentStrategy, fit_ppo_strategy
from trade_rl.strategies.rl.ppo_artifact import (
    load_normalized_ppo,
    save_normalized_ppo,
)
from trade_rl.strategies.rl.ppo_normalization import fit_ppo_feature_normalizer

ISSUE = 663
TARGET_SHA = "5cefa36ca3ba47cc9ed9b80df18130ec2720f86e"
PROTOCOL_SHA = "de6179570dae339c4a3fe3d62d076edc811c332372185efc17d1566e7602fd55"
SEEDS = (0, 1, 2, 3, 4)


def _canonical_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"canonical JSON object required: {path}")
    return raw, value


def _write_once(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    raw, protocol = _canonical_object(path)
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA:
        raise ValueError("Issue 663 protocol SHA differs from sealed v2")
    if (
        protocol.get("issue") != ISSUE
        or protocol.get("factor") != "PPO_FIT_ONLY_FEATURE_STANDARDIZATION"
        or protocol["candidate"]["normalize_features"] is not True
        or protocol["candidate"]["seeds"] != list(SEEDS)
        or protocol["candidate"]["total_timesteps"] != 262_144
        or protocol["candidate"]["training_layout"] != "sequential"
        or protocol["candidate"]["risk_config"] is not None
        or protocol["absolute_gate"].get("terminal_flatness_semantics")
        != "issue645_reporting_quantity_abs_le_1e-10"
        or "exact_terminal_flatness" in protocol["absolute_gate"]
    ):
        raise ValueError("Issue 663 sealed protocol semantics drifted")
    if any(protocol["boundaries"].values()):
        raise ValueError("Issue 663 sealed protocol boundary flags drifted")
    return protocol


def _load_source(source: Path, protocol: dict[str, Any]):
    dataset = load_market_dataset_artifact(source / "dataset")
    identity = inspect_published_market_dataset_artifact(source / "dataset")
    expected = protocol["dataset"]
    if (
        dataset.dataset_id != expected["dataset_id"]
        or identity.artifact_digest != expected["dataset_artifact_digest"]
    ):
        raise ValueError("Dataset authority differs from Issue 663 protocol")
    plan = inspect_study(source / "study").plan
    if plan.digest != expected["study_digest"]:
        raise ValueError("Study authority differs from Issue 663 protocol")
    enriched = with_price_channels(dataset)
    start, stop = development_indices(enriched)
    if stop - start != protocol["window"]["evaluation_intervals"]:
        raise ValueError("development interval count differs from Issue 663 protocol")
    cutoff = (
        int(
            np.searchsorted(
                enriched.timestamps,
                np.datetime64(plan.baseline_config.fit_cutoff),
            )
        )
        - 1
    )
    if cutoff <= 0 or cutoff >= start:
        raise ValueError("PPO fit scope overlaps development evaluation")
    return enriched, plan.baseline_config, start, stop, cutoff


class _CountingStrategy:
    def __init__(
        self,
        strategy: SingleSymbolStrategy,
        counter: Counter[str],
    ) -> None:
        self.strategy = strategy
        self.counter = counter

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        intent = self.strategy.decide(observation)
        self.counter[intent.name] += 1
        return intent


def _strategy_factory(strategy: PPOIntentStrategy):
    def factory() -> PPOIntentStrategy:
        return PPOIntentStrategy(
            strategy.policy,
            feature_indices=strategy.feature_indices,
            feature_normalizer=strategy.feature_normalizer,
        )

    return factory


def run_seed(
    *,
    seed: int,
    source: Path,
    protocol_path: Path,
    output: Path,
) -> None:
    if seed not in SEEDS:
        raise ValueError("seed is outside frozen Issue 663 roster")
    protocol = load_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    _write_once(
        output / "started.json",
        {
            "schema": "issue663_normalization_seed_started_v1",
            "issue": ISSUE,
            "seed": seed,
            "protocol_sha256": PROTOCOL_SHA,
            "implementation_head": TARGET_SHA,
            "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
            "workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
            "candidate_result_inspected_before_fit": False,
            "unused_data_accessed": False,
        },
    )

    try:
        dataset, config, start, stop, cutoff = _load_source(source, protocol)
        before = build_candidate_run_provenance()
        import torch

        torch.set_num_threads(1)
        strategy = fit_ppo_strategy(
            dataset,
            feature_indices=config.feature_indices,
            fit_symbol_indices=config.fit_symbol_indices,
            start_index=0,
            stop_index=cutoff,
            gross_budget=protocol["candidate"]["gross_training_budget"],
            total_timesteps=protocol["candidate"]["total_timesteps"],
            seed=seed,
            initial_capital=protocol["candidate"]["initial_capital"],
            execution_cost=replace(
                ExecutionCostConfig.zero(),
                processing_bar_volume_capacity=False,
            ),
            training_layout=protocol["candidate"]["training_layout"],
            normalize_features=True,
        )
        if strategy.feature_normalizer is None:
            raise RuntimeError("normalized candidate lacks fitted normalizer")

        bundle = output / "bundle"
        bundle_digest = save_normalized_ppo(bundle, strategy)
        loaded = load_normalized_ppo(
            bundle,
            expected_digest=bundle_digest,
            feature_names=dataset.feature_names,
        )
        counter: Counter[str] = Counter()

        def base_factory() -> SingleSymbolStrategy:
            return _CountingStrategy(_strategy_factory(loaded)(), counter)

        base = evaluate_directional_arm(
            dataset,
            base_factory,
            start_index=start,
            stop_index=stop,
        )
        base_counts = {name: int(counter.get(name, 0)) for name in ("SHORT", "FLAT", "LONG")}
        decision_count = sum(base_counts.values())
        dominant_intent = max(base_counts, key=base_counts.get) if decision_count else None
        dominant_fraction = (
            max(base_counts.values()) / decision_count if decision_count else None
        )

        if base["qualified"]:
            factory = _strategy_factory(loaded)
            stresses = (
                {
                    "cost_multiplier": protocol["absolute_gate"][
                        "cost_stress_multiplier"
                    ]
                },
                {
                    "latency_bars": protocol["absolute_gate"][
                        "latency_stress_bars"
                    ]
                },
            )
            base["stress"] = [
                evaluate_directional_arm(
                    dataset,
                    factory,
                    start_index=start,
                    stop_index=stop,
                    **stress,
                )
                for stress in stresses
            ]
            base["by_symbol"] = {
                symbol: evaluate_directional_arm(
                    dataset,
                    factory,
                    start_index=start,
                    stop_index=stop,
                    symbol_index=index,
                )
                for index, symbol in enumerate(dataset.symbols)
            }

        after = build_candidate_run_provenance()
        if after != before:
            raise ValueError("candidate implementation/runtime changed during seed execution")

        diagnostics = {
            "intent_counts": base_counts,
            "decision_count": decision_count,
            "dominant_intent": dominant_intent,
            "dominant_fraction": dominant_fraction,
            "net_total_return": base["metrics"]["total_return"],
            "turnover_total": base["metrics"]["turnover_total"],
            "total_cost": base["metrics"]["total_cost"],
            "funding_pnl": base["metrics"]["funding_pnl"],
            "borrow_cost": base["metrics"]["borrow_cost"],
            "gross_total_return": None,
            "gross_total_return_status": "not_exposed_by_frozen_issue645_evaluator",
        }
        result = {
            "schema": "issue663_normalization_seed_result_v1",
            "issue": ISSUE,
            "seed": seed,
            "protocol_sha256": PROTOCOL_SHA,
            "implementation_head": TARGET_SHA,
            "bundle_digest": bundle_digest,
            "result": base,
            "diagnostics": diagnostics,
            "provenance": before,
            "model_fit_performed": True,
            "unused_data_accessed": False,
            "final_test_accessed": False,
            "production_eligible": False,
            "live_trading_authorized": False,
        }
        _write_once(output / "result.json", result)
        _write_once(output / "result.sha256.json", {"sha256": _sha(output / "result.json")})
        _write_once(
            output / "attempt.json",
            {
                "schema": "issue663_normalization_seed_attempt_v1",
                "issue": ISSUE,
                "seed": seed,
                "exit_code": 0,
                "result_written": True,
                "failure_written": False,
                "model_fit_may_have_started": True,
                "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
                "workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
                "unused_data_accessed": False,
                "production_eligible": False,
                "live_trading_authorized": False,
            },
        )
    except BaseException as error:
        if output.exists() and not (output / "failed.json").exists():
            _write_once(
                output / "failed.json",
                {
                    "schema": "issue663_normalization_seed_failure_v1",
                    "issue": ISSUE,
                    "seed": seed,
                    "error": repr(error),
                    "model_fit_may_have_started": True,
                    "unused_data_accessed": False,
                    "production_eligible": False,
                    "live_trading_authorized": False,
                },
            )
        if output.exists() and not (output / "attempt.json").exists():
            _write_once(
                output / "attempt.json",
                {
                    "schema": "issue663_normalization_seed_attempt_v1",
                    "issue": ISSUE,
                    "seed": seed,
                    "exit_code": 1,
                    "result_written": (output / "result.json").exists(),
                    "failure_written": (output / "failed.json").exists(),
                    "model_fit_may_have_started": True,
                    "workflow_run_id": int(os.environ["GITHUB_RUN_ID"]),
                    "workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
                    "unused_data_accessed": False,
                    "production_eligible": False,
                    "live_trading_authorized": False,
                },
            )
        raise


def fresh_verify(
    *,
    seed: int,
    source: Path,
    protocol_path: Path,
    candidate: Path,
    output: Path,
) -> None:
    if seed not in SEEDS:
        raise ValueError("seed is outside frozen Issue 663 roster")
    protocol = load_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    _, attempt = _canonical_object(candidate / "attempt.json")
    if attempt.get("seed") != seed:
        raise ValueError("candidate attempt seed mismatch")
    if attempt.get("exit_code") != 0:
        _write_once(
            output / "fresh.json",
            {
                "schema": "issue663_normalization_seed_fresh_v1",
                "seed": seed,
                "verified": False,
                "candidate_failed": True,
                "policy_model_refit_performed": False,
                "unused_data_accessed": False,
                "production_eligible": False,
            },
        )
        return

    result_raw, wrapper = _canonical_object(candidate / "result.json")
    _, recorded_sha = _canonical_object(candidate / "result.sha256.json")
    if recorded_sha != {"sha256": hashlib.sha256(result_raw).hexdigest()}:
        raise ValueError("candidate result digest mismatch")
    if (
        wrapper.get("schema") != "issue663_normalization_seed_result_v1"
        or wrapper.get("seed") != seed
        or wrapper.get("protocol_sha256") != PROTOCOL_SHA
        or wrapper.get("implementation_head") != TARGET_SHA
    ):
        raise ValueError("candidate result identity drifted")

    dataset, config, start, stop, cutoff = _load_source(source, protocol)
    loaded = load_normalized_ppo(
        candidate / "bundle",
        expected_digest=wrapper["bundle_digest"],
        feature_names=dataset.feature_names,
    )
    recomputed = fit_ppo_feature_normalizer(
        dataset,
        feature_indices=config.feature_indices,
        fit_symbol_indices=config.fit_symbol_indices,
        start_index=0,
        stop_index=cutoff,
    )
    if (
        loaded.feature_normalizer is None
        or loaded.feature_normalizer.to_payload() != recomputed.to_payload()
    ):
        raise ValueError("published normalizer differs from fresh fit-scope reconstruction")

    base = wrapper["result"]
    returns = tuple(float(value) for value in base["returns"])
    if len(returns) != protocol["window"]["evaluation_intervals"]:
        raise ValueError("candidate replay is incomplete")
    total_return = compound_return(returns)
    if not math.isclose(
        total_return,
        float(base["metrics"]["total_return"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("candidate total return differs from raw returns")
    years = dataset.timestamps[start:stop].astype("datetime64[Y]").astype(str)
    array = np.asarray(returns, dtype=np.float64)
    year_returns = {
        year: compound_return(tuple(float(x) for x in array[years == year]))
        for year in ("2023", "2024")
    }
    for year, value in year_returns.items():
        if not math.isclose(
            value,
            float(base["year_returns"][year]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"candidate {year} return differs from raw returns")
    terminal_flat = bool(
        np.all(np.abs(np.asarray(base["terminal_quantities"], dtype=np.float64)) <= 1e-10)
    )
    if terminal_flat is not base["terminal_flat"]:
        raise ValueError("candidate terminal-flat flag differs from #645 semantics")

    base_pass = passes_screen(base, require_positive_years=True)
    stress_pass = bool(base_pass and passes_stress(base))
    fresh = {
        "schema": "issue663_normalization_seed_fresh_v1",
        "issue": ISSUE,
        "seed": seed,
        "verified": True,
        "candidate_failed": False,
        "candidate_result_sha256": hashlib.sha256(result_raw).hexdigest(),
        "bundle_digest": wrapper["bundle_digest"],
        "normalizer_payload_digest": content_digest(recomputed.to_payload()),
        "base_pass": base_pass,
        "stress_pass": stress_pass,
        "policy_model_refit_performed": False,
        "normalizer_recomputed_from_frozen_fit_scope": True,
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    _write_once(output / "fresh.json", fresh)


def _load_candidate(root: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    package = root / f"issue663-normalization-seed{seed}-v1"
    _, attempt = _canonical_object(package / "attempt.json")
    if attempt.get("exit_code") != 0:
        raise ValueError(f"candidate seed {seed} failed and cannot enter comparison")
    _, result = _canonical_object(package / "result.json")
    return attempt, result


def _load_fresh(root: Path, seed: int) -> dict[str, Any]:
    package = root / f"issue663-normalization-seed{seed}-fresh-v1"
    _, fresh = _canonical_object(package / "fresh.json")
    return fresh


def _load_control(root: Path, seed: int) -> dict[str, Any]:
    package = root / f"seed{seed}"
    raw, result = _canonical_object(package / "result.json")
    _, recorded = _canonical_object(package / "result.sha256.json")
    if recorded != {"sha256": hashlib.sha256(raw).hexdigest()}:
        raise ValueError(f"control seed {seed} result digest mismatch")
    if result.get("arm") != f"ppo{seed}":
        raise ValueError(f"control seed {seed} arm mismatch")
    return result


def compare(
    *,
    protocol_path: Path,
    candidates: Path,
    fresh_root: Path,
    controls: Path,
    output: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    rows = []
    candidate_rows = []
    controls_rows = []
    verified = True
    for seed in SEEDS:
        _attempt, candidate = _load_candidate(candidates, seed)
        fresh = _load_fresh(fresh_root, seed)
        control = _load_control(controls, seed)
        verified = verified and fresh.get("verified") is True
        base = candidate["result"]
        candidate_rows.append(base)
        controls_rows.append(control)
        rows.append(
            {
                "seed": seed,
                "control_total_return": control["metrics"]["total_return"],
                "candidate_total_return": base["metrics"]["total_return"],
                "paired_delta": (
                    base["metrics"]["total_return"] - control["metrics"]["total_return"]
                ),
                "candidate_base_pass": fresh["base_pass"],
                "candidate_stress_pass": fresh["stress_pass"],
                "control_termination_count": len(control["termination_reasons"]),
                "candidate_termination_count": len(base["termination_reasons"]),
                "candidate_turnover_total": base["metrics"]["turnover_total"],
                "control_turnover_total": control["metrics"]["turnover_total"],
                "candidate_total_cost": base["metrics"]["total_cost"],
                "control_total_cost": control["metrics"]["total_cost"],
                "diagnostics": candidate["diagnostics"],
            }
        )

    deltas = [float(row["paired_delta"]) for row in rows]
    wins = sum(delta > 0.0 for delta in deltas)
    relative = bool(
        verified
        and wins >= protocol["relative_gate"]["required_seed_wins"]
        and median(deltas) > 0.0
        and all(len(row["returns"]) == 17_544 for row in candidate_rows)
        and all(
            0.0 <= float(row["ledger_max_drawdown"])
            <= protocol["relative_gate"]["max_candidate_drawdown"]
            for row in candidate_rows
        )
        and all(
            len(candidate["termination_reasons"])
            <= len(control["termination_reasons"])
            for candidate, control in zip(candidate_rows, controls_rows, strict=True)
        )
    )
    base_passes = [passes_screen(row, require_positive_years=True) for row in candidate_rows]
    stress_passes = [
        base and passes_stress(row)
        for base, row in zip(base_passes, candidate_rows, strict=True)
    ]
    median_total = float(
        np.median([row["metrics"]["total_return"] for row in candidate_rows])
    )
    median_years = {
        year: float(np.median([row["year_returns"][year] for row in candidate_rows]))
        for year in ("2023", "2024")
    }
    absolute = bool(
        sum(stress_passes)
        >= protocol["absolute_gate"]["ppo_required_qualifying_seeds"]
        and median_total > 0.0
        and all(value > 0.0 for value in median_years.values())
    )
    if not relative:
        decision = "KEEP_BASELINE"
    elif absolute:
        decision = "PROSPECTIVE_PAPER_REQUIRED"
    else:
        decision = "RELATIVE_IMPROVEMENT_ONLY"

    output.mkdir(parents=True, exist_ok=False)
    comparison = {
        "schema": "issue663_normalization_comparison_v1",
        "issue": ISSUE,
        "protocol_sha256": PROTOCOL_SHA,
        "seed_rows": rows,
        "paired_win_count": wins,
        "median_paired_delta": float(median(deltas)),
        "robust_relative_improvement": relative,
        "candidate_base_pass_count": sum(base_passes),
        "candidate_stress_pass_count": sum(stress_passes),
        "candidate_family_median_total_return": median_total,
        "candidate_family_median_year_returns": median_years,
        "absolute_family_qualified": absolute,
        "decision": decision,
        "mandatory_diagnostics": {
            "candidate_median_turnover": float(
                np.median([row["metrics"]["turnover_total"] for row in candidate_rows])
            ),
            "control_median_turnover": float(
                np.median([row["metrics"]["turnover_total"] for row in controls_rows])
            ),
            "candidate_median_cost": float(
                np.median([row["metrics"]["total_cost"] for row in candidate_rows])
            ),
            "control_median_cost": float(
                np.median([row["metrics"]["total_cost"] for row in controls_rows])
            ),
            "gross_return_status": "not_exposed_by_frozen_issue645_evaluator",
        },
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }
    _write_once(output / "comparison.json", comparison)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-seed")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)

    fresh = sub.add_parser("fresh")
    fresh.add_argument("--seed", type=int, required=True)
    fresh.add_argument("--source", type=Path, required=True)
    fresh.add_argument("--protocol", type=Path, required=True)
    fresh.add_argument("--candidate", type=Path, required=True)
    fresh.add_argument("--output", type=Path, required=True)

    comparison = sub.add_parser("compare")
    comparison.add_argument("--protocol", type=Path, required=True)
    comparison.add_argument("--candidates", type=Path, required=True)
    comparison.add_argument("--fresh-root", type=Path, required=True)
    comparison.add_argument("--controls", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "run-seed":
        run_seed(
            seed=args.seed,
            source=args.source,
            protocol_path=args.protocol,
            output=args.output,
        )
    elif args.command == "fresh":
        fresh_verify(
            seed=args.seed,
            source=args.source,
            protocol_path=args.protocol,
            candidate=args.candidate,
            output=args.output,
        )
    else:
        compare(
            protocol_path=args.protocol,
            candidates=args.candidates,
            fresh_root=args.fresh_root,
            controls=args.controls,
            output=args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
