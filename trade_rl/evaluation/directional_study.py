"""Fixed-budget development study CLI; no live order or final-test capability."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.artifacts import load_market_dataset_artifact
from trade_rl.data.features.price_channels import with_price_channels
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_candidates import (
    ARMS,
    PPO_TIMESTEPS,
    fit_directional_candidate,
)
from trade_rl.evaluation.directional_candidates import validate_arm as validate_arm
from trade_rl.evaluation.directional_selection import select_development_candidates
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.runs import build_candidate_run_provenance

SOURCE_DATASET_ID = "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
SOURCE_ARTIFACT_DIGEST = (
    "af481dd978db7d84cd3aa8ff4f5a35d8608ac44c755dd74f61e934105c02b6b7"
)
SOURCE_STUDY_DIGEST = "bfa2fcb307773f5384d7dcb884444164d6d3b575373d8f7dc1c810a61bf4c820"

_REQUIRED_RUNTIME = (
    ("lightgbm", "lightgbm", "4.7.0"),
    ("stable-baselines3", "stable_baselines3", "2.3.2"),
    ("torch", "torch", "2.4.1"),
    ("scikit-learn", "sklearn", None),
)
_OFFICIAL_REF_NAME = "run/issue640-directional-study-v1"
_OFFICIAL_WORKFLOW_REF = (
    "shuntatsu/trade_rl/.github/workflows/issue640-directional-study-v1.yml"
    "@refs/heads/run/issue640-directional-study-v1"
)


def _environment_positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, str) or not value.isdigit():
        raise RuntimeError(f"official activation {field} must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise RuntimeError(f"official activation {field} must be a positive integer")
    return resolved


def official_activation_from_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Accept only the first immutable Issue #640 GitHub Actions push lineage."""

    env: Mapping[str, str] = os.environ if environment is None else environment
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_EVENT_NAME") != "push"
        or env.get("GITHUB_REF_NAME") != _OFFICIAL_REF_NAME
        or env.get("GITHUB_WORKFLOW_REF") != _OFFICIAL_WORKFLOW_REF
    ):
        raise RuntimeError("official activation environment is unavailable or drifted")
    run_id = _environment_positive_int(env.get("GITHUB_RUN_ID"), field="run id")
    run_number = _environment_positive_int(
        env.get("GITHUB_RUN_NUMBER"), field="run number"
    )
    run_attempt = _environment_positive_int(
        env.get("GITHUB_RUN_ATTEMPT"), field="run attempt"
    )
    if run_number != 1 or run_attempt != 1:
        raise RuntimeError("official activation must be run number 1 / attempt 1")
    head_sha = env.get("GITHUB_SHA")
    if (
        not isinstance(head_sha, str)
        or len(head_sha) != 40
        or any(char not in "0123456789abcdef" for char in head_sha)
    ):
        raise RuntimeError("official activation GITHUB_SHA must be an exact commit SHA")
    return {
        "schema_version": "directional_development_activation_v1",
        "workflow_run_id": run_id,
        "workflow_run_number": run_number,
        "workflow_run_attempt": run_attempt,
        "workflow_event": "push",
        "workflow_ref": _OFFICIAL_WORKFLOW_REF,
        "ref_name": _OFFICIAL_REF_NAME,
        "head_sha": head_sha,
    }


def validate_required_runtime() -> dict[str, str]:
    """Fail before study reservation unless every frozen trainer can import."""

    versions: dict[str, str] = {}
    for distribution, module_name, expected_version in _REQUIRED_RUNTIME:
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"{distribution} runtime is unavailable before directional study"
            ) from error
        if expected_version is not None and version != expected_version:
            raise RuntimeError(
                f"{distribution} runtime version {version!r} differs from "
                f"frozen {expected_version!r}"
            )
        try:
            importlib.import_module(module_name)
        except Exception as error:
            raise RuntimeError(
                f"{distribution} runtime is installed but not importable"
            ) from error
        versions[distribution] = version
    return versions


def _write_once(path: Path, payload: object) -> None:
    encoded = canonical_json_bytes(payload)
    with path.open("xb") as stream:
        stream.write(encoded)


def development_indices(dataset: MarketDataset) -> tuple[int, int]:
    start_time = np.datetime64("2023-01-01T00", "ns")
    stop_time = np.datetime64("2025-01-01T00", "ns")
    start = int(np.searchsorted(dataset.timestamps, start_time))
    stop = int(np.searchsorted(dataset.timestamps, stop_time))
    if (
        stop >= dataset.n_bars
        or start >= stop
        or dataset.timestamps[start] != start_time
        or dataset.timestamps[stop] != stop_time
        or stop - start != 17_544
        or not np.all(
            np.diff(dataset.timestamps[start : stop + 1]) == np.timedelta64(1, "h")
        )
    ):
        raise ValueError("development clock differs from preregistration")
    years, counts = np.unique(
        dataset.timestamps[start:stop].astype("datetime64[Y]").astype(str),
        return_counts=True,
    )
    if tuple(zip(years, counts, strict=True)) != (("2023", 8760), ("2024", 8784)):
        raise ValueError("development year coverage differs from preregistration")
    return start, stop


def reserve_study(root: Path, protocol: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=False)
    _write_once(root / "protocol.json", protocol)
    _write_once(root / "protocol.digest.json", {"digest": content_digest(protocol)})


def validate_protocol(root: Path, expected: dict[str, Any]) -> None:
    if (root / "protocol.json").read_bytes() != canonical_json_bytes(expected):
        raise ValueError(
            "study protocol differs from frozen code/source/runtime contract"
        )
    if (root / "protocol.digest.json").read_bytes() != canonical_json_bytes(
        {"digest": content_digest(expected)}
    ):
        raise ValueError("study protocol digest changed")


def prepare_study(source: Path, output: Path) -> None:
    from trade_rl.data.artifacts import inspect_published_market_dataset_artifact

    activation = official_activation_from_environment()
    required_runtime = validate_required_runtime()
    dataset = load_market_dataset_artifact(source / "dataset")
    identity = inspect_published_market_dataset_artifact(source / "dataset")
    if (
        dataset.dataset_id != SOURCE_DATASET_ID
        or identity.artifact_digest != SOURCE_ARTIFACT_DIGEST
    ):
        raise ValueError("source is not the frozen successor artifact")
    development_indices(dataset)
    protocol = expected_protocol(
        source, activation=activation, required_runtime=required_runtime
    )
    reserve_study(output, protocol)
    print(f"Prepared fixed study: {content_digest(protocol)}", flush=True)


def expected_protocol(
    source: Path,
    *,
    activation: dict[str, object] | None = None,
    required_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    resolved_activation = (
        official_activation_from_environment() if activation is None else activation
    )
    resolved_runtime = (
        validate_required_runtime() if required_runtime is None else required_runtime
    )
    plan = inspect_study(source / "study").plan
    if plan.digest != SOURCE_STUDY_DIGEST or plan.dataset_id != SOURCE_DATASET_ID:
        raise ValueError("source StudyPlan is not the frozen successor")
    config = plan.baseline_config
    if np.datetime64(config.fit_cutoff) > np.datetime64("2023-01-01T00"):
        raise ValueError("fit cutoff overlaps development evaluation")
    return {
        "schema": "directional_development_protocol_v1",
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_study_digest": plan.digest,
        "source_plan_sha256": sha256(
            (source / "study" / "plan.json").read_bytes()
        ).hexdigest(),
        "source_config": plan.baseline_config.to_payload(),
        "arms": list(ARMS),
        "ppo_timesteps": PPO_TIMESTEPS,
        "ppo_training_layout": "sequential",
        "initial_capital": 10_000.0,
        "maximum_drawdown": 0.2,
        "deleveraging_start": 0.1,
        "per_symbol_gross": 0.1,
        "account_gross": 0.5,
        "channel_entry_bars": 480,
        "channel_exit_bars": 240,
        "training_risk": "maintained_PPO_defaults_with_0.1_initial_symbol_exposure",
        "screen": "positive_full_and_each_year;maximum_drawdown<=0.2;no_termination;terminal_flat;complete_clock",
        "ppo_qualification": "at_least_4_of_5_seeds_pass_base_and_both_stresses;positive_all_5_seed_median_full_and_year_returns",
        "ranking": "full_return_descending;turnover_ascending;complexity_trend_mean_reversion_channel_breakout_ridge24_lightgbm24_ppo",
        "stresses": [{"cost_multiplier": 2.0}, {"latency_bars": 1}],
        "official_activation": resolved_activation,
        "required_runtime_packages": resolved_runtime,
        "unused_data_accessed": False,
        "production_eligible": False,
        "provenance": build_candidate_run_provenance(),
    }


def execute_arm(source: Path, root: Path, arm: str) -> None:
    validate_arm(arm)
    protocol = expected_protocol(source)
    validate_protocol(root, protocol)
    before = build_candidate_run_provenance()
    if before != protocol["provenance"]:
        raise ValueError("source/runtime changed after study freeze")
    if (
        sha256((source / "study" / "plan.json").read_bytes()).hexdigest()
        != protocol["source_plan_sha256"]
    ):
        raise ValueError("source plan changed after freeze")
    original = load_market_dataset_artifact(source / "dataset")
    if original.dataset_id != SOURCE_DATASET_ID:
        raise ValueError("source dataset changed")
    dataset = with_price_channels(original)
    config = inspect_study(source / "study").plan.baseline_config
    start, stop = development_indices(dataset)
    output = root / arm
    output.mkdir(exist_ok=False)
    _write_once(
        output / "started.json",
        {"arm": arm, "protocol_digest": content_digest(protocol)},
    )
    try:
        print(f"{arm}: fitting", flush=True)
        factory = fit_directional_candidate(arm, dataset, config, output)
        print(f"{arm}: replaying fixed development window", flush=True)
        result = evaluate_directional_arm(
            dataset, factory, start_index=start, stop_index=stop
        )
        if result["qualified"] and arm not in (
            "cash",
            "constant_long",
            "constant_short",
        ):
            result["stress"] = [
                evaluate_directional_arm(
                    dataset, factory, start_index=start, stop_index=stop, **stress
                )
                for stress in protocol["stresses"]
            ]
            result["by_symbol"] = {
                symbol: evaluate_directional_arm(
                    dataset,
                    factory,
                    start_index=start,
                    stop_index=stop,
                    symbol_index=index,
                )
                for index, symbol in enumerate(dataset.symbols)
            }
        if before != build_candidate_run_provenance():
            raise ValueError("source/runtime changed during evaluation")
        validate_protocol(root, expected_protocol(source))
        result["arm"] = arm
        result["protocol_digest"] = content_digest(protocol)
        result["provenance"] = before
        result["model_sha256"] = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in output.glob("model.*")
        }
        _write_once(output / "result.json", result)
        _write_once(
            output / "result.sha256.json",
            {"sha256": sha256((output / "result.json").read_bytes()).hexdigest()},
        )
        print(
            f"{arm}: published; return={result['metrics']['total_return']:.6%}; qualified={result['qualified']}",
            flush=True,
        )
    except BaseException as error:
        _write_once(output / "failed.json", {"arm": arm, "error": repr(error)})
        raise


def finalize_study(source: Path, root: Path) -> None:
    protocol = expected_protocol(source)
    validate_protocol(root, protocol)
    results = {}
    for arm in ARMS:
        directory = root / arm
        if (directory / "failed.json").exists():
            raise ValueError(f"{arm} failed; study is incomplete")
        raw = (directory / "result.json").read_bytes()
        digest = sha256(raw).hexdigest()
        if json.loads((directory / "result.sha256.json").read_bytes()) != {
            "sha256": digest
        }:
            raise ValueError(f"{arm} result digest mismatch")
        result = json.loads(raw)
        if (
            result["arm"] != arm
            or result["protocol_digest"] != content_digest(protocol)
            or result["provenance"] != protocol["provenance"]
        ):
            raise ValueError(f"{arm} result belongs to another protocol")
        for name, model_digest in result["model_sha256"].items():
            if (
                Path(name).name != name
                or sha256((directory / name).read_bytes()).hexdigest() != model_digest
            ):
                raise ValueError(f"{arm} model digest mismatch")
        results[arm] = result
    selection = select_development_candidates(results)
    selection["protocol_digest"] = content_digest(protocol)
    selection["result_sha256"] = {
        arm: sha256((root / arm / "result.json").read_bytes()).hexdigest()
        for arm in ARMS
    }
    _write_once(root / "selection.json", selection)
    print(
        f"Published selection: {selection['decision']}; winner={selection['winner']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_study(args.source, args.output)
        return
    if args.action == "finalize":
        finalize_study(args.source, args.output)
        return
    if args.arm is None:
        parser.error("run requires --arm")
    execute_arm(args.source, args.output, args.arm)


if __name__ == "__main__":
    main()
