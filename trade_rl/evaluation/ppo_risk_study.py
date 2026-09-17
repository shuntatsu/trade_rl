"""Exactly-once PPO training-risk comparison against a completed frozen study."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.artifacts import load_market_dataset_artifact
from trade_rl.data.features.price_channels import with_price_channels
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_candidates import ARMS, fit_directional_candidate
from trade_rl.evaluation.directional_selection import select_development_candidates
from trade_rl.evaluation.directional_study import (
    SOURCE_DATASET_ID,
    development_indices,
    expected_protocol,
    reserve_study,
    validate_protocol,
)
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.risk import PreTradeRiskConfig

BASELINE_PROTOCOL = "aa9bf63e43668a2edbc0c8a54a5c6e689a277de4fb9e2af7862892914e783000"
PPO_ARMS = tuple(f"ppo{seed}" for seed in range(5))
MATCHED_RISK = PreTradeRiskConfig(max_gross=0.5, max_abs_weight=0.1, max_turnover=None)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, value: object) -> None:
    encoded = canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(encoded)


def _read_arm(root: Path, arm: str, protocol: dict[str, Any]) -> dict[str, Any]:
    directory = root / arm
    if (directory / "failed.json").exists():
        raise ValueError(f"{arm} failed; complete evidence is required")
    raw = (directory / "result.json").read_bytes()
    if json.loads((directory / "result.sha256.json").read_bytes()) != {
        "sha256": sha256(raw).hexdigest()
    }:
        raise ValueError(f"{arm} result hash mismatch")
    row: dict[str, Any] = json.loads(raw)
    if (
        row["arm"] != arm
        or row["protocol_digest"] != content_digest(protocol)
        or row["provenance"] != protocol["provenance"]
    ):
        raise ValueError(f"{arm} protocol/provenance mismatch")
    if (
        isinstance(protocol.get("training_risk"), dict)
        and row.get("training_risk") != protocol["training_risk"]
    ):
        raise ValueError(f"{arm} training risk mismatch")
    expected_models = (
        {"model.zip"}
        if arm.startswith("ppo")
        else (
            {"model.npz"}
            if arm == "ridge24"
            else {"model.txt"}
            if arm == "lightgbm24"
            else set()
        )
    )
    if set(row["model_sha256"]) != expected_models:
        raise ValueError(f"{arm} model evidence incomplete")
    for filename, digest in row["model_sha256"].items():
        if _digest(directory / filename) != digest:
            raise ValueError(f"{arm} model hash mismatch")
    return row


def _verify_factor_boundary(baseline: Path, protocol: dict[str, Any]) -> None:
    package = Path(__file__).resolve().parents[1]
    allowed = {"strategies/rl/ppo.py", "evaluation/directional_candidates.py"}
    with ZipFile(baseline / "source-snapshot.zip") as source:
        for row in protocol["provenance"]["implementation"]["files"]:
            relative = row["path"]
            original = source.read(f"trade_rl/{relative}")
            if sha256(original).hexdigest() != row["sha256"]:
                raise ValueError("baseline source snapshot hash mismatch")
            if relative not in allowed and ast.dump(ast.parse(original)) != ast.dump(
                ast.parse((package / relative).read_bytes())
            ):
                raise ValueError(f"non-factor production semantics changed: {relative}")


def _baseline(source: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads((root / "protocol.json").read_bytes())
    if content_digest(protocol) != BASELINE_PROTOCOL:
        raise ValueError("not the frozen directional baseline protocol")
    validate_protocol(root, protocol)
    current = expected_protocol(source)
    if (
        current["provenance"]["runtime_environment"]
        != protocol["provenance"]["runtime_environment"]
    ):
        raise ValueError("runtime changed; training risk must be the only factor")
    if {key: value for key, value in current.items() if key != "provenance"} != {
        key: value for key, value in protocol.items() if key != "provenance"
    }:
        raise ValueError("baseline data/config/evaluation contract changed")
    rows = {arm: _read_arm(root, arm, protocol) for arm in ARMS}
    selection = select_development_candidates(rows)
    selection["protocol_digest"] = BASELINE_PROTOCOL
    selection["result_sha256"] = {
        arm: _digest(root / arm / "result.json") for arm in ARMS
    }
    if (root / "selection.json").read_bytes() != canonical_json_bytes(selection):
        raise ValueError("baseline selection is incomplete or inconsistent")
    _verify_factor_boundary(root, protocol)
    return protocol, rows


def risk_protocol(source: Path, baseline: Path) -> dict[str, Any]:
    original, _ = _baseline(source, baseline)
    return {
        "schema": "ppo_training_risk_alignment_v1",
        "baseline_protocol_digest": BASELINE_PROTOCOL,
        "baseline_selection_sha256": _digest(baseline / "selection.json"),
        "baseline_result_sha256": {
            arm: _digest(baseline / arm / "result.json") for arm in ARMS
        },
        "source_contract": {
            key: value for key, value in original.items() if key != "provenance"
        },
        "arms": list(PPO_ARMS),
        "training_risk": asdict(MATCHED_RISK),
        "sole_factor": "training_pretrade_risk_configuration",
        "relative_gate": "4_of_5_paired_return_wins;positive_median_return_delta;all_complete;all_candidate_drawdowns<=0.2;no_new_termination",
        "absolute_gate": original["ppo_qualification"],
        "cost_and_turnover": "diagnostic_only;net_profit_is_the_objective",
        "provenance": build_candidate_run_provenance(),
        "production_eligible": False,
        "unused_data_accessed": False,
    }


def compare_paired_ppo(
    baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if set(candidate) != set(PPO_ARMS):
        raise ValueError("comparison requires exactly five candidate seeds")
    old_family = select_development_candidates(baseline)["families"]["ppo"]
    new_family = select_development_candidates({**baseline, **candidate})["families"][
        "ppo"
    ]
    paired = {
        arm: candidate[arm]["metrics"]["total_return"]
        - baseline[arm]["metrics"]["total_return"]
        for arm in PPO_ARMS
    }
    complete = all(
        len(row["returns"]) == 17544 and row["stop_index"] - row["start_index"] == 17544
        for arm in PPO_ARMS
        for row in (baseline[arm], candidate[arm])
    )
    within_risk = all(
        0 <= candidate[arm]["ledger_max_drawdown"] <= 0.2 for arm in PPO_ARMS
    )
    no_new_termination = all(
        len(candidate[arm]["termination_reasons"])
        <= len(baseline[arm]["termination_reasons"])
        for arm in PPO_ARMS
    )
    improved = bool(
        complete
        and within_risk
        and no_new_termination
        and sum(value > 0 for value in paired.values()) >= 4
        and np.median(list(paired.values())) > 0
    )
    return {
        "schema": "ppo_training_risk_comparison_v1",
        "paired_return_deltas": paired,
        "baseline_family": old_family,
        "candidate_family": new_family,
        "relative_improvement": improved,
        "decision": "PROSPECTIVE_PAPER_REQUIRED"
        if improved and new_family["qualified"]
        else "RELATIVE_IMPROVEMENT_ONLY"
        if improved
        else "KEEP_BASELINE",
        "production_eligible": False,
        "unused_data_accessed": False,
    }


def run_risk_candidate(source: Path, baseline: Path, root: Path, arm: str) -> None:
    if arm not in PPO_ARMS:
        raise ValueError("only the five preregistered PPO seeds are allowed")
    protocol = risk_protocol(source, baseline)
    validate_protocol(root, protocol)
    original = load_market_dataset_artifact(source / "dataset")
    if original.dataset_id != SOURCE_DATASET_ID:
        raise ValueError("source dataset changed")
    dataset = with_price_channels(original)
    config = inspect_study(source / "study").plan.baseline_config
    start, stop = development_indices(dataset)
    directory = root / arm
    directory.mkdir(exist_ok=False)
    _write_once(
        directory / "started.json",
        {"arm": arm, "protocol_digest": content_digest(protocol)},
    )
    try:
        print(f"{arm}: fitting with matched training risk", flush=True)
        factory = fit_directional_candidate(
            arm, dataset, config, directory, ppo_risk_config=MATCHED_RISK
        )
        print(f"{arm}: replaying unchanged development evaluation", flush=True)
        result = evaluate_directional_arm(
            dataset, factory, start_index=start, stop_index=stop
        )
        if result["qualified"]:
            result["stress"] = [
                evaluate_directional_arm(
                    dataset, factory, start_index=start, stop_index=stop, **stress
                )
                for stress in protocol["source_contract"]["stresses"]
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
        validate_protocol(root, risk_protocol(source, baseline))
        result.update(
            arm=arm,
            protocol_digest=content_digest(protocol),
            provenance=protocol["provenance"],
            training_risk=asdict(MATCHED_RISK),
            model_sha256={"model.zip": _digest(directory / "model.zip")},
        )
        _write_once(directory / "result.json", result)
        _write_once(
            directory / "result.sha256.json",
            {"sha256": _digest(directory / "result.json")},
        )
        print(
            f"{arm}: published; return={result['metrics']['total_return']:.6%}; qualified={result['qualified']}",
            flush=True,
        )
    except BaseException as error:
        _write_once(directory / "failed.json", {"arm": arm, "error": repr(error)})
        raise


def finalize_risk_study(source: Path, baseline: Path, root: Path) -> None:
    protocol = risk_protocol(source, baseline)
    validate_protocol(root, protocol)
    _, original = _baseline(source, baseline)
    rows = {arm: _read_arm(root, arm, protocol) for arm in PPO_ARMS}
    report = compare_paired_ppo(original, rows)
    report["protocol_digest"] = content_digest(protocol)
    report["result_sha256"] = {
        arm: _digest(root / arm / "result.json") for arm in PPO_ARMS
    }
    _write_once(root / "comparison.json", report)
    print(f"Published paired comparison: {report['decision']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "finalize"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=PPO_ARMS)
    args = parser.parse_args()
    if args.action == "prepare":
        protocol = risk_protocol(args.source, args.baseline)
        reserve_study(args.output, protocol)
        print(f"Prepared fixed comparison: {content_digest(protocol)}", flush=True)
    elif args.action == "finalize":
        finalize_risk_study(args.source, args.baseline, args.output)
    elif args.arm is None:
        parser.error("run requires --arm")
    else:
        run_risk_candidate(args.source, args.baseline, args.output, args.arm)


if __name__ == "__main__":
    main()
