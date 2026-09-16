"""Exactly-once Issue 610 candidate execution and deterministic interpretation.

This module is intentionally outside :mod:`trade_rl`. Economic computation is
performed by the sealed Issue 607 package. The only cross-version bridge is the
Issue 609 candidate-artifact serializer/loader, loaded as an external module so
that exact Issue 607 implementation provenance remains unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np

import trade_rl.evaluation.experiments.evidence as evidence_module
from tools.tmp_issue610_ppo_global_btc_regime_evaluation import (
    EXPECTED_SEMANTIC_CHANGED_FIELDS,
    build_candidate_carrier_plan,
    candidate_run_config_from_resolved,
    evaluate_frozen_gate,
    execute_candidate_after_precompute_gate,
    validate_candidate_ppo_returns,
    validate_controlled_semantic_delta,
    validate_unaffected_raw_returns,
)
from tools.tmp_issue610_ppo_global_btc_regime_termination import (
    validate_no_new_ppo_terminations,
)
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import (
    inspect_published_market_dataset_artifact,
    load_market_dataset_artifact,
)
from trade_rl.evaluation.comparison.strategies import compare_strategies_by_symbol
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.experiments.analysis import compare_evidence_sets
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import (
    CandidateRunConfig,
    build_candidate_run_provenance,
)
from trade_rl.evaluation.runs.execute import CandidateRunResult
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rl.ppo import (
    PPO_GLOBAL_BTC_REGIME_CONTEXT,
    PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
    ppo_observation_contract_payload,
)

ISSUE_NUMBER = 610
IMPLEMENTATION_SHA = "1ae494ce182189c5be61e8007dc956f912d8b9d5"
IMPLEMENTATION_TREE = "4f14c5a06505b8ff9811b473bf6dea296453a13b"
HELPER_SHA = "358b4917e988561f9a5601ef25a6d53291e07585"
ARTIFACT_BRIDGE_SHA = "ff332f0cc2dec46828aa1031d438f2f2f1705ba8"
ARTIFACT_BRIDGE_BLOB = "8f95053b62239d674c18ed8564ffe4fc2c1ceaf6"
PRECOMPUTE_RUN_ID = 35098793118
PRECOMPUTE_SCHEMA = "issue610_precompute_authority_v1"
DATASET_ID = "d7a04ede97a1bb37b811c3e071f325fa007525a6040927e6793d8cc7c10f538f"
DATASET_ARTIFACT_DIGEST = (
    "77362e148c713840dda64e0ef70e663cce6611407eac31fefbb9fccca73ae8f8"
)
SOURCE_STUDY_DIGEST = "3d8404061a4082a8e9b3c786d9f5fc9a4347631dff39c201e3cba70470dfeb79"
BASELINE_EVIDENCE_FINGERPRINT = (
    "526b485d60b394739b7a8d03535e1cfa08b26fa920119c70ee53f05faa53dd29"
)
EVALUATION_PROTOCOL_DIGEST = (
    "e52a19b859500ea01e900a1ea7209bd545c03acd08435d1ff615b7539ec75d2b"
)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
SEEDS = (0, 1, 2, 3, 4)
UNAFFECTED_STRATEGIES = (
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "cash",
    "constant_long",
    "constant_short",
)


def _load_canonical_object(path: Path) -> tuple[dict[str, object], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise ValueError(f"canonical object required: {path}")
    if canonical_json_bytes(payload) != raw:
        raise ValueError(f"non-canonical JSON: {path}")
    unsigned = dict(payload)
    digest = unsigned.pop("content_digest", None)
    if not isinstance(digest, str) or content_digest(unsigned) != digest:
        raise ValueError(f"content digest mismatch: {path}")
    return cast(dict[str, object], payload), raw


def validate_precompute_authority(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Return all result-blind authority violations without touching economics."""

    expected: dict[str, object] = {
        "schema_version": PRECOMPUTE_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "precompute_run_id": PRECOMPUTE_RUN_ID,
        "implementation_git_sha": IMPLEMENTATION_SHA,
        "implementation_tree_sha": IMPLEMENTATION_TREE,
        "helper_git_sha": HELPER_SHA,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "source_study_digest": SOURCE_STUDY_DIGEST,
        "baseline_evidence_fingerprint": BASELINE_EVIDENCE_FINGERPRINT,
        "evaluation_protocol_digest": EVALUATION_PROTOCOL_DIGEST,
        "prior_issue610_economic_artifacts": 0,
        "precompute_inputs_validated": True,
        "economic_publisher_requires_run_number_one": True,
        "economic_publisher_requires_attempt_one": True,
        "baseline_retrained": False,
        "candidate_training_performed": False,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    violations: list[str] = []
    for field, value in expected.items():
        if type(payload.get(field)) is not type(value) or payload.get(field) != value:
            violations.append(f"precompute authority mismatch: {field}")
    changed = payload.get("expected_semantic_changed_paths")
    expected_changed = [list(path) for path in EXPECTED_SEMANTIC_CHANGED_FIELDS]
    if changed != expected_changed:
        violations.append("precompute semantic changed paths mismatch")
    unaffected = payload.get("unaffected_strategy_names")
    if unaffected != list(UNAFFECTED_STRATEGIES):
        violations.append("precompute unaffected strategy roster mismatch")
    for field in (
        "implementation_digest",
        "runtime_environment_digest",
        "candidate_carrier_study_digest",
        "candidate_resolved_config_digest",
        "candidate_requested_config_digest",
        "content_digest",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or len(value) != 64:
            violations.append(f"precompute digest malformed: {field}")
    return tuple(violations)


def install_artifact_bridge(artifact_module_path: Path) -> ModuleType:
    """Install only the sealed Issue 609 artifact contract into Issue 607 evidence."""

    if not artifact_module_path.is_file():
        raise FileNotFoundError(artifact_module_path)
    spec = importlib.util.spec_from_file_location(
        "issue610_external_artifact_contract",
        artifact_module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load external artifact contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in (
        "publish_candidate_run",
        "load_candidate_run_artifact",
        "inspect_candidate_run_artifact",
    ):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"external artifact contract missing callable: {name}")
    evidence_module.publish_candidate_run = module.publish_candidate_run  # type: ignore[attr-defined]
    evidence_module.load_candidate_run_artifact = module.load_candidate_run_artifact  # type: ignore[attr-defined]
    evidence_module.inspect_candidate_run_artifact = (
        module.inspect_candidate_run_artifact
    )  # type: ignore[attr-defined]
    return module


def _candidate_contract(
    source_plan: Any, precompute: Mapping[str, object]
) -> tuple[Any, CandidateRunConfig, Any]:
    provenance = build_candidate_run_provenance()
    if provenance.get("implementation_digest") != precompute.get(
        "implementation_digest"
    ):
        raise RuntimeError(
            "current Issue 607 implementation digest differs from precompute"
        )
    if provenance.get("runtime_environment_digest") != precompute.get(
        "runtime_environment_digest"
    ):
        raise RuntimeError("current runtime environment digest differs from precompute")

    resolved = replace(
        source_plan.baseline_config,
        schema_version="resolved_run_config_v3",
        ppo_observation_schema=PPO_GLOBAL_BTC_REGIME_OBSERVATION_SCHEMA,
        ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
    )
    baseline_semantic = source_plan.baseline_config.to_payload()
    candidate_semantic = resolved.to_payload()
    baseline_semantic.pop("ppo_seed")
    candidate_semantic.pop("ppo_seed")
    changed, violations = validate_controlled_semantic_delta(
        baseline_semantic,
        candidate_semantic,
    )
    if violations or changed != EXPECTED_SEMANTIC_CHANGED_FIELDS:
        raise RuntimeError(f"candidate semantic delta is not sealed: {violations!r}")

    executable = candidate_run_config_from_resolved(resolved)
    carrier = build_candidate_carrier_plan(
        source_plan=source_plan,
        candidate_config=resolved,
        implementation_digest=str(provenance["implementation_digest"]),
        runtime_environment_digest=str(provenance["runtime_environment_digest"]),
    )
    if carrier.digest != precompute.get("candidate_carrier_study_digest"):
        raise RuntimeError(
            "candidate carrier StudyPlan differs from precompute authority"
        )
    if content_digest(resolved.to_payload()) != precompute.get(
        "candidate_resolved_config_digest"
    ):
        raise RuntimeError(
            "candidate resolved config differs from precompute authority"
        )
    if content_digest(executable.to_json_payload()) != precompute.get(
        "candidate_requested_config_digest"
    ):
        raise RuntimeError(
            "candidate executable config differs from precompute authority"
        )
    return resolved, executable, carrier


def _strict_source(source_root: Path) -> tuple[Any, Any, Any]:
    dataset = load_market_dataset_artifact(source_root / "dataset")
    dataset_artifact = inspect_published_market_dataset_artifact(
        source_root / "dataset"
    )
    snapshot = inspect_study(source_root / "study")
    baseline = evidence_module.load_evidence_set(
        source_root / "study" / "baseline" / "evidence"
    )
    if dataset.dataset_id != DATASET_ID:
        raise RuntimeError("source dataset id drift")
    if dataset_artifact.artifact_digest != DATASET_ARTIFACT_DIGEST:
        raise RuntimeError("source dataset artifact digest drift")
    if tuple(dataset.symbols) != SYMBOLS:
        raise RuntimeError("source symbol roster/order drift")
    if snapshot.plan.digest != SOURCE_STUDY_DIGEST:
        raise RuntimeError("source Study digest drift")
    if baseline.evidence.fingerprint != BASELINE_EVIDENCE_FINGERPRINT:
        raise RuntimeError("source baseline EvidenceSet drift")
    if baseline.evidence.ppo_seeds != SEEDS:
        raise RuntimeError("source baseline seed roster drift")
    return dataset, snapshot, baseline


def _run_return_matrix(loaded: Any) -> dict[int, dict[tuple[str, str], np.ndarray]]:
    matrix: dict[int, dict[tuple[str, str], np.ndarray]] = {}
    for seed in SEEDS:
        run = loaded.runs.get(seed)
        if run is None:
            continue
        symbols_raw = run.summary.get("symbols")
        by_symbol = run.summary.get("by_symbol")
        if symbols_raw != list(SYMBOLS) or not isinstance(by_symbol, list):
            continue
        cells: dict[tuple[str, str], np.ndarray] = {}
        for symbol, symbol_entry in zip(SYMBOLS, by_symbol, strict=True):
            if (
                not isinstance(symbol_entry, dict)
                or symbol_entry.get("symbol") != symbol
            ):
                continue
            strategies = symbol_entry.get("strategies")
            if not isinstance(strategies, list):
                continue
            for strategy in strategies:
                if not isinstance(strategy, dict):
                    continue
                name = strategy.get("name")
                key = strategy.get("return_key")
                if (
                    isinstance(name, str)
                    and isinstance(key, str)
                    and key in run.returns
                ):
                    cells[(symbol, name)] = run.returns[key]
        matrix[seed] = cells
    return matrix


def termination_violations(baseline: Any, candidate: Any) -> tuple[str, ...]:
    """Apply the independently verified fail-closed termination oracle."""

    return validate_no_new_ppo_terminations(
        baseline.runs,
        candidate.runs,
        seeds=SEEDS,
        symbols=SYMBOLS,
    )


def execute_candidate(
    *,
    source_root: Path,
    precompute_path: Path,
    output_root: Path,
    artifact_module_path: Path,
    execution_run_id: int,
    precompute_artifact_id: int,
    precompute_artifact_digest: str,
) -> dict[str, object]:
    """Execute the candidate exactly once after strict result-blind gates."""

    install_artifact_bridge(artifact_module_path)
    precompute, _ = _load_canonical_object(precompute_path)
    violations = validate_precompute_authority(precompute)
    if violations:
        raise RuntimeError("; ".join(violations))
    dataset, snapshot, _baseline = _strict_source(source_root)
    resolved, executable, carrier = _candidate_contract(snapshot.plan, precompute)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("candidate output root is not empty")
    output_root.mkdir(parents=True, exist_ok=True)

    research_context_digest = str(precompute["content_digest"])
    store = StudyStore(output_root)
    evidence = execute_candidate_after_precompute_gate(
        precompute_violations=(),
        candidate_executor=lambda: evidence_module.execute_evidence_set(
            store=store,
            target=Path("evidence"),
            dataset_root=source_root / "dataset",
            plan=carrier,
            config=executable,
            research_context_digest=research_context_digest,
        ),
    )
    loaded = evidence_module.load_evidence_set(output_root / "evidence")
    if loaded.evidence.fingerprint != evidence.fingerprint:
        raise RuntimeError("candidate EvidenceSet reload fingerprint mismatch")
    expected_semantic = resolved.to_payload()
    expected_semantic.pop("ppo_seed")
    if loaded.semantic_config != expected_semantic:
        raise RuntimeError("candidate EvidenceSet semantic config mismatch")
    ppo_violations = validate_candidate_ppo_returns(
        _run_return_matrix(loaded),
        seeds=SEEDS,
        symbols=SYMBOLS,
    )
    if ppo_violations:
        raise RuntimeError(
            "candidate PPO raw evidence invalid: " + "; ".join(ppo_violations)
        )
    expected_observation = ppo_observation_contract_payload(
        global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT
    )
    for seed in SEEDS:
        run = loaded.runs[seed]
        config = run.summary.get("candidate_config")
        if (
            not isinstance(config, dict)
            or config.get("ppo_global_context") != PPO_GLOBAL_BTC_REGIME_CONTEXT
        ):
            raise RuntimeError(f"candidate run global context missing: seed={seed}")
        if run.summary.get("ppo_observation") != expected_observation:
            raise RuntimeError(f"candidate observation authority mismatch: seed={seed}")
        if run.provenance.get("implementation_digest") != precompute.get(
            "implementation_digest"
        ):
            raise RuntimeError(
                f"candidate implementation provenance mismatch: seed={seed}"
            )
        if run.provenance.get("runtime_environment_digest") != precompute.get(
            "runtime_environment_digest"
        ):
            raise RuntimeError(f"candidate runtime provenance mismatch: seed={seed}")
        if run.provenance.get("research_context_digest") != research_context_digest:
            raise RuntimeError(f"candidate research context mismatch: seed={seed}")

    authority: dict[str, object] = {
        "schema_version": "issue610_candidate_execution_v1",
        "issue_number": ISSUE_NUMBER,
        "execution_run_id": execution_run_id,
        "precompute_run_id": PRECOMPUTE_RUN_ID,
        "precompute_artifact_id": precompute_artifact_id,
        "precompute_artifact_api_digest": precompute_artifact_digest,
        "precompute_authority_content_digest": precompute["content_digest"],
        "implementation_git_sha": IMPLEMENTATION_SHA,
        "implementation_tree_sha": IMPLEMENTATION_TREE,
        "helper_git_sha": HELPER_SHA,
        "artifact_bridge_git_sha": ARTIFACT_BRIDGE_SHA,
        "artifact_bridge_blob_sha1": ARTIFACT_BRIDGE_BLOB,
        "implementation_digest": precompute["implementation_digest"],
        "runtime_environment_digest": precompute["runtime_environment_digest"],
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "source_study_digest": SOURCE_STUDY_DIGEST,
        "baseline_evidence_fingerprint": BASELINE_EVIDENCE_FINGERPRINT,
        "candidate_carrier_study_digest": carrier.digest,
        "candidate_evidence_fingerprint": evidence.fingerprint,
        "candidate_semantic_config_digest": evidence.semantic_config_digest,
        "research_context_digest": research_context_digest,
        "ppo_seeds": list(SEEDS),
        "symbols": list(SYMBOLS),
        "exactly_once_run_number_required": 1,
        "exactly_once_run_attempt_required": 1,
        "baseline_retrained": False,
        "candidate_training_performed": True,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    authority["content_digest"] = content_digest(authority)
    (output_root / "candidate-authority.json").write_bytes(
        canonical_json_bytes(authority)
    )
    print("ISSUE610_CANDIDATE_EXECUTION=PASS")
    print("CANDIDATE_EVIDENCE_FINGERPRINT=" + evidence.fingerprint)
    print("BASELINE_RETRAINED=false")
    print("CANDIDATE_TRAINING_PERFORMED=true")
    print("ECONOMIC_VALUES_INTERPRETED=false")
    print("FINAL_TEST_ACCESSED=false")
    return authority


def interpret_candidate(
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
    """Deterministically apply the sealed Issue 609 development gate."""

    install_artifact_bridge(artifact_module_path)
    precompute, _ = _load_canonical_object(precompute_path)
    precompute_violations = validate_precompute_authority(precompute)
    if precompute_violations:
        raise RuntimeError("; ".join(precompute_violations))
    _dataset, snapshot, baseline = _strict_source(source_root)
    candidate_authority, _ = _load_canonical_object(
        candidate_root / "candidate-authority.json"
    )
    if candidate_authority.get("candidate_training_performed") is not True:
        raise RuntimeError("candidate authority does not prove candidate training")
    if candidate_authority.get("economic_values_interpreted") is not False:
        raise RuntimeError("candidate artifact was already economically interpreted")
    if candidate_authority.get("precompute_authority_content_digest") != precompute.get(
        "content_digest"
    ):
        raise RuntimeError("candidate/precompute authority binding mismatch")
    candidate = evidence_module.load_evidence_set(candidate_root / "evidence")
    if candidate.evidence.fingerprint != candidate_authority.get(
        "candidate_evidence_fingerprint"
    ):
        raise RuntimeError("candidate EvidenceSet authority mismatch")

    baseline_semantic = dict(baseline.semantic_config)
    candidate_semantic = dict(candidate.semantic_config)
    changed_paths, semantic_violations = validate_controlled_semantic_delta(
        baseline_semantic,
        candidate_semantic,
    )
    baseline_matrix = _run_return_matrix(baseline)
    candidate_matrix = _run_return_matrix(candidate)
    unaffected_violations = validate_unaffected_raw_returns(
        baseline_matrix,
        candidate_matrix,
        seeds=SEEDS,
        symbols=SYMBOLS,
        unaffected_strategies=UNAFFECTED_STRATEGIES,
    )
    ppo_violations = validate_candidate_ppo_returns(
        candidate_matrix,
        seeds=SEEDS,
        symbols=SYMBOLS,
    )
    termination_errors = termination_violations(baseline, candidate)

    comparison = compare_evidence_sets(
        baseline.runs,
        candidate.runs,
        n_bootstrap=snapshot.plan.n_bootstrap,
        bootstrap_seed=snapshot.plan.bootstrap_seed,
        schema_version="controlled_evidence_comparison_v2",
    )
    recorded_comparison_digest = comparison.get("analysis_digest")
    comparison_unsigned = dict(comparison)
    comparison_unsigned.pop("analysis_digest", None)
    if recorded_comparison_digest != content_digest(comparison_unsigned):
        raise RuntimeError("comparison analysis digest mismatch")

    validity_violations = tuple(semantic_violations) + tuple(ppo_violations)
    gate = evaluate_frozen_gate(
        comparison,
        seeds=SEEDS,
        symbols=SYMBOLS,
        no_new_termination=not termination_errors,
        unaffected_raw_returns_equal=not unaffected_violations,
        validity_violations=validity_violations,
    )
    if changed_paths != EXPECTED_SEMANTIC_CHANGED_FIELDS:
        raise RuntimeError("interpreted semantic changed paths drift")
    if unaffected_violations:
        # Raw-return invariance is an INVALID condition under the sealed protocol.
        gate = evaluate_frozen_gate(
            comparison,
            seeds=SEEDS,
            symbols=SYMBOLS,
            no_new_termination=not termination_errors,
            unaffected_raw_returns_equal=False,
            validity_violations=validity_violations,
        )

    cross_symbol = comparison.get("cross_symbol")
    if not isinstance(cross_symbol, dict) or not isinstance(
        cross_symbol.get("ppo"), dict
    ):
        raise RuntimeError("comparison PPO cross-symbol diagnostics missing")
    ppo_diagnostic = cast(dict[str, object], cross_symbol["ppo"])

    result: dict[str, object] = {
        "schema_version": "issue610_development_evaluation_result_v1",
        "issue_number": ISSUE_NUMBER,
        "interpretation_run_id": interpretation_run_id,
        "precompute_run_id": PRECOMPUTE_RUN_ID,
        "precompute_artifact_id": precompute_artifact_id,
        "precompute_artifact_api_digest": precompute_artifact_digest,
        "precompute_authority_content_digest": precompute["content_digest"],
        "candidate_artifact_id": candidate_artifact_id,
        "candidate_artifact_api_digest": candidate_artifact_digest,
        "candidate_authority_content_digest": candidate_authority["content_digest"],
        "candidate_evidence_fingerprint": candidate.evidence.fingerprint,
        "baseline_evidence_fingerprint": baseline.evidence.fingerprint,
        "comparison_schema": comparison["schema_version"],
        "comparison_digest": recorded_comparison_digest,
        "comparison": comparison,
        "semantic_changed_paths": [list(path) for path in changed_paths],
        "semantic_violations": list(semantic_violations),
        "candidate_ppo_violations": list(ppo_violations),
        "unaffected_raw_return_violations": list(unaffected_violations),
        "termination_violations": list(termination_errors),
        "decision_gate": gate,
        "decision": gate["decision"],
        "absolute_candidate_profitability_diagnostic": ppo_diagnostic,
        "development_acceptance_establishes_profitability": False,
        "baseline_retrained": False,
        "candidate_retrained_during_interpretation": False,
        "economic_values_interpreted": True,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    result["content_digest"] = content_digest(result)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_bytes(canonical_json_bytes(result))
    print("ISSUE610_INTERPRETATION=PASS")
    print("DECISION=" + str(gate["decision"]))
    print("FINAL_TEST_ACCESSED=false")
    print("PRODUCTION_ELIGIBLE=false")
    print("LIVE_TRADING_AUTHORIZED=false")
    return result


def synthetic_bridge_check(artifact_module_path: Path, output_root: Path) -> None:
    """Cross-version serializer/loader oracle without PPO training or market data."""

    module = install_artifact_bridge(artifact_module_path)
    n = 8
    close = np.asarray(
        [
            [100.0, 100.0],
            [100.0, 100.0],
            [105.0, 95.0],
            [110.0, 90.0],
            [115.0, 85.0],
            [120.0, 80.0],
            [125.0, 75.0],
            [130.0, 70.0],
        ]
    )
    signal = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    features = np.stack((signal, -signal), axis=1).reshape(n, 2, 1)
    from trade_rl.data.market import MarketDataset
    from trade_rl.evaluation.runs.config import resolve_candidate_run_spec

    dataset = MarketDataset(
        dataset_id="b" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n, 2), 1_000_000.0),
        funding_rate=np.zeros((n, 2)),
        tradable=np.ones((n, 2), dtype=np.bool_),
        feature_available=np.ones((n, 2, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )
    config = CandidateRunConfig(
        signal_name="signal",
        feature_names=("signal",),
        fit_symbol_names=("BTCUSDT",),
        fit_cutoff=np.datetime64("2026-01-01T04:00:00", "ns"),
        evaluation_start=np.datetime64("2026-01-01T04:00:00", "ns"),
        evaluation_stop_exclusive=np.datetime64("2026-01-01T07:00:00", "ns"),
        rule_entry_threshold=0.10,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=256,
        ppo_seed=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
        ppo_global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT,
    )
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema="market_dataset_artifact_v3",
        dataset_artifact_digest="d" * 64,
        config=config,
    )
    comparison = compare_strategies_by_symbol(
        dataset,
        {"cash": ConstantIntentStrategy(PositionIntent.FLAT)},
        start_index=4,
        stop_index=7,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )
    result = CandidateRunResult(
        spec=spec, symbols=tuple(dataset.symbols), comparison=comparison
    )
    artifact = module.publish_candidate_run(
        output_root / "candidate",
        result,
        build_candidate_run_provenance(research_context_digest="e" * 64),
    )
    summary = json.loads(artifact.summary_path.read_text(encoding="utf-8"))
    if (
        summary["candidate_config"]["ppo_global_context"]
        != PPO_GLOBAL_BTC_REGIME_CONTEXT
    ):
        raise RuntimeError("bridge did not persist global context")
    if summary["ppo_observation"] != ppo_observation_contract_payload(
        global_context=PPO_GLOBAL_BTC_REGIME_CONTEXT
    ):
        raise RuntimeError("bridge did not persist exact observation contract")
    loaded = module.load_candidate_run_artifact(artifact.root)
    if loaded.summary != summary:
        raise RuntimeError("bridge round-trip changed summary")
    summary["ppo_observation"] = ppo_observation_contract_payload()
    artifact.summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    try:
        module.load_candidate_run_artifact(artifact.root)
    except ValueError as error:
        if "PPO observation contract mismatch" not in str(error):
            raise
    else:
        raise RuntimeError("bridge accepted forged baseline observation contract")
    print("ISSUE610_ARTIFACT_BRIDGE_SYNTHETIC=PASS")
    print("CANDIDATE_TRAINING_PERFORMED=false")


def self_check() -> None:
    payload: dict[str, object] = {
        "schema_version": PRECOMPUTE_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "precompute_run_id": PRECOMPUTE_RUN_ID,
        "implementation_git_sha": IMPLEMENTATION_SHA,
        "implementation_tree_sha": IMPLEMENTATION_TREE,
        "helper_git_sha": HELPER_SHA,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "source_study_digest": SOURCE_STUDY_DIGEST,
        "baseline_evidence_fingerprint": BASELINE_EVIDENCE_FINGERPRINT,
        "evaluation_protocol_digest": EVALUATION_PROTOCOL_DIGEST,
        "prior_issue610_economic_artifacts": 0,
        "precompute_inputs_validated": True,
        "economic_publisher_requires_run_number_one": True,
        "economic_publisher_requires_attempt_one": True,
        "baseline_retrained": False,
        "candidate_training_performed": False,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
        "expected_semantic_changed_paths": [
            list(path) for path in EXPECTED_SEMANTIC_CHANGED_FIELDS
        ],
        "unaffected_strategy_names": list(UNAFFECTED_STRATEGIES),
        "implementation_digest": "a" * 64,
        "runtime_environment_digest": "b" * 64,
        "candidate_carrier_study_digest": "c" * 64,
        "candidate_resolved_config_digest": "d" * 64,
        "candidate_requested_config_digest": "e" * 64,
        "content_digest": "f" * 64,
    }
    if validate_precompute_authority(payload):
        raise RuntimeError("valid precompute fixture was rejected")
    bad = dict(payload)
    bad["candidate_training_performed"] = True
    if (
        "precompute authority mismatch: candidate_training_performed"
        not in validate_precompute_authority(bad)
    ):
        raise RuntimeError("invalid precompute fixture was accepted")

    baseline_entry = {
        "metrics": {"termination_count": 1},
        "diagnostics": {"termination_reasons": ["risk_limit"]},
    }
    candidate_entry = {
        "metrics": {"termination_count": 1},
        "diagnostics": {"termination_reasons": ["risk_limit"]},
    }
    fake_runs = {
        seed: SimpleNamespace(
            summary={
                "by_symbol": [
                    {
                        "symbol": symbol,
                        "strategies": [{"name": "ppo", **candidate_entry}],
                    }
                    for symbol in SYMBOLS
                ]
            }
        )
        for seed in SEEDS
    }
    baseline_runs = {
        seed: SimpleNamespace(
            summary={
                "by_symbol": [
                    {
                        "symbol": symbol,
                        "strategies": [{"name": "ppo", **baseline_entry}],
                    }
                    for symbol in SYMBOLS
                ]
            }
        )
        for seed in SEEDS
    }
    if termination_violations(
        SimpleNamespace(runs=baseline_runs), SimpleNamespace(runs=fake_runs)
    ):
        raise RuntimeError("equal termination evidence was rejected")
    fake_runs[0].summary["by_symbol"][0]["strategies"][0]["metrics"][
        "termination_count"
    ] = 2
    if not termination_violations(
        SimpleNamespace(runs=baseline_runs), SimpleNamespace(runs=fake_runs)
    ):
        raise RuntimeError("new termination evidence was accepted")
    print("ISSUE610_ORCHESTRATOR_SELF_CHECK=PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    bridge = sub.add_parser("bridge-check")
    bridge.add_argument("--artifact-module", type=Path, required=True)
    bridge.add_argument("--output-root", type=Path, required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--precompute", type=Path, required=True)
    execute.add_argument("--output-root", type=Path, required=True)
    execute.add_argument("--artifact-module", type=Path, required=True)
    execute.add_argument("--execution-run-id", type=int, required=True)
    execute.add_argument("--precompute-artifact-id", type=int, required=True)
    execute.add_argument("--precompute-artifact-digest", required=True)
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
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "self-check":
        self_check()
    elif args.command == "bridge-check":
        synthetic_bridge_check(args.artifact_module, args.output_root)
    elif args.command == "execute":
        execute_candidate(
            source_root=args.source_root,
            precompute_path=args.precompute,
            output_root=args.output_root,
            artifact_module_path=args.artifact_module,
            execution_run_id=args.execution_run_id,
            precompute_artifact_id=args.precompute_artifact_id,
            precompute_artifact_digest=args.precompute_artifact_digest,
        )
    elif args.command == "interpret":
        interpret_candidate(
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
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
