from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.evaluation.experiments.ppo_interleaved_evaluation import (
    PPOSeedEvidence,
    PPOReturnPathEvidence,
    PPOSymbolEvidence,
    canonical_ppo_interleaved_evaluator_spec,
    return_path_sha256,
)
from trade_rl.evaluation.experiments.ppo_interleaved_evidence_codec import (
    load_ppo_seed_evidence,
    ppo_seed_evidence_from_payload,
)


def _path(name: str) -> PPOReturnPathEvidence:
    values = (0.0, 0.0)
    return PPOReturnPathEvidence(
        strategy_name=name,
        returns=values,
        return_sha256=return_path_sha256(values),
        total_return=0.0,
        total_cost=0.0,
        turnover_total=0.0,
        max_drawdown=0.0,
        termination_count=0,
        termination_reasons=(),
        n_periods=2,
        periods_per_year=8_760,
    )


def _evidence() -> PPOSeedEvidence:
    spec = canonical_ppo_interleaved_evaluator_spec()
    rows = tuple(
        PPOSymbolEvidence(
            symbol_index=index,
            symbol=symbol,
            ppo=_path("ppo"),
            cash=_path("cash"),
            constant_long=_path("constant_long"),
            constant_short=_path("constant_short"),
            ppo_returns_equal_cash=True,
            ppo_returns_equal_constant_long=True,
            ppo_returns_equal_constant_short=True,
        )
        for index, symbol in enumerate(spec.symbols)
    )
    return PPOSeedEvidence(
        spec_digest=spec.digest,
        protocol_head=spec.protocol_head,
        protocol_digest=spec.protocol_digest,
        implementation_head=spec.implementation_head,
        successor_bundle_run_id=spec.successor_bundle_run_id,
        successor_bundle_artifact_id=spec.successor_bundle_artifact_id,
        successor_bundle_artifact_digest=spec.successor_bundle_artifact_digest,
        dataset_id=spec.dataset_id,
        dataset_artifact_digest=spec.dataset_artifact_digest,
        study_digest=spec.study_digest,
        execution_overlay=spec.execution_overlay,
        symbols=spec.symbols,
        feature_names=spec.feature_names,
        feature_indices=spec.feature_indices,
        fit_symbol_names=spec.fit_symbol_names,
        fit_cutoff=spec.fit_cutoff,
        evaluation_start=spec.evaluation_start,
        evaluation_stop_exclusive=spec.evaluation_stop_exclusive,
        gross_budget=spec.gross_budget,
        initial_capital=spec.initial_capital,
        arm="baseline",
        seed=0,
        training_layout="sequential",
        rollout_steps_per_env=None,
        caller_total_timesteps=spec.ppo_total_timesteps,
        realized_num_timesteps=100_352,
        by_symbol=rows,
    )


def test_canonical_loader_round_trips_exact_seed_evidence(tmp_path: Path) -> None:
    evidence = _evidence()
    path = tmp_path / "seed.json"
    path.write_bytes(canonical_json_bytes(evidence.to_payload()))

    loaded = load_ppo_seed_evidence(path)
    assert loaded == evidence
    assert loaded.digest == evidence.digest


def test_loader_rejects_noncanonical_json_bytes(tmp_path: Path) -> None:
    evidence = _evidence()
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(evidence.to_payload(), indent=2))

    with pytest.raises(ValueError, match="canonical JSON"):
        load_ppo_seed_evidence(path)


def test_loader_rejects_missing_or_unknown_top_level_keys() -> None:
    payload = _evidence().to_payload()
    payload.pop("study_digest")
    with pytest.raises(ValueError, match="keys differ"):
        ppo_seed_evidence_from_payload(payload)

    payload = _evidence().to_payload()
    payload["posthoc_result"] = 123
    with pytest.raises(ValueError, match="keys differ"):
        ppo_seed_evidence_from_payload(payload)


def test_loader_rejects_reordered_or_duplicate_symbol_cells() -> None:
    payload = _evidence().to_payload()
    rows = list(payload["by_symbol"])
    payload["by_symbol"] = list(reversed(rows))
    with pytest.raises(ValueError, match="symbol roster"):
        ppo_seed_evidence_from_payload(payload)

    payload = _evidence().to_payload()
    rows = list(payload["by_symbol"])
    payload["by_symbol"] = [rows[0], rows[0], *rows[2:]]
    with pytest.raises(ValueError, match="symbol roster"):
        ppo_seed_evidence_from_payload(payload)


def test_loader_rejects_bool_or_zero_realized_timestep_evidence() -> None:
    payload = _evidence().to_payload()
    payload["realized_num_timesteps"] = True
    with pytest.raises(ValueError, match="realized_num_timesteps"):
        ppo_seed_evidence_from_payload(payload)

    payload = _evidence().to_payload()
    payload["realized_num_timesteps"] = 0
    with pytest.raises(ValueError, match="realized_num_timesteps"):
        ppo_seed_evidence_from_payload(payload)


def test_loader_rejects_raw_return_hash_or_equality_flag_tampering() -> None:
    payload = _evidence().to_payload()
    rows = list(payload["by_symbol"])
    first = dict(rows[0])
    ppo = dict(first["ppo"])
    ppo["return_sha256"] = "f" * 64
    first["ppo"] = ppo
    rows[0] = first
    payload["by_symbol"] = rows
    with pytest.raises(ValueError, match="return_sha256"):
        ppo_seed_evidence_from_payload(payload)

    payload = _evidence().to_payload()
    rows = list(payload["by_symbol"])
    first = dict(rows[0])
    first["ppo_returns_equal_cash"] = False
    rows[0] = first
    payload["by_symbol"] = rows
    with pytest.raises(ValueError, match="cash equality"):
        ppo_seed_evidence_from_payload(payload)


def test_loader_rejects_bool_alias_inside_raw_returns() -> None:
    payload = _evidence().to_payload()
    rows = list(payload["by_symbol"])
    first = dict(rows[0])
    ppo = dict(first["ppo"])
    ppo["returns"] = [True, 0.0]
    first["ppo"] = ppo
    rows[0] = first
    payload["by_symbol"] = rows
    with pytest.raises(ValueError, match="returns"):
        ppo_seed_evidence_from_payload(payload)
