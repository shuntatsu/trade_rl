from __future__ import annotations

import json

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.bootstrap.ridge_economic_gate_evaluation import (
    RidgeEconomicGateEvaluation,
    RidgeEconomicGateSymbolResult,
    canonical_ridge_economic_gate_evaluation_bytes,
    canonical_ridge_economic_gate_evaluation_spec,
    load_ridge_economic_gate_evaluation,
)

_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")


def _row(symbol: str, index: int) -> RidgeEconomicGateSymbolResult:
    baseline = 0.10 + index * 0.01
    candidate = baseline + 0.01
    return RidgeEconomicGateSymbolResult(
        symbol=symbol,
        baseline_total_return=baseline,
        candidate_total_return=candidate,
        excess_total_return=candidate - baseline,
        baseline_total_cost=10.0 + index,
        candidate_total_cost=8.0 + index,
        baseline_turnover_total=100.0 + index,
        candidate_turnover_total=80.0 + index,
        baseline_max_drawdown=0.20 + index * 0.001,
        candidate_max_drawdown=0.19 + index * 0.001,
        baseline_termination_count=0,
        candidate_termination_count=0,
        baseline_termination_reasons=(),
        candidate_termination_reasons=(),
        new_termination=False,
        baseline_n_periods=2,
        candidate_n_periods=2,
        baseline_return_sha256=f"{index + 1:x}" * 64,
        candidate_return_sha256=f"{index + 6:x}" * 64,
    )


def _result() -> RidgeEconomicGateEvaluation:
    spec = canonical_ridge_economic_gate_evaluation_spec()
    rows = tuple(_row(symbol, index) for index, symbol in enumerate(_SYMBOLS))
    return RidgeEconomicGateEvaluation(
        spec_digest=spec.digest,
        dataset_id=spec.dataset_id,
        model_sha256="c" * 64,
        symbols=_SYMBOLS,
        by_symbol=rows,
        positive_effect_symbols=5,
        median_excess_total_return=0.01,
        cost_reduction_symbols=5,
        turnover_reduction_symbols=5,
        drawdown_nonworse_symbols=5,
        new_termination_symbols=0,
        candidate_positive_total_return_symbols=5,
        research_status="PROMOTE_RESEARCH_REFERENCE",
    )


def _canonical_document_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(document, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def test_canonical_result_round_trip_is_byte_stable(tmp_path) -> None:
    result = _result()
    raw = canonical_ridge_economic_gate_evaluation_bytes(result)
    path = tmp_path / "result.json"
    path.write_bytes(raw)

    loaded = load_ridge_economic_gate_evaluation(path)

    assert loaded == result
    assert canonical_ridge_economic_gate_evaluation_bytes(loaded) == raw


def test_loader_rejects_unknown_nested_field_even_with_resigned_digest(tmp_path) -> None:
    document = json.loads(canonical_ridge_economic_gate_evaluation_bytes(_result()))
    document["result"]["unexpected"] = 1
    document["content_digest"] = content_digest(document["result"])
    path = tmp_path / "result.json"
    path.write_bytes(_canonical_document_bytes(document))

    with pytest.raises(ValueError, match="result keys differ"):
        load_ridge_economic_gate_evaluation(path)


def test_loader_rejects_resigned_aggregate_and_status_forgery(tmp_path) -> None:
    document = json.loads(canonical_ridge_economic_gate_evaluation_bytes(_result()))
    document["result"]["positive_effect_symbols"] = 4
    document["content_digest"] = content_digest(document["result"])
    path = tmp_path / "result.json"
    path.write_bytes(_canonical_document_bytes(document))

    with pytest.raises(ValueError, match="positive_effect_symbols"):
        load_ridge_economic_gate_evaluation(path)

    document = json.loads(canonical_ridge_economic_gate_evaluation_bytes(_result()))
    document["result"]["research_status"] = "INCONCLUSIVE"
    document["content_digest"] = content_digest(document["result"])
    path.write_bytes(_canonical_document_bytes(document))
    with pytest.raises(ValueError, match="research_status"):
        load_ridge_economic_gate_evaluation(path)


def test_loader_rejects_bool_int_spoof_even_with_resigned_digest(tmp_path) -> None:
    document = json.loads(canonical_ridge_economic_gate_evaluation_bytes(_result()))
    document["result"]["positive_effect_symbols"] = True
    document["content_digest"] = content_digest(document["result"])
    path = tmp_path / "result.json"
    path.write_bytes(_canonical_document_bytes(document))

    with pytest.raises(ValueError, match="non-negative integer"):
        load_ridge_economic_gate_evaluation(path)


def test_loader_rejects_noncanonical_json_bytes(tmp_path) -> None:
    document = json.loads(canonical_ridge_economic_gate_evaluation_bytes(_result()))
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="canonical"):
        load_ridge_economic_gate_evaluation(path)
