from __future__ import annotations

from statistics import median

import pytest

from tools.tmp_issue610_real_interpretation_audit_v1 import (
    SEEDS,
    SYMBOLS,
    _validate_absolute_diagnostic,
    _validate_candidate_authority,
)


def _v2_with_symbols(symbols: tuple[str, ...]) -> dict[str, object]:
    medians = {symbol: 0.01 * (index + 1) for index, symbol in enumerate(SYMBOLS)}
    ordered = {symbol: medians[symbol] for symbol in symbols if symbol in medians}
    by_symbol: dict[str, object] = {
        symbol: {"median_candidate_total_return": value}
        for symbol, value in ordered.items()
    }
    values = list(ordered.values())
    diagnostic: dict[str, object] = {
        "schema_version": "issue610_absolute_candidate_ppo_diagnostic_v1",
        "gates_development_decision": False,
        "by_symbol": by_symbol,
        "cross_symbol": {
            "symbol_count": len(values),
            "positive_symbol_count": sum(value > 0.0 for value in values),
            "negative_symbol_count": sum(value < 0.0 for value in values),
            "zero_symbol_count": sum(value == 0.0 for value in values),
            "median_candidate_total_return": float(median(values)),
            "worst_candidate_total_return": min(values),
            "best_candidate_total_return": max(values),
        },
    }
    return {"absolute_candidate_profitability_diagnostic": diagnostic}


def test_canonical_sorted_symbol_keys_are_accepted() -> None:
    canonical_key_order = tuple(sorted(SYMBOLS))
    _validate_absolute_diagnostic(_v2_with_symbols(canonical_key_order))


def test_missing_symbol_is_rejected() -> None:
    payload = _v2_with_symbols(tuple(sorted(SYMBOLS))[:-1])
    with pytest.raises(RuntimeError, match="absolute diagnostic symbol roster drift"):
        _validate_absolute_diagnostic(payload)


def test_extra_symbol_is_rejected() -> None:
    payload = _v2_with_symbols(tuple(sorted(SYMBOLS)))
    diagnostic = payload["absolute_candidate_profitability_diagnostic"]
    assert isinstance(diagnostic, dict)
    by_symbol = diagnostic["by_symbol"]
    assert isinstance(by_symbol, dict)
    by_symbol["ZZZUSDT"] = {"median_candidate_total_return": 0.0}
    with pytest.raises(RuntimeError, match="absolute diagnostic symbol roster drift"):
        _validate_absolute_diagnostic(payload)


_VALID_API_DIGEST = "sha256:" + "a" * 64
_PRECOMPUTE_CONTENT_DIGEST = "b" * 64


def _candidate_authority() -> dict[str, object]:
    return {
        "issue_number": 610,
        "execution_run_id": 35103004952,
        "precompute_run_id": 35098793118,
        "precompute_artifact_id": 10447243668,
        "precompute_artifact_api_digest": _VALID_API_DIGEST,
        "precompute_authority_content_digest": _PRECOMPUTE_CONTENT_DIGEST,
        "ppo_seeds": list(SEEDS),
        "symbols": list(SYMBOLS),
        "baseline_retrained": False,
        "candidate_training_performed": True,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }


def _validate_candidate(candidate: dict[str, object]) -> None:
    _validate_candidate_authority(
        candidate,
        precompute={"content_digest": _PRECOMPUTE_CONTENT_DIGEST},
        candidate_execution_run_id=35103004952,
        candidate_artifact_id=10453782249,
        candidate_artifact_digest=_VALID_API_DIGEST,
        precompute_artifact_id=10447243668,
        precompute_artifact_digest=_VALID_API_DIGEST,
        producer_result_artifact_id=10455728296,
        producer_result_artifact_digest=_VALID_API_DIGEST,
        fresh_result_artifact_id=10454484504,
        fresh_result_artifact_digest=_VALID_API_DIGEST,
    )


def test_candidate_authority_contract_is_accepted() -> None:
    _validate_candidate(_candidate_authority())


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("issue_number", 611, "candidate issue drift"),
        ("execution_run_id", 1, "candidate execution run mismatch"),
        ("precompute_run_id", 1, "candidate precompute run mismatch"),
        ("precompute_artifact_id", 1, "candidate precompute artifact id mismatch"),
        (
            "precompute_artifact_api_digest",
            "sha256:" + "c" * 64,
            "candidate precompute artifact digest mismatch",
        ),
        (
            "precompute_authority_content_digest",
            "c" * 64,
            "candidate/precompute content binding mismatch",
        ),
        ("ppo_seeds", [0, 1, 2, 3], "candidate seed roster drift"),
        ("symbols", list(reversed(SYMBOLS)), "candidate symbol roster drift"),
        ("baseline_retrained", True, "candidate boundary mismatch: baseline_retrained"),
        (
            "candidate_training_performed",
            False,
            "candidate boundary mismatch: candidate_training_performed",
        ),
        (
            "economic_values_interpreted",
            True,
            "candidate boundary mismatch: economic_values_interpreted",
        ),
        (
            "final_test_accessed",
            True,
            "candidate boundary mismatch: final_test_accessed",
        ),
        (
            "operational_eligibility_established",
            True,
            "candidate boundary mismatch: operational_eligibility_established",
        ),
        (
            "production_eligible",
            True,
            "candidate boundary mismatch: production_eligible",
        ),
        (
            "live_trading_authorized",
            True,
            "candidate boundary mismatch: live_trading_authorized",
        ),
        ("merge_authorized", True, "candidate boundary mismatch: merge_authorized"),
    ),
)
def test_candidate_authority_drift_is_rejected(
    field: str, replacement: object, message: str
) -> None:
    candidate = _candidate_authority()
    candidate[field] = replacement
    with pytest.raises(RuntimeError, match=message):
        _validate_candidate(candidate)


def test_noncanonical_api_digest_is_rejected() -> None:
    candidate = _candidate_authority()
    with pytest.raises(RuntimeError, match="candidate artifact API digest malformed"):
        _validate_candidate_authority(
            candidate,
            precompute={"content_digest": _PRECOMPUTE_CONTENT_DIGEST},
            candidate_execution_run_id=35103004952,
            candidate_artifact_id=10453782249,
            candidate_artifact_digest="a" * 64,
            precompute_artifact_id=10447243668,
            precompute_artifact_digest=_VALID_API_DIGEST,
            producer_result_artifact_id=10455728296,
            producer_result_artifact_digest=_VALID_API_DIGEST,
            fresh_result_artifact_id=10454484504,
            fresh_result_artifact_digest=_VALID_API_DIGEST,
        )
